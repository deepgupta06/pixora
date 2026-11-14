"""Reading originals: content hash, EXIF, thumbnails. Originals are never written."""

from __future__ import annotations

import hashlib
import io
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
# Large panoramas are legitimate; the default limit rejects them.
Image.MAX_IMAGE_PIXELS = 500_000_000

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})
# RAW files are never developed: only the preview the camera embedded is used.
RAW_EXTENSIONS = frozenset(
    {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
)
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | RAW_EXTENSIONS
# Files up to this size are read once and both hashed and decoded from memory.
IN_MEMORY_LIMIT = 128 * 1024 * 1024

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825


@dataclass
class ImageInfo:
    width: int
    height: int
    taken_at: str
    taken_at_source: str  # "exif" or "file"
    lat: float | None
    lon: float | None
    camera: str | None
    perceptual_hash: str


def is_supported(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS


def is_raw(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in RAW_EXTENSIONS


def _new_hash():
    return hashlib.blake2b(digest_size=16)


def read_and_hash(path: str, size: int) -> tuple[str, bytes | None]:
    """Return the content hash, plus the file bytes when small enough to keep."""
    if size <= IN_MEMORY_LIMIT:
        with open(path, "rb") as f:
            data = f.read()
        h = _new_hash()
        h.update(data)
        return h.hexdigest(), data
    h = _new_hash()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest(), None


def _parse_exif_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip().rstrip("\x00")[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _gps_coordinate(values: object, ref: object, positive: str, limit: float) -> float | None:
    try:
        d, m, s = (float(v) for v in values)  # type: ignore[union-attr]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    coord = d + m / 60 + s / 3600
    if not math.isfinite(coord) or coord > limit:
        return None
    if isinstance(ref, str) and ref.strip().upper()[:1] not in ("", positive):
        coord = -coord
    return coord


def _read_exif(img: Image.Image) -> tuple[str | None, float | None, float | None, str | None]:
    try:
        exif = img.getexif()
        sub = exif.get_ifd(_EXIF_IFD)
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:
        return None, None, None, None

    taken = (
        _parse_exif_date(sub.get(36867))  # DateTimeOriginal
        or _parse_exif_date(sub.get(36868))  # DateTimeDigitized
        or _parse_exif_date(exif.get(306))  # DateTime
    )

    lat = _gps_coordinate(gps.get(2), gps.get(1), "N", 90)
    lon = _gps_coordinate(gps.get(4), gps.get(3), "E", 180)
    if lat is None or lon is None or (lat == 0 and lon == 0):
        lat = lon = None

    parts = [str(exif.get(tag) or "").strip().rstrip("\x00") for tag in (271, 272)]  # Make, Model
    make, model = parts
    if make and model.lower().startswith(make.lower()):
        make = ""
    camera = " ".join(p for p in (make, model) if p) or None
    return taken, lat, lon, camera


def _dhash(img: Image.Image) -> str:
    small = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = small.tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def _save_webp(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{id(img)}.tmp")
    # method=1 encodes about 3x faster than the default for roughly 20% larger files.
    img.save(tmp, "WEBP", quality=80, method=1)
    os.replace(tmp, dest)


# LibRaw's flip code -> the rotation that turns the sensor image upright.
_RAW_ROTATIONS = {
    3: Image.Transpose.ROTATE_180,
    5: Image.Transpose.ROTATE_90,
    6: Image.Transpose.ROTATE_270,
}


def _open_raw_preview(source: bytes | str) -> tuple[Image.Image, tuple[int, int], int]:
    """Return the embedded preview, the upright full-size dimensions, and the flip code."""
    import rawpy  # imported lazily: only RAW files need LibRaw

    with rawpy.imread(io.BytesIO(source) if isinstance(source, bytes) else source) as raw:
        sizes = raw.sizes
        size = (sizes.width, sizes.height)
        if sizes.flip in (5, 6):
            size = (size[1], size[0])
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(bytes(thumb.data)))
        else:
            img = Image.fromarray(thumb.data)
        return img, size, sizes.flip


def _container_exif(path: str) -> tuple[str | None, float | None, float | None, str | None]:
    """EXIF from the RAW container itself; most RAW formats are TIFF underneath."""
    try:
        with Image.open(path) as img:
            return _read_exif(img)
    except Exception:
        return None, None, None, None


def process_image(
    source: bytes | str,
    thumb_256: Path,
    thumb_1024: Path,
    file_mtime_ns: int,
    path: str = "",
) -> ImageInfo:
    """Read metadata and write both thumbnails with a single decode.

    `source` is the file's bytes when they are already in memory, else its path.
    `path` names the file so RAW formats can be recognised.
    """
    raw_size: tuple[int, int] | None = None
    raw_flip = 0
    if is_raw(path):
        opened, raw_size, raw_flip = _open_raw_preview(source)
    else:
        opened = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)

    with opened as img:
        width, height = img.size
        orientation = img.getexif().get(274, 1)
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        taken, lat, lon, camera = _read_exif(img)
        if raw_size is not None:
            width, height = raw_size
            if taken is None and path:
                taken, lat, lon, camera = _container_exif(path)

        # JPEG only: decode at reduced scale, still at least 1024 px on each side.
        img.draft("RGB", (1024, 1024))
        work = ImageOps.exif_transpose(img)
        if orientation == 1 and raw_flip in _RAW_ROTATIONS:
            # The preview carries no orientation of its own; follow the RAW's.
            if raw_flip == 3 or (work.width > work.height) != (width > height):
                work = work.transpose(_RAW_ROTATIONS[raw_flip])
        if work.mode not in ("RGB", "RGBA"):
            has_alpha = "A" in work.mode or "transparency" in work.info
            work = work.convert("RGBA" if has_alpha else "RGB")

        large = work.copy()
        large.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        # The grid lays photos out by row height, so the small size is height-bound.
        small = large.copy()
        small.thumbnail((768, 256), Image.Resampling.LANCZOS)

    _save_webp(large, thumb_1024)
    _save_webp(small, thumb_256)

    if taken:
        source_kind = "exif"
    else:
        taken = datetime.fromtimestamp(file_mtime_ns / 1e9).replace(microsecond=0).isoformat()
        source_kind = "file"

    return ImageInfo(
        width=width,
        height=height,
        taken_at=taken,
        taken_at_source=source_kind,
        lat=lat,
        lon=lon,
        camera=camera,
        perceptual_hash=_dhash(small),
    )


def load_bgr(path: str, min_side: int = 1600):
    """Decode a photo upright as a BGR array for the face engine.

    JPEGs are decoded at reduced scale when they stay at least `min_side` on
    each side, which is plenty for face crops and much faster.
    """
    import numpy as np

    raw_size: tuple[int, int] | None = None
    raw_flip = 0
    if is_raw(path):
        opened, raw_size, raw_flip = _open_raw_preview(path)
    else:
        opened = Image.open(path)
    with opened as img:
        orientation = img.getexif().get(274, 1)
        img.draft("RGB", (min_side, min_side))
        work = ImageOps.exif_transpose(img)
        if raw_size is not None and orientation == 1 and raw_flip in _RAW_ROTATIONS:
            if raw_flip == 3 or (work.width > work.height) != (raw_size[0] > raw_size[1]):
                work = work.transpose(_RAW_ROTATIONS[raw_flip])
        rgb = np.asarray(work.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])

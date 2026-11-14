"""Editing and generation jobs.

One job runs at a time on a background thread, reports progress, and can be
cancelled. A job never touches the photo it started from: every result is
saved as a new file in the creations folder and added to the library, marked
as edited or generated, with the settings that made it.

The models sit behind `StudioEngine`. This module decides sizes, masks, and
how a model's output is put back into the full-size photo.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import queue
import random
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image, PngImagePlugin

from . import compose, identity, media, model_store
from .config import Config
from .db import Database
from .scanner import ScanManager

log = logging.getLogger("pixora.studio")

EDITED, GENERATED = "edited", "generated"
MAX_COUNT = 4
# Results larger than this gain nothing visible and cost a lot of time.
MAX_EDIT_SIDE = 4096
MAX_BACKGROUND_SIDE = 2048
MAX_ENHANCED_SIDE = 6000
# The models are dropped from memory after this long without a job.
IDLE_UNLOAD_SECONDS = 300
_KEEP_JOBS = 50

DEFAULT_NEGATIVE = "blurry, low quality, deformed, bad anatomy, extra fingers, watermark, text"


@dataclass(frozen=True)
class Style:
    id: str
    label: str
    prompt: str
    negative: str = ""
    model: str | None = None  # id in model_store.MODELS of the LoRA, if the look uses one
    weight: float = 1.0
    restyle: bool = True  # offered for restyling a photo, not only for new pictures
    # How far a restyled photo may move from the original by default (0 to 1).
    # Flat, drawn looks need more freedom than looks that keep photographic shading.
    strength: float = 0.6


STYLES: dict[str, Style] = {
    s.id: s
    for s in (
        Style(
            "photo", "Photo", "photograph, natural light, sharp focus, highly detailed",
            "cartoon, painting, illustration, 3d render", restyle=False,
        ),
        Style(
            "cinematic", "Cinematic", "cinematic film still, shallow depth of field, dramatic lighting, film grain",
            "cartoon, painting, illustration", restyle=False,
        ),
        Style(
            "anime", "Anime", "anime style, anime artwork, cel shading, clean line art, vibrant colours",
            "photo, photorealistic, 3d render", "lora-anime", 2.0, strength=0.8,
        ),
        Style(
            "3d", "3D render", "3d style, 3d render, animated film character, soft studio lighting, smooth shading",
            "photo, sketch, flat colours", "lora-3d", 1.0,
        ),
        Style(
            "watercolor", "Watercolour", "watercolor painting, soft washes of colour, paper texture",
            "photo, photorealistic, 3d render", "lora-watercolor", 1.0,
        ),
        Style(
            "pixel", "Pixel art", "pixel art, 16-bit game sprite style", "photo, photorealistic, blurry", "lora-pixel",
            1.0, strength=0.8,
        ),
        Style(
            "clay", "Clay", "Clay Animation, Clay, stop-motion clay figure, handmade look",
            "photo, photorealistic, flat colours", "lora-clay", 1.0,
        ),
    )
}

SIZES: dict[str, tuple[str, int, int]] = {
    "square": ("Square", 1024, 1024),
    "portrait": ("Portrait", 832, 1216),
    "landscape": ("Landscape", 1216, 832),
    "wide": ("Wide", 1344, 768),
    "tall": ("Tall", 768, 1344),
}

OPERATIONS = {
    "remove_background": EDITED,
    "replace_background": EDITED,
    "enhance": EDITED,
    "restore": EDITED,
    "stylize": EDITED,
    "inpaint": EDITED,
    "extend": EDITED,
    "generate": GENERATED,
    "download": None,
}


class Cancelled(Exception):
    pass


class StudioError(Exception):
    """A request that cannot run; the message is shown to the user."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Progress:
    """How a running job reports back and learns that it was cancelled."""

    def __init__(self, job: "Job", start: float = 0.0, end: float = 1.0):
        self._job, self._start, self._end = job, start, end

    def report(self, fraction: float, message: str | None = None) -> None:
        self.check()
        self._job.progress = self._start + (self._end - self._start) * min(max(fraction, 0.0), 1.0)
        if message:
            self._job.message = message

    def check(self) -> None:
        if self._job.cancel.is_set():
            raise Cancelled()

    def cancelled(self) -> bool:
        return self._job.cancel.is_set()

    def part(self, start: float, end: float) -> "Progress":
        """A slice of this job's progress bar, for one step of several."""
        width = self._end - self._start
        return Progress(self._job, self._start + width * start, self._start + width * end)


class StudioEngine(Protocol):
    """The models. Images are RGB at the size they are to be worked on."""

    def matte(self, image: Image.Image, progress: Progress) -> Image.Image:
        """An 'L' mask: white where the subject is."""
        ...

    def upscale(self, image: Image.Image, scale: float, progress: Progress) -> Image.Image:
        ...

    def restore_faces(self, image: Image.Image, strength: float, progress: Progress) -> Image.Image:
        ...

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, negative: str, seed: int, progress: Progress
    ) -> Image.Image:
        """Repaint the white part of `mask`. Returns an image of the same size."""
        ...

    def restyle(
        self, image: Image.Image, prompt: str, negative: str, style: Style, strength: float, seed: int,
        progress: Progress,
    ) -> Image.Image:
        ...

    def generate(
        self, prompt: str, negative: str, width: int, height: int, seed: int, style: Style | None,
        references: list[identity.Reference], progress: Progress,
    ) -> Image.Image:
        """A new picture. With `references`, it shows the person in them."""
        ...

    def unload(self) -> None:
        """Forget every model."""
        ...


@dataclass
class Job:
    id: int
    operation: str
    params: dict
    photo_id: int | None = None
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = "Waiting"
    error: str | None = None
    result_ids: list[int] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decode_mask(data_url: str, size: tuple[int, int]) -> Image.Image:
    """The mask the user painted, as sent by the UI: an image, white where painted."""
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(raw)) as img:
            if "A" in img.mode:
                # Painted strokes on a transparent canvas.
                mask = img.getchannel("A")
            else:
                mask = img.convert("L")
            mask = mask.resize(size, Image.Resampling.BILINEAR)
    except Exception:
        raise StudioError("The painted area could not be read")
    return mask.point(lambda v: 255 if v >= 64 else 0)


def _safe_stem(text: str) -> str:
    return re.sub(r"[^\w\- ]+", "", text).strip()[:40] or "picture"


def default_engine(config: Config, db: Database) -> StudioEngine:
    try:
        from .studio_engine import TorchEngine
    except ImportError as e:
        raise StudioError(
            "The editing engine is not installed. In the service folder run: "
            '.venv\\Scripts\\python -m pip install -e ".[studio]" '
            "--extra-index-url https://download.pytorch.org/whl/cu128",
            503,
        ) from e
    return TorchEngine(config.models_dir)


class Studio:
    def __init__(
        self,
        db: Database,
        config: Config,
        scanner: ScanManager,
        engine_factory: Callable[[], StudioEngine] | None = None,
    ):
        self.db, self.config, self.scanner = db, config, scanner
        # With an injected engine there are no model files to look for.
        self._check_models = engine_factory is None
        self._engine_factory = engine_factory or (lambda: default_engine(config, db))
        self._engine: StudioEngine | None = None
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ---- the queue ---------------------------------------------------------

    def submit(self, operation: str, params: dict) -> Job:
        if operation not in OPERATIONS:
            raise StudioError("Unknown operation")
        params = self._validate(operation, dict(params))
        with self._lock:
            job = Job(id=self._next_id, operation=operation, params=params, photo_id=params.get("photo_id"))
            self._next_id += 1
            self._jobs[job.id] = job
            for old in sorted(self._jobs)[:-_KEEP_JOBS]:
                if self._jobs[old].state not in ("queued", "running"):
                    del self._jobs[old]
            if not (self._thread and self._thread.is_alive()):
                self._thread = threading.Thread(target=self._work, name="pixora-studio", daemon=True)
                self._thread.start()
        self._queue.put(job)
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        return [self._jobs[i] for i in sorted(self._jobs, reverse=True)]

    def cancel(self, job_id: int) -> Job | None:
        job = self._jobs.get(job_id)
        if job and job.state in ("queued", "running"):
            job.cancel.set()
            if job.state == "queued":
                job.state, job.message = "cancelled", "Cancelled"
        return job

    def shutdown(self) -> None:
        for job in list(self._jobs.values()):
            job.cancel.set()
        self._queue.put(None)
        thread = self._thread
        if thread:
            thread.join(timeout=30)

    def wait(self, job_id: int, timeout: float = 30) -> Job:
        """Block until a job has finished. For tests."""
        import time

        deadline = time.monotonic() + timeout
        job = self._jobs[job_id]
        while job.state in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.01)
        return job

    def _work(self) -> None:
        try:
            while True:
                try:
                    job = self._queue.get(timeout=IDLE_UNLOAD_SECONDS)
                except queue.Empty:
                    if self._engine is not None:
                        self._engine.unload()
                    continue
                if job is None:
                    return
                if job.state != "queued":
                    continue
                self._run(job)
        finally:
            self.db.close_thread()

    def _run(self, job: Job) -> None:
        job.state, job.message = "running", "Starting"
        self.scanner.hold(True)
        if job.operation != "download":
            # The scan's own models would otherwise sit in graphics memory the job needs.
            self.scanner.release_models()
        try:
            progress = Progress(job)
            if job.operation == "download":
                model_store.download(self.config.models_dir, job.params["models"], progress.report, progress.cancelled)
                progress.check()
            else:
                if self._engine is None:
                    job.message = "Loading the editing engine"
                    self._engine = self._engine_factory()
                self._execute(job, self._engine, progress)
            job.state, job.progress, job.message = "done", 1.0, "Done"
        except Cancelled:
            job.state, job.message = "cancelled", "Cancelled"
        except StudioError as e:
            job.state, job.error, job.message = "error", str(e), "Failed"
        except Exception as e:
            log.exception("Job %s (%s) failed", job.id, job.operation)
            if type(e).__name__ == "OutOfMemoryError":
                error = "The graphics card ran out of memory. Close other programs that use it, then try again."
                if self._engine is not None:
                    self._engine.unload()
            else:
                error = f"{type(e).__name__}: {e}"
            job.state, job.error, job.message = "error", error, "Failed"
        finally:
            rest = getattr(self._engine, "rest", None)
            if rest is not None:
                try:
                    rest()
                except Exception:
                    log.exception("Could not free the graphics card after job %s", job.id)
            self.scanner.hold(False)
        if job.result_ids and not self.scanner.paused:
            # Faces and scene tags for the new pictures.
            self.scanner.start(walk=False)

    # ---- checking a request ------------------------------------------------

    def _need_models(self, feature: str, extra: tuple[str, ...] = ()) -> None:
        if not self._check_models:
            return
        missing = model_store.missing_for(self.config.models_dir, feature, extra)
        if missing:
            labels = ", ".join(model_store.MODELS[m].label for m in missing)
            raise StudioError(f"This needs models that are not installed yet: {labels}", 409)

    def _style(self, params: dict, restyle: bool) -> Style | None:
        style_id = params.get("style")
        if not style_id:
            return None
        style = STYLES.get(style_id)
        if style is None or (restyle and not style.restyle):
            raise StudioError("Unknown style")
        return style

    def _validate(self, operation: str, p: dict) -> dict:
        if operation == "download":
            ids = p.get("models") or list(model_store.MODELS)
            if any(m not in model_store.MODELS for m in ids):
                raise StudioError("Unknown model")
            return {"models": ids}

        p["prompt"] = " ".join((p.get("prompt") or "").split())
        p["negative_prompt"] = " ".join((p.get("negative_prompt") or "").split())
        p["count"] = min(max(int(p.get("count") or 1), 1), MAX_COUNT)
        if p.get("seed") is None:
            p["seed"] = random.randrange(2**31)
        p["seed"] = int(p["seed"]) % 2**31

        if operation == "generate":
            if not p["prompt"]:
                raise StudioError("Describe the picture you want")
            if p.get("size", "square") not in SIZES:
                raise StudioError("Unknown size")
            p.setdefault("size", "square")
            style = self._style(p, restyle=False)
            person_id = p.get("person_id")
            if person_id is not None:
                row = self.db.conn().execute("SELECT name FROM people WHERE id=?", (person_id,)).fetchone()
                if row is None:
                    raise StudioError("Person not found", 404)
                p["person_name"] = row["name"]
                if not identity.references_for(self.db, person_id, 1):
                    raise StudioError(f"There are no photos of {row['name']} to learn their face from")
            extra = (style.model,) if style and style.model else ()
            self._need_models("generate_person" if person_id is not None else "generate", extra)
            return p

        row = self.db.conn().execute(
            "SELECT pf.path FROM photos p JOIN photo_files pf ON pf.id = p.primary_file_id "
            "WHERE p.id=? AND p.ready=1 AND pf.missing=0",
            (p.get("photo_id"),),
        ).fetchone()
        if row is None or not os.path.isfile(row["path"]):
            raise StudioError("Photo not found", 404)

        extra: tuple[str, ...] = ()
        if operation == "replace_background" and not p["prompt"]:
            raise StudioError("Describe the new background")
        if operation == "stylize":
            style = self._style(p, restyle=True)
            if style is None:
                raise StudioError("Choose a style")
            extra = (style.model,) if style.model else ()
            p["strength"] = min(max(float(p.get("strength") or style.strength), 0.2), 0.9)
        if operation == "inpaint":
            if not p.get("mask"):
                raise StudioError("Paint over the area to change first")
            if not p["prompt"]:
                p["erase"] = True
        if operation == "extend":
            sides = [min(max(float(p.get(k) or 0), 0.0), 1.0) for k in ("left", "top", "right", "bottom")]
            if not any(sides):
                raise StudioError("Choose at least one side to extend")
            p["left"], p["top"], p["right"], p["bottom"] = sides
        if operation in ("enhance", "restore"):
            p["faces"] = bool(p.get("faces", operation == "restore"))
        self._need_models(operation, extra)
        return p

    # ---- running a job -----------------------------------------------------

    def _load_source(self, photo_id: int) -> tuple[Image.Image, str]:
        row = self.db.conn().execute(
            "SELECT pf.path FROM photos p JOIN photo_files pf ON pf.id = p.primary_file_id WHERE p.id=?",
            (photo_id,),
        ).fetchone()
        if row is None or not os.path.isfile(row["path"]):
            raise StudioError("The photo's file is no longer available", 404)
        # Decoded upright and at full size; the file itself is only ever read.
        bgr = media.load_bgr(row["path"], min_side=10**6)
        return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1])), Path(row["path"]).stem

    def _scene_hint(self, photo_id: int) -> str:
        """A few words about a photo, from what the library already knows of it."""
        conn = self.db.conn()
        faces = conn.execute("SELECT count(*) FROM faces WHERE photo_id=? AND dismissed=0", (photo_id,)).fetchone()[0]
        words = ["a person"] if faces == 1 else ["two people"] if faces == 2 else ["a group of people"] if faces else []
        from .scenes import TAG_FLOOR

        words += [
            r["name"]
            for r in conn.execute(
                "SELECT t.name FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
                "WHERE pt.photo_id=? AND pt.score>=? ORDER BY pt.score DESC LIMIT 2",
                (photo_id, TAG_FLOOR),
            )
        ]
        return ", ".join(words)

    def _execute(self, job: Job, engine: StudioEngine, progress: Progress) -> None:
        p, op = job.params, job.operation
        negative = ", ".join(x for x in (p["negative_prompt"], DEFAULT_NEGATIVE) if x)
        count = p["count"] if op in ("generate", "stylize", "replace_background", "inpaint", "extend") else 1

        source, stem = (None, "generated")
        if op != "generate":
            progress.report(0.0, "Reading the photo")
            source, stem = self._load_source(p["photo_id"])
            if op != "enhance":
                source = compose.limit(source, MAX_EDIT_SIDE)

        # Work shared by every variation is done once.
        subject = None
        if op == "replace_background":
            source = compose.limit(source, MAX_BACKGROUND_SIDE)
            subject = engine.matte(source, progress.part(0.0, 0.1))
        references: list[identity.Reference] = []
        if op == "generate" and p.get("person_id") is not None:
            references = identity.references_for(self.db, p["person_id"])
        shared = 0.1 if subject is not None else 0.0

        for n in range(count):
            seed = (p["seed"] + n) % 2**31
            step = progress.part(shared + (1 - shared) * n / count, shared + (1 - shared) * (n + 1) / count)
            step.report(0.0, f"Picture {n + 1} of {count}" if count > 1 else "Working")
            image = self._make(op, p, engine, source, subject, references, negative, seed, step)
            step.report(1.0, "Saving")
            recorded = {k: v for k, v in p.items() if k not in ("mask", "photo_id", "count")}
            recorded["seed"] = seed
            photo_id = self.save(image, OPERATIONS[op], op, stem, recorded, p.get("photo_id"))
            if photo_id not in job.result_ids:  # the same picture twice is one library item
                job.result_ids.append(photo_id)

    def _make(
        self, op: str, p: dict, engine: StudioEngine, source: Image.Image | None, subject: Image.Image | None,
        references: list[identity.Reference], negative: str, seed: int, progress: Progress,
    ) -> Image.Image:
        if op == "generate":
            style = STYLES.get(p.get("style") or "")
            _, width, height = SIZES[p["size"]]
            prompt = p["prompt"]
            if references:
                # The model has never heard of this person; the face comes from the references.
                people = [c for c in identity.list_candidates(self.db) if c.person_id == p["person_id"]]
                mentions = [m for m in identity.find_mentions(prompt, people) if m.status != "unknown"]
                prompt = identity.describe_without_names(prompt, mentions)
            if style:
                prompt = f"{prompt}, {style.prompt}"
                negative = ", ".join(x for x in (style.negative, negative) if x)
            return engine.generate(prompt, negative, width, height, seed, style, references, progress)

        assert source is not None
        if op == "remove_background":
            cut = source.convert("RGBA")
            cut.putalpha(engine.matte(source, progress))
            return cut

        if op == "replace_background":
            assert subject is not None
            work = source.resize(compose.working_size(*source.size), Image.Resampling.LANCZOS)
            # Repaint a little into the subject's outline so no rim of the old background survives.
            background = compose.grow(subject.resize(work.size, Image.Resampling.BILINEAR), -3).point(lambda v: 255 - v)
            painted = engine.inpaint(work, background, p["prompt"], negative, seed, progress)
            # The subject keeps its own full-size pixels.
            return Image.composite(source, painted.resize(source.size, Image.Resampling.LANCZOS), subject)

        if op == "enhance":
            scale = min(4.0, max(1.0, MAX_ENHANCED_SIDE / max(source.size)))
            out = engine.upscale(source, scale, progress.part(0.0, 0.8 if p["faces"] else 1.0))
            return engine.restore_faces(out, 0.6, progress.part(0.8, 1.0)) if p["faces"] else out

        if op == "restore":
            out = engine.restore_faces(source, 1.0, progress.part(0.0, 0.4)) if p["faces"] else source
            scale = min(4.0, max(1.0, 3000 / max(source.size)))
            return engine.upscale(out, scale, progress.part(0.4, 1.0))

        if op == "stylize":
            style = STYLES[p["style"]]
            work = source.resize(compose.working_size(*source.size), Image.Resampling.LANCZOS)
            hint = p["prompt"] or self._scene_hint(p["photo_id"])
            prompt = ", ".join(x for x in (style.prompt, hint) if x)
            style_negative = ", ".join(x for x in (style.negative, negative) if x)
            return engine.restyle(work, prompt, style_negative, style, p["strength"], seed, progress)

        if op == "inpaint":
            mask = decode_mask(p["mask"], source.size)
            box = compose.mask_box(mask)
            if box is None:
                raise StudioError("Paint over the area to change first")
            crop, crop_mask = source.crop(box), mask.crop(box)
            size = compose.working_size(*crop.size)
            scale = size[0] / crop.width
            work_mask = compose.grow(crop_mask.resize(size, Image.Resampling.BILINEAR), 8)
            prompt = p["prompt"]
            if p.get("erase"):
                prompt = "empty background, seamless continuation of the surroundings"
                negative = ", ".join(("object, person, animal, text", negative))
            painted = engine.inpaint(
                crop.resize(size, Image.Resampling.LANCZOS), work_mask, prompt, negative, seed, progress
            )
            # Only the painted area, softened at its edge, replaces the photo's pixels.
            soft = compose.feather(compose.grow(crop_mask, round(8 / scale)), max(2, round(6 / scale)))
            return compose.blend(source, painted, soft, box)

        if op == "extend":
            w, h = source.size
            left, top, right, bottom = (
                round(w * p["left"]), round(h * p["top"]), round(w * p["right"]), round(h * p["bottom"]),
            )
            full = (w + left + right, h + top + bottom)
            if max(full) > MAX_EDIT_SIDE:
                shrink = MAX_EDIT_SIDE / max(full)
                source = source.resize((round(w * shrink), round(h * shrink)), Image.Resampling.LANCZOS)
                w, h = source.size
                left, top, right, bottom = (round(v * shrink) for v in (left, top, right, bottom))
                full = (w + left + right, h + top + bottom)
            size = compose.working_size(*full)
            scale = size[0] / full[0]
            small = source.resize((max(8, round(w * scale)), max(8, round(h * scale))), Image.Resampling.LANCZOS)
            pads = [round(v * scale) for v in (left, top)]
            canvas, mask = compose.extended_canvas(
                small, pads[0], pads[1], size[0] - small.width - pads[0], size[1] - small.height - pads[1]
            )
            prompt = p["prompt"] or self._scene_hint(p["photo_id"]) or "seamless continuation of the scene"
            painted = engine.inpaint(canvas, mask, prompt, negative, seed, progress)
            out = painted.resize(full, Image.Resampling.LANCZOS)
            sides = (left > 0, top > 0, right > 0, bottom > 0)
            return compose.keep_original(out, source, (left, top), sides, ramp=max(8, round(24 / scale)))

        raise StudioError("Unknown operation")

    # ---- saving ------------------------------------------------------------

    def save(
        self, image: Image.Image, kind: str, operation: str, stem: str, params: dict, source_photo_id: int | None
    ) -> int:
        """Write a result as a new file and add it to the library."""
        folder = self.config.creations_dir
        folder.mkdir(parents=True, exist_ok=True)
        transparent = image.mode == "RGBA"
        note = f"Pixora, AI {kind}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{_safe_stem(stem)}-{operation.replace('_', '-')}-{stamp}"
        ext = ".png" if transparent else ".jpg"
        path = folder / (base + ext)
        n = 1
        while path.exists():  # never overwrite anything, not even an earlier result
            n += 1
            path = folder / f"{base}-{n}{ext}"
        if transparent:
            meta = PngImagePlugin.PngInfo()
            meta.add_text("Software", note)
            image.save(path, "PNG", pnginfo=meta)
        else:
            exif = Image.Exif()
            exif[0x0131] = note  # Software: says plainly that this is not a camera's picture
            image.convert("RGB").save(path, "JPEG", quality=95, subsampling=0, exif=exif)

        photo_id = self.scanner.import_file(str(path), str(folder))
        if photo_id is None:
            raise StudioError("The result was saved but could not be added to the library")
        with self.db.write() as conn:
            conn.execute("UPDATE photos SET origin=? WHERE id=? AND origin IS NULL", (kind, photo_id))
            conn.execute(
                "INSERT INTO creations(photo_id, kind, operation, source_photo_id, params, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (photo_id, kind, operation, source_photo_id, json.dumps(params), _now()),
            )
        return photo_id

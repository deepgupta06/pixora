"""Image arithmetic around the models: working sizes, masks, and putting a
model's output back into the full-size photo. No model code lives here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

WORKING_LONG_SIDE = 1024  # what SDXL was trained around
_MULTIPLE = 8


def working_size(width: int, height: int, long_side: int = WORKING_LONG_SIDE) -> tuple[int, int]:
    """The size a photo is worked on at: same shape, sides divisible by 8."""
    scale = long_side / max(width, height)
    return (
        max(_MULTIPLE * 8, round(width * scale / _MULTIPLE) * _MULTIPLE),
        max(_MULTIPLE * 8, round(height * scale / _MULTIPLE) * _MULTIPLE),
    )


def limit(image: Image.Image, long_side: int) -> Image.Image:
    """Shrink so the longer side is at most `long_side`. Never enlarges."""
    if max(image.size) <= long_side:
        return image
    scale = long_side / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def grow(mask: Image.Image, pixels: int) -> Image.Image:
    """Widen (or, with a negative amount, narrow) the white area of a mask."""
    if pixels == 0:
        return mask
    size = abs(pixels) * 2 + 1
    return mask.filter(ImageFilter.MaxFilter(size) if pixels > 0 else ImageFilter.MinFilter(size))


def feather(mask: Image.Image, pixels: int) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(pixels)) if pixels > 0 else mask


def mask_box(mask: Image.Image, pad: float = 0.5, min_side: int = 512) -> tuple[int, int, int, int] | None:
    """The region to rework around a painted mask: its bounds plus context.

    The model needs to see the surroundings to blend with them, so the box is
    padded and never smaller than `min_side` where the photo allows.
    """
    bounds = mask.point(lambda v: 255 if v >= 128 else 0).getbbox()
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    w, h = x2 - x1, y2 - y1
    side_w = min(mask.width, max(min_side, round(w * (1 + 2 * pad))))
    side_h = min(mask.height, max(min_side, round(h * (1 + 2 * pad))))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left = min(max(0, cx - side_w // 2), mask.width - side_w)
    top = min(max(0, cy - side_h // 2), mask.height - side_h)
    return left, top, left + side_w, top + side_h


def blend(base: Image.Image, patch: Image.Image, mask: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Put `patch` into `base` at `box`, showing it only where `mask` is white."""
    out = base.copy()
    region = (box[2] - box[0], box[3] - box[1])
    if patch.size != region:
        patch = patch.resize(region, Image.Resampling.LANCZOS)
    if mask.size != region:
        mask = mask.resize(region, Image.Resampling.BILINEAR)
    out.paste(patch.convert(base.mode), box[:2], mask.convert("L"))
    return out


def extended_canvas(
    image: Image.Image, left: int, top: int, right: int, bottom: int, overlap: int = 16
) -> tuple[Image.Image, Image.Image]:
    """A larger canvas with the photo placed in it, and the mask of what to fill.

    The new area starts as a blurred smear of the photo's edges, which gives
    the model colours to continue from. The mask reaches `overlap` pixels into
    the photo so the join is repainted too.
    """
    w, h = image.size
    canvas_size = (w + left + right, h + top + bottom)
    # Stretching the photo over the whole canvas, heavily blurred, is a cheap
    # stand-in for "more of the same".
    canvas = image.resize(canvas_size, Image.Resampling.BILINEAR).filter(
        ImageFilter.GaussianBlur(max(canvas_size) // 24)
    )
    canvas.paste(image, (left, top))
    mask = Image.new("L", canvas_size, 255)
    keep = (
        left + (overlap if left else 0),
        top + (overlap if top else 0),
        left + w - (overlap if right else 0),
        top + h - (overlap if bottom else 0),
    )
    mask.paste(0, keep)
    return canvas, mask


def keep_original(
    result: Image.Image, original: Image.Image, offset: tuple[int, int], sides: tuple[bool, bool, bool, bool], ramp: int
) -> Image.Image:
    """Lay the untouched photo back over an extended result.

    `sides` says which edges (left, top, right, bottom) were extended; those
    edges fade in over `ramp` pixels so the join does not show.
    """
    w, h = original.size
    alpha = np.ones((h, w), dtype=np.float32)
    ramp = max(0, min(ramp, w // 2, h // 2))
    if ramp:
        fade = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        if sides[0]:
            alpha[:, :ramp] *= fade[None, :]
        if sides[2]:
            alpha[:, w - ramp :] *= fade[None, ::-1]
        if sides[1]:
            alpha[:ramp, :] *= fade[:, None]
        if sides[3]:
            alpha[h - ramp :, :] *= fade[::-1, None]
    out = result.copy()
    out.paste(original, offset, Image.fromarray((alpha * 255).astype(np.uint8), "L"))
    return out
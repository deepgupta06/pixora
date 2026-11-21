"""How a known person looks, so a picture of them holds together.

A picture of a known person is composed from words first, and only then is the face
painted again as them. Left alone, the neck, hands and hair belong to an invented
stranger: a different skin tone, hair of another colour, another build. Three things
close that gap. The words describe the person's hair colour, read from their photos.
The skin outside the repainted face is brought to the face's new tone. And the user
can name a build, which no photo of a face can show.

The person's own skin tone is deliberately not put into words: the same person reads
as pale in one photo and dark in the next, depending on the flash and the white
balance, and a wrong word would make the picture someone else. The skin is matched to
the face as painted instead, which needs no absolute tone.

Nothing here needs the graphics card or the image models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from . import faces

# Where the skin is on a face aligned to the 112 px ArcFace template: under the eyes on
# either side of the nose, and the forehead above the brows.
_CHEEKS = ((27, 60, 46, 76), (67, 60, 86, 76))
_FOREHEAD = (40, 24, 73, 38)
_MIN_PIXELS = 60

# What the user can choose for the body, which a face does not show. The words are plain
# on purpose: "curvy" and the like make the image model dress and pose a real person as a
# pin-up.
BUILDS = {
    "slim": ("Slim", "a slim build"),
    "average": ("Average", "an average build"),
    "athletic": ("Athletic", "an athletic build"),
    "fuller": ("Fuller", "a fuller, rounder build"),
    "larger": ("Larger", "a large, heavyset build"),
}

_HAIR_COLOURS = ("black", "dark brown", "light brown", "blond", "red", "grey", "white")
# Asked of a scene model about each reference photo, one line each.
HAIR_PROMPTS = [f"a photo of a person with {colour} hair" for colour in _HAIR_COLOURS] + ["a photo of a bald person"]

_SAYS_HAIR = re.compile(
    r"\b(hair|haired|bald|blonde?|brunette|redhead|ponytail|braids?|curls|curly|afro|dreadlocks|hijab|turban)\b", re.I
)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB with values 0 to 255 as L*a*b*, where L runs 0 to 100."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB) * 255.0


# ---- reading skin and hair from a face ----------------------------------------------


def skin_samples(aligned_rgb: np.ndarray) -> np.ndarray | None:
    """L*a*b* values of the skin on a face aligned to the ArcFace template (112 px, RGB).

    Cheeks and forehead, without shadows, highlights, stubble or the rims of glasses.
    None when too little skin could be read.
    """
    lab = to_lab(aligned_rgb)

    def patch(box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return lab[y1:y2, x1:x2].reshape(-1, 3)

    cheeks = np.concatenate([patch(box) for box in _CHEEKS])
    cheeks = cheeks[(cheeks[:, 0] > 20) & (cheeks[:, 0] < 95)]
    if len(cheeks) < _MIN_PIXELS:
        return None
    middle = np.median(cheeks, axis=0)
    both = np.concatenate([cheeks, patch(_FOREHEAD)])
    near = (np.abs(both[:, 0] - middle[0]) < 18) & (np.hypot(both[:, 1] - middle[1], both[:, 2] - middle[2]) < 10)
    both = both[near]
    return both if len(both) >= _MIN_PIXELS else None


def hair_words(logits: np.ndarray) -> str | None:
    """Hair colour in words, from a scene model's scores for `HAIR_PROMPTS`.

    `logits` has one row per reference photo. Says nothing when the photos disagree, or
    the person looks bald, rather than describe hair they may not have.
    """
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)
    if p[-1] >= 0.4:
        return None
    dark = p[0] + p[1]
    if dark >= 0.7:  # nearly everyone's hair: black, or dark brown that photographs as black
        return "black hair" if p[0] >= p[1] else "dark brown hair"
    best = int(np.argmax(p[:-1]))
    return f"{_HAIR_COLOURS[best]} hair" if p[best] >= 0.5 else None


@dataclass(frozen=True)
class Look:
    """A person in the words an image model reads."""

    person: str  # "a 30 year old woman"
    hair: str | None = None

    def words(self, prompt: str, build: str | None = None) -> str:
        """The person with their features. What the user already said about the person's
        hair is not contradicted."""
        features = []
        if build in BUILDS:
            features.append(BUILDS[build][1])
        if self.hair and not _SAYS_HAIR.search(prompt):
            features.append(self.hair)
        if not features:
            return self.person
        joined = features[0] if len(features) == 1 else ", ".join(features[:-1]) + " and " + features[-1]
        return f"{self.person} with {joined}"


# ---- bringing the rest of the skin to the face ----------------------------------------


def _face_tone(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The median L*a*b* of a face's skin and how much its colour varies, or None."""
    matrix, _ = cv2.estimateAffinePartial2D(points, faces._ARCFACE_DST, method=cv2.LMEDS)
    if matrix is None:
        return None
    aligned = cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), matrix, (112, 112), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    samples = skin_samples(aligned)
    if samples is None:
        return None
    return np.median(samples, axis=0), samples.std(axis=0)


def face_size(points: np.ndarray) -> float:
    """About half a face's width: the larger of the eye distance and the eye-to-mouth distance."""
    eyes, mouth = (points[0] + points[1]) / 2, (points[3] + points[4]) / 2
    return max(float(np.linalg.norm(points[0] - points[1])), float(np.linalg.norm(eyes - mouth)))


def match_skin(
    before: np.ndarray,
    after: np.ndarray,
    points: np.ndarray,
    repainted: np.ndarray,
    figure: np.ndarray | None = None,
) -> np.ndarray:
    """Carry the change in a face's skin over to the rest of the person's skin.

    `before` is the picture as composed; `after` is the same with the face painted again,
    so a new tone and little else has changed on the face. Every skin-coloured pixel
    outside the repainted part is moved by that same change, so the neck, arms and hands
    stay the face's colour instead of keeping an invented person's.

    `before` and `after` are float RGB (0 to 255), `points` the face's five landmarks,
    `repainted` (0 to 1) how much of each pixel is new face, which is not moved again.
    `figure` (0 to 1) is where the person is, to keep walls, sand and wood out of it.
    Returns `after`, changed.
    """
    old, new = _face_tone(before, points), _face_tone(after, points)
    if old is None or new is None:
        return after
    (old_tone, spread), (new_tone, _) = old, new
    # Large changes are a sign that something else moved, not a skin tone.
    delta = np.clip(new_tone - old_tone, (-20.0, -10.0, -12.0), (20.0, 10.0, 12.0)).astype(np.float32)
    if float(np.abs(delta).max()) < 1.5:
        return after

    lab = to_lab(before)  # what was skin is judged in the picture as it was composed
    sigma_a, sigma_b = np.clip(spread[1:] * 2.0, 5.0, 9.0)
    chroma = ((lab[..., 1] - old_tone[1]) / sigma_a) ** 2 + ((lab[..., 2] - old_tone[2]) / sigma_b) ** 2
    skin = np.exp(-0.5 * chroma)
    skin *= np.clip((lab[..., 0] - 12.0) / 13.0, 0.0, 1.0)  # black is not skin
    skin *= np.clip((98.0 - lab[..., 0]) / 8.0, 0.0, 1.0)  # nor are blown highlights
    where = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 1.5) * (1.0 - np.clip(repainted, 0.0, 1.0))
    if figure is not None:
        where *= figure
    else:
        # Without the figure, trust only what is near: about a body's length from the face.
        h, w = where.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = points.mean(axis=0)
        far = np.hypot(xx - cx, yy - cy) / max(face_size(points), 1.0)
        where *= np.clip(1.0 - (far - 14.0) / 12.0, 0.0, 1.0)
    shifted = np.clip(from_lab(to_lab(after) + delta * where[..., None]), 0.0, 255.0)
    # Pixels that were not moved keep their exact values, not the colour space's rounding of them.
    return np.where(where[..., None] > 0.003, shifted, after)

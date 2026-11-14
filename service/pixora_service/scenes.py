"""Scene tags and natural-language search with SigLIP 2.

One model gives each photo an embedding. Scene labels and search queries are
text embedded by the same model, so tagging and search are both a similarity
between a text and the stored photo embeddings.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from PIL import Image

from .faces import ModelsMissing

# The curated label set. Phase 0 showed the plan's nineteen labels miss what
# many libraries are mostly made of, so a few everyday ones were added.
LABELS: dict[str, list[str]] = {
    "nature": ["mountain", "beach", "forest", "lake", "desert", "snow", "garden"],
    "places": ["mall", "restaurant", "home", "office", "street", "temple", "stadium", "water park", "classroom", "park"],
    "occasions": ["wedding", "birthday", "festival", "graduation", "concert", "party"],
}
# Wording matters for zero-shot models; these read better than the bare word.
_PROMPTS = {
    "home": "the inside of a home",
    "office": "an office",
    "street": "a street",
    "mall": "a shopping mall",
    "wedding": "a wedding",
    "birthday": "a birthday celebration",
    "festival": "a festival",
    "graduation": "a graduation ceremony",
    "concert": "a concert",
    "party": "a party",
    "snow": "snow",
    "garden": "a garden",
    "water park": "a water park",
    "classroom": "a classroom",
    "park": "a park",
}
# A label applies to a photo at or above this probability. SigLIP's scores are
# small by nature; this value comes from one folder in Phase 0 and is a starting point.
TAG_FLOOR = 0.002
BATCH = 16


def label_prompt(label: str) -> str:
    return f"this is a photo of {_PROMPTS.get(label, 'a ' + label)}."


def all_labels() -> list[tuple[str, str]]:
    return [(label, category) for category, labels in LABELS.items() for label in labels]


class SceneEngine(Protocol):
    version: str

    def embed_images(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Return L2-normalised embeddings and, per image, a probability for each label."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Return the L2-normalised embedding of a search query."""
        ...

    def probability(self, cosine: np.ndarray) -> np.ndarray:
        """Turn text-to-photo cosine similarities into the model's probabilities."""
        ...


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _pixels(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((x - 0.5) / 0.5).transpose(2, 0, 1)


class SigLIP2Engine:
    """SigLIP 2 ViT-B/16 at 224 px, from the combined ONNX export (Apache 2.0)."""

    version = "siglip2-base-patch16-224"

    def __init__(self, models_dir: Path, providers: list[str]):
        from tokenizers import Tokenizer

        model = models_dir / "siglip2" / "onnx" / "model.onnx"
        tokenizer = models_dir / "siglip2" / "tokenizer.json"
        missing = [str(p) for p in (model, tokenizer) if not p.is_file()]
        if missing:
            raise ModelsMissing("Missing model files: " + ", ".join(missing))
        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        self._tokenizer.enable_padding(length=64, pad_id=0, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=64)
        self._session = ort.InferenceSession(str(model), providers=providers)
        self._outputs = [o.name for o in self._session.get_outputs()]
        # The scan thread and search requests share the session; DirectML wants one call at a time.
        self._lock = threading.Lock()
        self._label_ids = self._tokenize([label_prompt(label) for label, _ in all_labels()])
        self._blank = np.zeros((1, 3, 224, 224), dtype=np.float32)

        # logits = scale * cosine + bias. Recover both once, so stored photo
        # embeddings can be scored against any text without running the image side.
        out = self._run(self._label_ids, np.random.default_rng(0).normal(size=(4, 3, 224, 224)).astype(np.float32))
        cosine = (_unit(out["image_embeds"]) @ _unit(out["text_embeds"]).T).reshape(-1)
        self._scale, self._bias = (float(v) for v in np.polyfit(cosine, out["logits_per_image"].reshape(-1), 1))

    def _tokenize(self, texts: list[str]) -> np.ndarray:
        return np.array([e.ids for e in self._tokenizer.encode_batch([t.lower() for t in texts])], dtype=np.int64)

    def _run(self, input_ids: np.ndarray, pixels: np.ndarray) -> dict[str, np.ndarray]:
        with self._lock:
            values = self._session.run(None, {"input_ids": input_ids, "pixel_values": pixels})
        return dict(zip(self._outputs, values))

    def embed_images(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        out = self._run(self._label_ids, np.stack([_pixels(img) for img in images]))
        probs = 1 / (1 + np.exp(-out["logits_per_image"]))
        return _unit(out["image_embeds"]).astype(np.float32), probs

    def embed_text(self, text: str) -> np.ndarray:
        out = self._run(self._tokenize([text]), self._blank)
        return _unit(out["text_embeds"])[0].astype(np.float32)

    def probability(self, cosine: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self._scale * cosine + self._bias)))


def load_engine(models_dir: Path) -> SceneEngine:
    """Load the scene engine. Raises ModelsMissing when its files are absent."""
    import os

    if os.environ.get("PIXORA_SCENE_DEVICE") == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    return SigLIP2Engine(models_dir, providers)

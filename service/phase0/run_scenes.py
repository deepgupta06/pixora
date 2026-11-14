"""Phase 0, step 3: zero-shot scene and occasion labels with SigLIP 2.

    python -m phase0.run_scenes --data-dir .data-real --models .models

Scores every photo against the plan's curated label set and writes
<data-dir>/phase0/scenes.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

LABELS = {
    "nature": ["mountain", "beach", "forest", "lake", "desert", "snow"],
    "places": ["mall", "restaurant", "home", "office", "street", "temple", "stadium"],
    "occasions": ["wedding", "birthday", "festival", "graduation", "concert", "party"],
}
# Wording matters for zero-shot models; these read better than the bare word.
PROMPTS = {
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
}
BATCH = 16


def prompt(label: str) -> str:
    return f"this is a photo of {PROMPTS.get(label, 'a ' + label)}."


def load_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((x - 0.5) / 0.5).transpose(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    model_dir = args.models / "siglip2"
    out = args.data_dir / "phase0"
    out.mkdir(parents=True, exist_ok=True)

    labels = [label for group in LABELS.values() for label in group]
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_padding(length=64, pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=64)
    input_ids = np.array([e.ids for e in tokenizer.encode_batch([prompt(l) for l in labels])], dtype=np.int64)

    providers = ["CPUExecutionProvider"] if args.cpu else ["DmlExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), providers=providers)
    outputs = [o.name for o in session.get_outputs()]

    db = sqlite3.connect(args.data_dir / "pixora.db")
    photos = db.execute("SELECT id, content_hash FROM photos WHERE ready = 1 ORDER BY id").fetchall()

    logits, image_embeds, text_embeds = [], [], None
    seconds = 0.0
    for start in range(0, len(photos), BATCH):
        batch = photos[start : start + BATCH]
        pixels = np.stack(
            [load_pixels(args.data_dir / "thumbs" / "1024" / h[:2] / f"{h}.webp") for _, h in batch]
        )
        t = time.perf_counter()
        result = dict(zip(outputs, session.run(None, {"input_ids": input_ids, "pixel_values": pixels})))
        if start > 0:  # the first batch includes warm-up
            seconds += time.perf_counter() - t
        logits.append(result["logits_per_image"])
        image_embeds.append(result["image_embeds"])
        text_embeds = result["text_embeds"]
    logits = np.concatenate(logits)
    image_embeds = np.concatenate(image_embeds)
    probs = 1 / (1 + np.exp(-logits))

    # logits = scale * cosine + bias. Recover both so the product can score
    # stored image embeddings against cached label embeddings without the full model.
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    cosine = (unit(image_embeds) @ unit(text_embeds).T).reshape(-1)
    scale, bias = np.polyfit(cosine, logits.reshape(-1), 1)
    fit_error = float(np.abs(scale * cosine + bias - logits.reshape(-1)).max())

    np.save(out / "scene_image_embeddings.npy", image_embeds.astype(np.float32))
    timed = max(1, len(photos) - BATCH)
    (out / "scenes.json").write_text(
        json.dumps(
            {
                "model": "siglip2-base-patch16-224 (onnx-community, fp32)",
                "providers": providers,
                "labels": LABELS,
                "prompts": {l: prompt(l) for l in labels},
                "ms_per_photo": round(seconds / timed * 1000, 1),
                "embedding_size": int(image_embeds.shape[1]),
                "logit_scale": round(float(scale), 4),
                "logit_bias": round(float(bias), 4),
                "logit_fit_max_error": round(fit_error, 5),
                "photos": [
                    {"photo_id": pid, "hash": h, "probs": [round(float(p), 4) for p in row]}
                    for (pid, h), row in zip(photos, probs)
                ],
            }
        ),
        encoding="utf-8",
    )
    top = probs.argmax(axis=1)
    print("ms/photo", round(seconds / timed * 1000, 1), "scale", scale, "bias", bias, "fit error", fit_error)
    for i, label in enumerate(labels):
        print(f"{label:11} top-1 for {int((top == i).sum()):3} photos; p>=0.1: {int((probs[:, i] >= 0.1).sum()):3}; max {probs[:, i].max():.3f}")
    print("photos whose best label is under 0.05:", int((probs.max(axis=1) < 0.05).sum()))


if __name__ == "__main__":
    main()

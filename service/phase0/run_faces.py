"""Phase 0, step 1: run both face engines over a scanned library.

    python -m phase0.run_faces --data-dir .data-real --models .models

Writes <data-dir>/phase0/faces.npz, faces.json and one crop per detected face.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

from .engines import Face, InsightFaceEngine, OpenCVZooEngine, working_image


def read_image(path: str) -> np.ndarray | None:
    # imdecode applies the EXIF orientation; np.fromfile copes with non-ASCII paths.
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def save_crop(full: np.ndarray, face: Face, dest: Path) -> float:
    """Save a square crop with some margin and return its sharpness."""
    x1, y1, x2, y2 = face.bbox
    side = max(x2 - x1, y2 - y1) * 1.4
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = full.shape[:2]
    left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    right, bottom = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = full[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(dest))
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true", help="run ONNX Runtime on the CPU only")
    args = parser.parse_args()

    out = args.data_dir / "phase0"
    providers = ["CPUExecutionProvider"] if args.cpu else ["DmlExecutionProvider", "CPUExecutionProvider"]
    engines = {
        "insightface": InsightFaceEngine(args.models / "buffalo_l", providers),
        "opencv-zoo": OpenCVZooEngine(args.models / "opencv_zoo"),
    }
    for key in engines:
        (out / "faces" / key).mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(args.data_dir / "pixora.db")
    photos = db.execute(
        "SELECT p.id, p.content_hash, f.path FROM photos p JOIN photo_files f ON f.id = p.primary_file_id "
        "WHERE p.ready = 1 ORDER BY p.id"
    ).fetchall()

    records: list[dict] = []
    embeddings: dict[str, list[np.ndarray]] = {key: [] for key in engines}
    seconds = {key: 0.0 for key in engines}
    decode_seconds = 0.0
    for n, (photo_id, content_hash, path) in enumerate(photos, 1):
        t = time.perf_counter()
        full = read_image(path)
        if full is None:
            print("unreadable:", path)
            continue
        work, scale = working_image(full)
        decode_seconds += time.perf_counter() - t
        for key, engine in engines.items():
            t = time.perf_counter()
            faces = engine.analyze(full, work, scale)
            # The first photo pays for model warm-up, so it is left out of the timing.
            if n > 1:
                seconds[key] += time.perf_counter() - t
            for i, face in enumerate(faces):
                crop = f"faces/{key}/{photo_id}_{i}.jpg"
                sharpness = save_crop(full, face, out / crop)
                eye_l, eye_r, nose = face.kps[0], face.kps[1], face.kps[2]
                eye_dist = max(float(np.linalg.norm(eye_r - eye_l)), 1e-6)
                records.append(
                    {
                        "engine": key,
                        "photo_id": photo_id,
                        "hash": content_hash,
                        "index": len(embeddings[key]),
                        "bbox": [round(float(v), 1) for v in face.bbox],
                        "score": round(face.score, 4),
                        # Face size as the detector saw it, in working-image pixels.
                        "size": round(float(min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * scale), 1),
                        "sharpness": round(sharpness, 1),
                        # 0 is frontal; near 0.5 the nose sits over an eye.
                        "turn": round(abs(float(nose[0] - (eye_l[0] + eye_r[0]) / 2)) / eye_dist, 3),
                        "crop": crop,
                    }
                )
                embeddings[key].append(face.embedding.astype(np.float32))
        if n % 25 == 0:
            print(f"{n}/{len(photos)}")

    np.savez(out / "faces.npz", **{k: np.stack(v) if v else np.zeros((0, 1), np.float32) for k, v in embeddings.items()})
    timed = max(1, len(photos) - 1)
    summary = {
        "photos": len(photos),
        "providers": providers,
        "engines": {k: {"version": e.version, "faces": len(embeddings[k]), "ms_per_photo": round(seconds[k] / timed * 1000, 1)} for k, e in engines.items()},
        "decode_ms_per_photo": round(decode_seconds / len(photos) * 1000, 1),
    }
    (out / "faces.json").write_text(json.dumps({"summary": summary, "faces": records}), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

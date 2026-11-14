"""Face detection and recognition behind the `FaceEngine` interface.

Engines detect on a downscaled working image and compute embeddings from the
larger decoded photo. Every embedding is stored with the engine's version, so
changing engines means reprocessing faces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

WORKING_LONG_SIDE = 1280
# A face is good enough to learn a person from when it is not tiny and not
# strongly turned (values validated in Phase 0).
MIN_LEARN_SIZE = 40
MAX_LEARN_TURN = 0.35

# ArcFace's reference landmark positions in a 112x112 crop.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Thresholds:
    """Cosine-similarity thresholds, calibrated per engine in Phase 0."""

    cluster: float  # unknown faces are grouped while their average similarity stays above this
    auto: float  # assign to a known person above this
    suggest: float  # between `suggest` and `auto`: ask the user


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2 in full-image pixels
    score: float
    # Five points in full-image pixels: eye, eye, nose, mouth corner, mouth corner
    # (image-left first).
    kps: np.ndarray
    embedding: np.ndarray  # L2-normalised

    def size(self, scale: float) -> float:
        """Shorter side of the box, in working-image pixels."""
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]) * scale)

    @property
    def turn(self) -> float:
        """0 is frontal; near 0.5 the nose sits over an eye."""
        eye_l, eye_r, nose = self.kps[0], self.kps[1], self.kps[2]
        eye_dist = max(float(np.linalg.norm(eye_r - eye_l)), 1e-6)
        return abs(float(nose[0] - (eye_l[0] + eye_r[0]) / 2)) / eye_dist


class FaceEngine(Protocol):
    name: str
    version: str
    thresholds: Thresholds

    def analyze(self, full: np.ndarray, work: np.ndarray, scale: float) -> list[Face]:
        """`full` is a BGR image and `work` is `full` resized by `scale` (<= 1)."""
        ...


class ModelsMissing(Exception):
    """The engine's model files have not been downloaded yet."""


def working_image(full: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = full.shape[:2]
    scale = min(1.0, WORKING_LONG_SIDE / max(h, w))
    if scale == 1.0:
        return full, 1.0
    work = cv2.resize(full, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return work, scale


def is_learnable(face: Face, scale: float) -> bool:
    return face.size(scale) >= MIN_LEARN_SIZE and face.turn <= MAX_LEARN_TURN


def crop_face(full: np.ndarray, face: Face, side: int = 160) -> np.ndarray:
    """A square BGR crop around the face, with some margin."""
    x1, y1, x2, y2 = face.bbox
    half = max(x2 - x1, y2 - y1) * 0.7
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    matrix = np.array([[side / (2 * half), 0, side / 2 - cx * side / (2 * half)],
                       [0, side / (2 * half), side / 2 - cy * side / (2 * half)]], dtype=np.float32)
    return cv2.warpAffine(full, matrix, (side, side), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ModelsMissing("Missing model files: " + ", ".join(missing))


class InsightFaceEngine:
    """SCRFD-10G detection and ArcFace R50 recognition from the `buffalo_l` pack.

    The pretrained models are licensed for non-commercial research use only.
    """

    name = "insightface"
    version = "insightface:buffalo_l:det_10g+w600k_r50"
    thresholds = Thresholds(cluster=0.40, auto=0.45, suggest=0.35)

    def __init__(self, models_dir: Path, providers: list[str], det_thresh: float = 0.5):
        det, rec = models_dir / "buffalo_l" / "det_10g.onnx", models_dir / "buffalo_l" / "w600k_r50.onnx"
        _require(det, rec)
        self.size = WORKING_LONG_SIDE
        # DirectML cannot run this model with dynamic input dimensions, so pin them.
        options = ort.SessionOptions()
        options.add_free_dimension_override_by_name("?", self.size)
        self.det = ort.InferenceSession(str(det), options, providers=providers)
        self.rec = ort.InferenceSession(str(rec), providers=providers)
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name
        self.det_thresh = det_thresh

    def _detect(self, work: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = self.size
        h, w = work.shape[:2]
        scale = size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(work, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.det.run(None, {self.det_input: blob})

        boxes, scores, points = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            score = outs[i].reshape(-1)
            dist = outs[i + 3].reshape(-1, 4) * stride
            kps = outs[i + 6].reshape(-1, 10) * stride
            n = size // stride
            ys, xs = np.mgrid[:n, :n]
            centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centers = np.repeat(centers, 2, axis=0)  # two anchors per cell
            keep = score >= self.det_thresh
            c, d = centers[keep], dist[keep]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1))
            scores.append(score[keep])
            points.append(kps[keep].reshape(-1, 5, 2) + c[:, None, :])
        boxes, scores, points = np.concatenate(boxes), np.concatenate(scores), np.concatenate(points)
        if len(boxes) == 0:
            return boxes, scores, points
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
        keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.det_thresh, 0.4)).reshape(-1)
        return boxes[keep] / scale, scores[keep], points[keep] / scale

    def analyze(self, full: np.ndarray, work: np.ndarray, scale: float) -> list[Face]:
        boxes, scores, points = self._detect(work)
        faces = []
        for box, score, kps in zip(boxes, scores, points):
            kps_full = (kps / scale).astype(np.float32)
            matrix, _ = cv2.estimateAffinePartial2D(kps_full, _ARCFACE_DST, method=cv2.LMEDS)
            if matrix is None:
                continue
            aligned = cv2.warpAffine(full, matrix, (112, 112), borderValue=0)
            blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            emb = self.rec.run(None, {self.rec_input: blob})[0].reshape(-1)
            faces.append(Face(box / scale, float(score), kps_full, _normalise(emb)))
        return faces


class OpenCVZooEngine:
    """YuNet detection and SFace recognition from the OpenCV Zoo (MIT / Apache 2.0)."""

    name = "opencv-zoo"
    version = "opencv-zoo:yunet_2023mar+sface_2021dec"
    thresholds = Thresholds(cluster=0.45, auto=0.50, suggest=0.40)

    def __init__(self, models_dir: Path, det_thresh: float = 0.8):
        det = models_dir / "opencv_zoo" / "face_detection_yunet_2023mar.onnx"
        rec = models_dir / "opencv_zoo" / "face_recognition_sface_2021dec.onnx"
        _require(det, rec)
        self.det = cv2.FaceDetectorYN.create(str(det), "", (320, 320), det_thresh, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec), "")

    def analyze(self, full: np.ndarray, work: np.ndarray, scale: float) -> list[Face]:
        h, w = work.shape[:2]
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(work)
        faces = []
        for row in rows if rows is not None else []:
            row_full = row.copy()
            row_full[:14] /= scale
            aligned = self.rec.alignCrop(full, row_full)
            emb = self.rec.feature(aligned).reshape(-1)
            x, y, bw, bh = row_full[:4]
            faces.append(
                Face(
                    np.array([x, y, x + bw, y + bh], dtype=np.float32),
                    float(row[14]),
                    row_full[4:14].reshape(5, 2).astype(np.float32),
                    _normalise(emb),
                )
            )
        return faces


ENGINE_THRESHOLDS = {
    InsightFaceEngine.version: InsightFaceEngine.thresholds,
    OpenCVZooEngine.version: OpenCVZooEngine.thresholds,
}
DEFAULT_ENGINE = "insightface"


def load_engine(models_dir: Path, name: str | None = None) -> FaceEngine:
    """Load the configured engine. Raises ModelsMissing when its files are absent."""
    name = name or os.environ.get("PIXORA_FACE_ENGINE", DEFAULT_ENGINE)
    if name == "opencv-zoo":
        return OpenCVZooEngine(models_dir)
    if name != "insightface":
        raise ValueError(f"Unknown face engine: {name}")
    if os.environ.get("PIXORA_FACE_DEVICE") == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    return InsightFaceEngine(models_dir, providers)

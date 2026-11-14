"""The face engines now live in the service; Phase 0 scripts import them from here."""

from pathlib import Path

from pixora_service.faces import (  # noqa: F401
    Face,
    FaceEngine,
    OpenCVZooEngine as _OpenCVZooEngine,
    InsightFaceEngine as _InsightFaceEngine,
    working_image,
)


class InsightFaceEngine(_InsightFaceEngine):
    """Phase 0 passes the model pack's own folder."""

    def __init__(self, model_dir: Path, providers: list[str], det_thresh: float = 0.5):
        super().__init__(model_dir.parent, providers, det_thresh)


class OpenCVZooEngine(_OpenCVZooEngine):
    def __init__(self, model_dir: Path, det_thresh: float = 0.8):
        super().__init__(model_dir.parent, det_thresh)

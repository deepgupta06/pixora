from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

THUMB_SIZES = (256, 1024)


def default_data_dir() -> Path:
    override = os.environ.get("PIXORA_DATA_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "Pixora"


@dataclass(frozen=True)
class Config:
    data_dir: Path
    workers: int
    models_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pixora.db"

    def thumb_path(self, content_hash: str, size: int) -> Path:
        return self.data_dir / "thumbs" / str(size) / content_hash[:2] / f"{content_hash}.webp"

    def face_crop_path(self, face_id: int) -> Path:
        return self.data_dir / "faces" / f"{face_id % 256:02x}" / f"{face_id}.webp"

    @classmethod
    def load(
        cls,
        data_dir: Path | None = None,
        workers: int | None = None,
        models_dir: Path | None = None,
    ) -> "Config":
        if workers is None:
            env = os.environ.get("PIXORA_WORKERS")
            # Leave half the cores to the interface and the rest of the PC.
            workers = int(env) if env else max(1, min(6, (os.cpu_count() or 2) // 2))
        data_dir = data_dir or default_data_dir()
        if models_dir is None:
            env = os.environ.get("PIXORA_MODELS_DIR")
            # Model weights are downloaded here; they are never part of the repository.
            models_dir = Path(env) if env else data_dir / "models"
        cfg = cls(data_dir=data_dir, workers=workers, models_dir=models_dir)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        return cfg

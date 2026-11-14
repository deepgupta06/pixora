"""Download the editing and generation models into the models folder.

    python scripts/download_studio_models.py [--models-dir DIR] [model ids...]

With no ids, every model is fetched (about 21 GB). The same download can be
started from the app.
"""

import argparse
from pathlib import Path

from pixora_service.config import Config
from pixora_service.model_store import MODELS, download, is_installed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("models", nargs="*", choices=[*MODELS, []], help="default: all")
    args = parser.parse_args()
    models_dir = Config.load(models_dir=args.models_dir).models_dir
    ids = args.models or list(MODELS)
    for model_id in ids:
        model = MODELS[model_id]
        state = "installed" if is_installed(models_dir, model_id) else f"{model.size_mb:,} MB"
        print(f"{model.label:45} {model.licence:45} {state}")
    download(models_dir, ids, progress=lambda fraction, message: print(f"[{fraction:4.0%}] {message}", flush=True))


if __name__ == "__main__":
    main()

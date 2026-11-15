import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .api import create_app
from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(prog="pixora_service")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PIXORA_PORT", "8732")))
    parser.add_argument("--data-dir", type=Path, help="where the database and thumbnails live")
    parser.add_argument("--models-dir", type=Path, help="where downloaded model files live")
    parser.add_argument("--creations-dir", type=Path, help="where edited and generated pictures are saved")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = Config.load(data_dir=args.data_dir, models_dir=args.models_dir, creations_dir=args.creations_dir)
    logging.getLogger("pixora").info("Data in %s, listening on 127.0.0.1:%s", config.data_dir, args.port)
    # Loopback only: the service is never reachable from the network.
    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

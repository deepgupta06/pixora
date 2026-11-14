"""Scan one folder into a data directory without starting the API.

    python scripts/scan_folder.py <photo folder> --data-dir <data dir>

Used for model tests and benchmarks. The photo folder is only read.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pixora_service.config import Config
from pixora_service.db import Database
from pixora_service.scanner import ScanManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    config = Config.load(data_dir=args.data_dir)
    db = Database(config.db_path)
    with db.write() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO folders(path, added_at) VALUES(?, ?)",
            (os.path.abspath(args.folder), datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    scanner = ScanManager(db, config)
    started = time.perf_counter()
    scanner.start(walk=True)
    scanner.wait_idle(timeout=24 * 3600)
    status = scanner.status()
    print(f"{time.perf_counter() - started:.1f}s", status["library"], "errors:", status["errors"])
    for row in db.conn().execute(
        "SELECT f.path, j.error FROM jobs j JOIN photos p ON p.id = j.photo_id "
        "JOIN photo_files f ON f.photo_id = p.id WHERE j.status = 'error'"
    ):
        print("  failed:", row["path"], "-", row["error"])


if __name__ == "__main__":
    main()

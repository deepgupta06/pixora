"""Run the face stage on a scanned data directory, then answer every question
automatically, to measure how many questions the naming flow asks.

    python scripts/simulate_naming.py <data dir>

Every "Who is this?" is answered with a placeholder name and every
"Is this also NAME?" with no. Use a copy of a data directory: it writes people.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pixora_service.config import Config
from pixora_service.db import Database
from pixora_service.scanner import ScanManager


def main() -> None:
    config = Config.load(data_dir=Path(sys.argv[1]))
    db = Database(config.db_path)
    scanner = ScanManager(db, config)
    conn = db.conn()
    count = lambda sql: conn.execute(sql).fetchone()[0]

    started = time.perf_counter()
    scanner.start(walk=False)
    scanner.wait_idle(3600)
    print(
        f"face stage: {time.perf_counter() - started:.1f}s, "
        f"{count("SELECT count(*) FROM jobs WHERE stage='faces' AND status='done'")} photos, "
        f"{count('SELECT count(*) FROM faces')} faces, "
        f"{count('SELECT count(*) FROM faces WHERE quality=1')} learnable, note: {scanner.status()['faces_note']}"
    )

    people = scanner.people
    names = follow_ups = 0
    while (q := people.next_question())["type"] != "done":
        if q["type"] == "name":
            names += 1
            before = count("SELECT count(*) FROM faces WHERE person_id IS NOT NULL")
            people.name_faces(q["face_ids"], f"Person {names}")
            gained = count("SELECT count(*) FROM faces WHERE person_id IS NOT NULL") - before
            print(f"  Person {names}: named from {len(q['face_ids'])} faces, {gained - len(q['face_ids'])} more matched")
        else:
            follow_ups += 1
            print(f"  follow-up: is this also {q['name']}? {len(q['face_ids'])} faces {q['face_ids'][:6]}")
            people.confirm(q["person_id"], q["face_ids"], False)

    print(f"name questions: {names}, follow-up questions: {follow_ups}")
    print(
        "assigned:", count("SELECT count(*) FROM faces WHERE person_id IS NOT NULL"),
        "| by user:", count("SELECT count(*) FROM faces WHERE assignment_source='user'"),
        "| automatic:", count("SELECT count(*) FROM faces WHERE assignment_source='auto'"),
    )
    print(
        "unassigned learnable:", count("SELECT count(*) FROM faces WHERE person_id IS NULL AND quality=1"),
        "| unassigned low quality:", count("SELECT count(*) FROM faces WHERE person_id IS NULL AND quality=0"),
        "| with a suggestion:", count("SELECT count(*) FROM faces WHERE person_id IS NULL AND suggested_person_id IS NOT NULL"),
    )
    print(
        "same person twice in one photo (certain mistakes):",
        count(
            "SELECT count(*) FROM faces a JOIN faces b "
            "ON a.photo_id = b.photo_id AND a.id < b.id AND a.person_id = b.person_id"
        ),
    )


if __name__ == "__main__":
    main()

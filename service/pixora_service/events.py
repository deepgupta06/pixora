"""Events: runs of photos taken close together in time and place.

Occasions are inferred per event, by combining the scene scores of all its
photos, because a single photo is often weak evidence.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from .db import Database
from .scenes import LABELS, TAG_FLOOR

# A new event starts after this long without a photo...
TIME_GAP = timedelta(hours=6)
# ...or sooner, when the camera has clearly moved somewhere else.
MOVE_GAP = timedelta(hours=1)
MOVE_KM = 25.0
# Smaller runs are still events, but too small to name an occasion from.
MIN_OCCASION_PHOTOS = 4
# An occasion or place names the event when this share of its photos shows it.
OCCASION_SHARE = 0.25
PLACE_SHARE = 0.3


def _km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def _starts_new_event(prev, cur) -> bool:
    gap = datetime.fromisoformat(cur["taken_at"]) - datetime.fromisoformat(prev["taken_at"])
    if gap > TIME_GAP:
        return True
    located = prev["lat"] is not None and cur["lat"] is not None
    return located and gap > MOVE_GAP and _km(prev, cur) > MOVE_KM


def rebuild(db: Database) -> None:
    """Regroup every visible photo into events and infer each event's occasion."""
    conn = db.conn()
    photos = conn.execute(
        "SELECT id, taken_at, lat, lon FROM photos WHERE ready = 1 AND primary_file_id IS NOT NULL "
        "AND duplicate_of_photo_id IS NULL ORDER BY taken_at, id"
    ).fetchall()
    groups: list[list] = []
    for photo in photos:
        if groups and not _starts_new_event(groups[-1][-1], photo):
            groups[-1].append(photo)
        else:
            groups.append([photo])

    # For every photo, the labels that apply to it.
    tagged: dict[int, list[tuple[str, str]]] = {}
    for r in conn.execute(
        "SELECT pt.photo_id, t.name, t.category FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
        "WHERE pt.score >= ?",
        (TAG_FLOOR,),
    ):
        tagged.setdefault(r["photo_id"], []).append((r["name"], r["category"]))

    with db.write() as conn:
        conn.execute("UPDATE photos SET event_id = NULL WHERE event_id IS NOT NULL")
        conn.execute("DELETE FROM events")
        for group in groups:
            occasion, title = None, ""
            if len(group) >= MIN_OCCASION_PHOTOS:
                share: dict[tuple[str, str], float] = {}
                for photo in group:
                    for label in tagged.get(photo["id"], ()):
                        share[label] = share.get(label, 0) + 1 / len(group)
                occasions = {name: s for (name, cat), s in share.items() if cat == "occasions"}
                places = {name: s for (name, cat), s in share.items() if cat != "occasions"}
                if occasions and max(occasions.values()) >= OCCASION_SHARE:
                    # Ties go to the label listed first in the curated set.
                    occasion = max(occasions, key=lambda n: (occasions[n], -LABELS["occasions"].index(n)))
                if places and max(places.values()) >= PLACE_SHARE:
                    title = max(places, key=lambda n: (places[n], n))
            event_id = conn.execute(
                "INSERT INTO events(title, type, start_at, end_at) VALUES(?, ?, ?, ?)",
                (title, occasion, group[0]["taken_at"], group[-1]["taken_at"]),
            ).lastrowid
            conn.executemany("UPDATE photos SET event_id = ? WHERE id = ?", [(event_id, p["id"]) for p in group])

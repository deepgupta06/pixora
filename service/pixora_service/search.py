"""Natural-language search over the stored photo embeddings."""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from .db import Database

# A photo matches a query at or above this probability. Checked by eye on real
# photos: things absent from the library score near zero, real matches clear it.
SEARCH_FLOOR = 0.002
_QUERY_CACHE = 64


class SearchIndex:
    """Keeps the embedding matrix in memory and reloads it when photos are added."""

    def __init__(self, db: Database, get_engine):
        self.db = db
        self._get_engine = get_engine  # raises ModelsMissing when the model is not installed
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._ids = np.zeros(0, dtype=np.int64)
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._queries: OrderedDict[str, np.ndarray] = OrderedDict()

    def _load(self, version: str) -> None:
        conn = self.db.conn()
        stamp = (version, *conn.execute(
            "SELECT count(*), coalesce(max(rowid), 0) FROM photo_embeddings WHERE model_version = ?", (version,)
        ).fetchone())
        if stamp == self._stamp:
            return
        rows = conn.execute(
            "SELECT photo_id, embedding FROM photo_embeddings WHERE model_version = ? ORDER BY photo_id", (version,)
        ).fetchall()
        self._ids = np.array([r["photo_id"] for r in rows], dtype=np.int64)
        self._matrix = (
            np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 1), dtype=np.float32)
        )
        self._stamp = stamp

    def photo_ids(self, query: str) -> list[int]:
        """Photos that match the query, best first."""
        engine = self._get_engine()
        key = " ".join(query.lower().split())
        with self._lock:
            self._load(engine.version)
            if len(self._ids) == 0 or not key:
                return []
            vector = self._queries.get(key)
            if vector is None:
                vector = engine.embed_text(key)
                self._queries[key] = vector
                if len(self._queries) > _QUERY_CACHE:
                    self._queries.popitem(last=False)
            else:
                self._queries.move_to_end(key)
            scores = engine.probability(self._matrix @ vector)
            order = np.argsort(-scores)
            return [int(self._ids[i]) for i in order if scores[i] >= SEARCH_FLOOR]

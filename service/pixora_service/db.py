from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Tables for later phases (tags, embeddings, events, layouts) are added with
# their phase; their column types depend on choices made there.
SCHEMA_V1 = """
CREATE TABLE folders(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at TEXT NOT NULL
);

-- One row per unique image. `ready` is set once metadata and thumbnails exist.
CREATE TABLE photos(
    id INTEGER PRIMARY KEY,
    primary_file_id INTEGER,
    content_hash TEXT NOT NULL UNIQUE,
    perceptual_hash TEXT,
    duplicate_of_photo_id INTEGER REFERENCES photos(id),
    taken_at TEXT,
    taken_at_source TEXT,
    lat REAL,
    lon REAL,
    width INTEGER,
    height INTEGER,
    camera TEXT,
    event_id INTEGER,
    ready INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX photos_grid ON photos(taken_at DESC, id DESC)
    WHERE ready = 1 AND primary_file_id IS NOT NULL AND duplicate_of_photo_id IS NULL;

-- One row per file on disk. Several files can point at the same photo.
-- content_hash NULL means the file still needs hashing.
CREATE TABLE photo_files(
    id INTEGER PRIMARY KEY,
    photo_id INTEGER REFERENCES photos(id),
    folder_id INTEGER NOT NULL REFERENCES folders(id),
    path TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    modified_at INTEGER NOT NULL,
    content_hash TEXT,
    missing INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX photo_files_photo ON photo_files(photo_id);
CREATE INDEX photo_files_folder ON photo_files(folder_id);
CREATE INDEX photo_files_pending ON photo_files(id) WHERE content_hash IS NULL;

CREATE TABLE jobs(
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(photo_id, stage)
);
CREATE INDEX jobs_status ON jobs(stage, status);

CREATE TABLE settings(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# Phase 2: people. Embeddings are float32 blobs compared in memory.
SCHEMA_V2 = """
CREATE TABLE people(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    cover_face_id INTEGER,
    hidden INTEGER NOT NULL DEFAULT 0,
    -- follow-up questions already asked about this person (at most three)
    questions_asked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE faces(
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id),
    -- box as fractions of the upright photo's width and height
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    score REAL NOT NULL,
    size REAL NOT NULL,
    turn REAL NOT NULL,
    quality INTEGER NOT NULL,  -- 1: clear enough to learn a person from
    embedding BLOB NOT NULL,
    engine_version TEXT NOT NULL,
    person_id INTEGER REFERENCES people(id),
    assignment_source TEXT,  -- 'user' or 'auto'
    confidence REAL,
    suggested_person_id INTEGER REFERENCES people(id),
    cluster_id INTEGER,  -- group of unknown faces; recomputed, not stable
    dismissed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX faces_photo ON faces(photo_id);
CREATE INDEX faces_person ON faces(person_id);
CREATE INDEX faces_cluster ON faces(cluster_id) WHERE cluster_id IS NOT NULL;

CREATE TABLE person_prototypes(
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id),
    embedding BLOB NOT NULL,
    engine_version TEXT NOT NULL
);
CREATE INDEX person_prototypes_person ON person_prototypes(person_id);

CREATE TABLE face_constraints(
    face_id INTEGER NOT NULL REFERENCES faces(id),
    person_id INTEGER NOT NULL REFERENCES people(id),
    type TEXT NOT NULL,  -- 'not': the user said this face is not this person
    PRIMARY KEY(face_id, person_id, type)
);
"""

# Phase 3: scenes, search, and events.
SCHEMA_V3 = """
CREATE TABLE tags(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL  -- nature | places | occasions
);

CREATE TABLE photo_tags(
    photo_id INTEGER NOT NULL REFERENCES photos(id),
    tag_id INTEGER NOT NULL REFERENCES tags(id),
    score REAL NOT NULL,
    PRIMARY KEY(photo_id, tag_id)
);
CREATE INDEX photo_tags_tag ON photo_tags(tag_id, score);

CREATE TABLE photo_embeddings(
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id),
    embedding BLOB NOT NULL,
    model_version TEXT NOT NULL
);

-- Rebuilt after every scan, so event ids are not stable yet.
CREATE TABLE events(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,  -- the place most of its photos show, or empty
    type TEXT,  -- the inferred occasion, or NULL
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL
);
CREATE INDEX photos_event ON photos(event_id);
"""

# Editing and generation. Results are new files; originals are never changed.
SCHEMA_V4 = """
-- NULL: the user's own photo. 'edited' or 'generated': a picture Pixora made.
ALTER TABLE photos ADD COLUMN origin TEXT;

-- How each picture Pixora made came to be, so it can be made again.
CREATE TABLE creations(
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id),
    kind TEXT NOT NULL,  -- edited | generated
    operation TEXT NOT NULL,
    source_photo_id INTEGER REFERENCES photos(id),
    params TEXT NOT NULL,  -- JSON: prompt, seed, style and the other settings used
    created_at TEXT NOT NULL
);
CREATE INDEX creations_photo ON creations(photo_id);
"""

MIGRATIONS = [SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4]
SCHEMA_VERSION = len(MIGRATIONS)


class Database:
    """SQLite in WAL mode: one connection per thread, writes serialised by a lock."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _migrate(self) -> None:
        conn = self.conn()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"Database schema version {version} is newer than this service")
        for number, script in enumerate(MIGRATIONS[version:], version + 1):
            conn.executescript(f"BEGIN; {script} PRAGMA user_version={number}; COMMIT;")

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.write() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def close_thread(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

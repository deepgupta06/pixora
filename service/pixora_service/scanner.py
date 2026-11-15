"""Background scan: discover files, hash them, collapse exact duplicates, make thumbnails.

All progress lives in the database, so a scan resumes after a restart:
  - a `photo_files` row without a content hash still needs hashing
  - a `jobs` row that is not `done` still needs its stage run
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime, timezone

import cv2

from PIL import Image

from . import events, media, scenes
from . import faces as face_engines
from .config import Config
from .db import Database
from .people import People

log = logging.getLogger("pixora.scanner")

STAGE_THUMBNAIL = "thumbnail"
STAGE_FACES = "faces"
STAGE_SCENES = "scenes"
# Known people are matched and unknown faces regrouped this often during a scan.
_PEOPLE_REFRESH_EVERY = 250
_SKIP_DIRS = frozenset({"$recycle.bin", "system volume information"})
_BATCH = 500

_REPAIR_PRIMARY = """
UPDATE photos SET primary_file_id = COALESCE(
    (SELECT f.id FROM photo_files f
      WHERE f.id = photos.primary_file_id AND f.photo_id = photos.id AND f.missing = 0),
    (SELECT min(f.id) FROM photo_files f WHERE f.photo_id = photos.id AND f.missing = 0))
WHERE id = ?
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repair_primaries(conn, photo_ids) -> None:
    """Keep each photo's primary file valid, or NULL when no copy is left on disk."""
    conn.executemany(_REPAIR_PRIMARY, [(pid,) for pid in set(photo_ids) if pid is not None])


class ScanManager:
    def __init__(self, db: Database, config: Config, engine_factory=None, scene_factory=None):
        self.db = db
        self.config = config
        self.people = People(db)
        self._engine_factory = engine_factory or (lambda: face_engines.load_engine(config.models_dir))
        self._engine: face_engines.FaceEngine | None = None
        self._faces_note: str | None = None
        self._scene_factory = scene_factory or (lambda: scenes.load_engine(config.models_dir))
        self._scene_engine: scenes.SceneEngine | None = None
        self._scene_lock = threading.Lock()
        self._scenes_note: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._unpaused = threading.Event()
        # Cleared while an edit or generation job has the graphics card.
        self._free = threading.Event()
        self._free.set()
        self._rerun_walk = False
        self._phase = "idle"  # idle | discovering | processing | faces | scenes
        self._discovered = 0
        self._folder_errors: dict[int, str] = {}

        if db.get_setting("scan_paused") != "1":
            self._unpaused.set()
        # Anything left "running" was interrupted by the previous shutdown.
        with db.write() as conn:
            conn.execute("UPDATE jobs SET status='pending' WHERE status='running'")

    # ---- control -----------------------------------------------------------

    @property
    def paused(self) -> bool:
        return not self._unpaused.is_set()

    def start(self, walk: bool = True) -> None:
        """Start a scan, or ask the running one to pick up new work when it finishes."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._rerun_walk = self._rerun_walk or walk
                return
            self._stop.clear()
            thread = threading.Thread(target=self._run, args=(walk,), name="pixora-scan", daemon=True)
            # Started before it is published: other threads may join it at once.
            thread.start()
            self._thread = thread

    def pause(self) -> None:
        self._unpaused.clear()
        self.db.set_setting("scan_paused", "1")

    def resume(self) -> None:
        self.db.set_setting("scan_paused", "0")
        self._unpaused.set()
        if not (self._thread and self._thread.is_alive()):
            # A cheap incremental walk: the pause may have preceded discovery.
            self.start(walk=True)

    def shutdown(self) -> None:
        self._stop.set()
        self._unpaused.set()  # release anything waiting on the pause gate
        self._free.set()
        thread = self._thread
        if thread:
            thread.join(timeout=30)

    def hold(self, held: bool) -> None:
        """Make the scan wait (without pausing it) while something else needs the PC."""
        if held:
            self._free.clear()
        else:
            self._free.set()

    def release_models(self) -> None:
        """Drop the face and scene models; they load again when next needed."""
        self._engine = None
        with self._scene_lock:
            self._scene_engine = None

    def import_file(self, path: str, folder: str) -> int | None:
        """Add one new file to the library right away and return its photo.

        `folder` joins the library if no library folder contains the file yet,
        so later scans keep seeing it.
        """
        st = os.stat(path)
        with self.db.write() as conn:
            parent = os.path.normcase(os.path.abspath(path))
            folder_id = None
            for f in conn.execute("SELECT id, path FROM folders"):
                if parent.startswith(os.path.normcase(f["path"]).rstrip("\\/") + os.sep):
                    folder_id = f["id"]
                    break
            if folder_id is None:
                folder_id = conn.execute(
                    "INSERT INTO folders(path, added_at) VALUES(?, ?)", (folder, _now())
                ).lastrowid
            conn.execute(
                "INSERT INTO photo_files(folder_id, path, size, modified_at) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET size=excluded.size, modified_at=excluded.modified_at, "
                "content_hash=NULL, photo_id=NULL, missing=0, error=NULL",
                (folder_id, path, st.st_size, st.st_mtime_ns),
            )
            file_id = conn.execute("SELECT id FROM photo_files WHERE path=?", (path,)).fetchone()["id"]
        self._process_file(file_id)
        row = self.db.conn().execute(
            "SELECT pf.photo_id FROM photo_files pf JOIN photos p ON p.id = pf.photo_id "
            "WHERE pf.id=? AND p.ready=1",
            (file_id,),
        ).fetchone()
        return row["photo_id"] if row else None

    def wait_idle(self, timeout: float = 60) -> bool:
        thread = self._thread
        if thread:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _gate(self) -> bool:
        """Block while paused. Returns False when the scan should stop."""
        self._unpaused.wait()
        self._free.wait()
        return not self._stop.is_set()

    # ---- status ------------------------------------------------------------

    def status(self) -> dict:
        row = self.db.conn().execute(
            """
            SELECT
              (SELECT count(*) FROM photo_files WHERE missing=0) AS files,
              (SELECT count(*) FROM photo_files
                WHERE content_hash IS NULL AND error IS NULL AND missing=0) AS unhashed,
              (SELECT count(*) FROM photo_files WHERE error IS NOT NULL AND missing=0) AS file_errors,
              (SELECT count(*) FROM photos WHERE primary_file_id IS NOT NULL) AS photos,
              (SELECT count(*) FROM photos
                WHERE primary_file_id IS NOT NULL AND ready=1) AS photos_ready,
              (SELECT count(*) FROM jobs WHERE stage='thumbnail' AND status='error') AS job_errors,
              (SELECT count(*) FROM jobs j JOIN photos p ON p.id = j.photo_id
                WHERE j.stage='faces' AND j.status IN ('done', 'error')
                  AND p.primary_file_id IS NOT NULL) AS faces_done,
              (SELECT count(*) FROM jobs j JOIN photos p ON p.id = j.photo_id
                WHERE j.stage='scenes' AND j.status IN ('done', 'error')
                  AND p.primary_file_id IS NOT NULL) AS scenes_done,
              (SELECT count(*) FROM photo_files
                WHERE missing=0 AND photo_id IS NOT NULL) AS linked_files
            """
        ).fetchone()
        active = bool(self._thread and self._thread.is_alive())
        if self.paused:
            state = "paused"
        elif active:
            state = self._phase
        else:
            state = "idle"
        hashable = row["files"] - row["file_errors"]
        return {
            "state": state,
            "paused": self.paused,
            "discovered": self._discovered,
            "stages": {
                "hash": {"done": hashable - row["unhashed"], "total": hashable},
                # Files not hashed yet may still turn out to be new photos.
                "thumbnail": {
                    "done": row["photos_ready"],
                    "total": row["photos"] - row["job_errors"] + row["unhashed"],
                },
                "faces": {
                    "done": row["faces_done"],
                    "total": row["photos"] - row["job_errors"] + row["unhashed"],
                },
                "scenes": {
                    "done": row["scenes_done"],
                    "total": row["photos"] - row["job_errors"] + row["unhashed"],
                },
            },
            "faces_note": self._faces_note,
            "scenes_note": self._scenes_note,
            "errors": row["file_errors"] + row["job_errors"],
            "library": {
                "photos": row["photos_ready"],
                "files": row["files"],
                "duplicate_files": row["linked_files"] - row["photos"],
            },
            "folder_errors": [
                {"folder_id": fid, "error": err} for fid, err in self._folder_errors.items()
            ],
        }

    # ---- scan thread -------------------------------------------------------

    def _run(self, walk: bool) -> None:
        try:
            while True:
                if walk:
                    self._phase = "discovering"
                    self._discovered = 0
                    folders = self.db.conn().execute("SELECT id, path FROM folders").fetchall()
                    for folder in folders:
                        if not self._gate():
                            return
                        self._walk(folder["id"], folder["path"])
                self._phase = "processing"
                self._process()
                # AI work comes after every photo is visible in the grid.
                self._phase = "faces"
                self._process_faces()
                self._phase = "scenes"
                self._process_scenes()
                if not self._stop.is_set():
                    events.rebuild(self.db)
                with self._lock:
                    if self._stop.is_set() or not self._rerun_walk:
                        self._phase = "idle"
                        self._thread = None
                        return
                    self._rerun_walk = False
                    walk = True
        except Exception:
            log.exception("Scan failed")
            self._phase = "idle"
        finally:
            self.db.close_thread()

    def _walk(self, folder_id: int, root: str) -> None:
        try:
            os.scandir(root).close()
        except OSError as e:
            # An unplugged drive must not mark the whole folder as missing.
            self._folder_errors[folder_id] = str(e)
            return
        self._folder_errors.pop(folder_id, None)

        existing = {
            r["path"]: r
            for r in self.db.conn().execute(
                "SELECT id, path, size, modified_at, missing, photo_id FROM photo_files "
                "WHERE folder_id=?",
                (folder_id,),
            )
        }
        seen: set[str] = set()
        new: list[tuple] = []
        changed: list[tuple] = []
        restored: list[int] = []
        touched_photos: list[int] = []

        def flush() -> None:
            if not (new or changed or restored):
                return
            with self.db.write() as conn:
                # A path already owned by another library folder is left with it.
                conn.executemany(
                    "INSERT OR IGNORE INTO photo_files(folder_id, path, size, modified_at) "
                    "VALUES(?, ?, ?, ?)",
                    new,
                )
                conn.executemany(
                    "UPDATE photo_files SET size=?, modified_at=?, content_hash=NULL, "
                    "photo_id=NULL, missing=0, error=NULL WHERE id=?",
                    changed,
                )
                conn.executemany(
                    "UPDATE photo_files SET missing=0 WHERE id=?", [(i,) for i in restored]
                )
                repair_primaries(conn, touched_photos)
            new.clear(), changed.clear(), restored.clear(), touched_photos.clear()

        stack = [root]
        while stack:
            if not self._gate():
                flush()
                return
            try:
                entries = list(os.scandir(stack.pop()))
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        name = entry.name.lower()
                        if not name.startswith(".") and name not in _SKIP_DIRS:
                            stack.append(entry.path)
                        continue
                    if not media.is_supported(entry.name) or not entry.is_file():
                        continue
                    st = entry.stat()
                except OSError:
                    continue
                path = entry.path
                seen.add(path)
                self._discovered += 1
                rec = existing.get(path)
                if rec is None:
                    new.append((folder_id, path, st.st_size, st.st_mtime_ns))
                elif rec["size"] != st.st_size or rec["modified_at"] != st.st_mtime_ns:
                    changed.append((st.st_size, st.st_mtime_ns, rec["id"]))
                    touched_photos.append(rec["photo_id"])
                elif rec["missing"]:
                    restored.append(rec["id"])
                    touched_photos.append(rec["photo_id"])
                if len(new) + len(changed) + len(restored) >= _BATCH:
                    flush()
        flush()

        gone = [r for p, r in existing.items() if p not in seen and not r["missing"]]
        if gone:
            with self.db.write() as conn:
                conn.executemany(
                    "UPDATE photo_files SET missing=1 WHERE id=?", [(r["id"],) for r in gone]
                )
                repair_primaries(conn, [r["photo_id"] for r in gone])

    def _process(self) -> None:
        conn = self.db.conn()
        work: queue.SimpleQueue = queue.SimpleQueue()
        # Photos already identified come first: they are closest to being visible.
        for r in conn.execute(
            "SELECT photo_id FROM jobs WHERE stage=? AND status='pending'", (STAGE_THUMBNAIL,)
        ):
            work.put(("job", r[0]))
        for r in conn.execute(
            "SELECT id FROM photo_files "
            "WHERE content_hash IS NULL AND error IS NULL AND missing=0 ORDER BY id"
        ):
            work.put(("file", r[0]))
        if work.empty():
            return

        def worker() -> None:
            try:
                while self._gate():
                    try:
                        kind, item_id = work.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        if kind == "file":
                            self._process_file(item_id)
                        else:
                            self._run_thumbnail_job(item_id, None)
                    except Exception:
                        log.exception("Unexpected failure on %s %s", kind, item_id)
            finally:
                self.db.close_thread()

        threads = [
            threading.Thread(target=worker, name=f"pixora-work-{i}", daemon=True)
            for i in range(self.config.workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _process_file(self, file_id: int) -> None:
        row = self.db.conn().execute(
            "SELECT path, size FROM photo_files WHERE id=? AND content_hash IS NULL AND missing=0",
            (file_id,),
        ).fetchone()
        if row is None:
            return
        try:
            content_hash, data = media.read_and_hash(row["path"], row["size"])
        except OSError as e:
            with self.db.write() as conn:
                conn.execute("UPDATE photo_files SET error=? WHERE id=?", (str(e), file_id))
            return

        with self.db.write() as conn:
            photo = conn.execute(
                "SELECT id, primary_file_id FROM photos WHERE content_hash=?", (content_hash,)
            ).fetchone()
            if photo is not None:
                # Same content as a known photo: remember the path, show it once.
                photo_id, is_new = photo["id"], False
                if photo["primary_file_id"] is None:
                    conn.execute(
                        "UPDATE photos SET primary_file_id=? WHERE id=?", (file_id, photo_id)
                    )
            else:
                is_new = True
                photo_id = conn.execute(
                    "INSERT INTO photos(content_hash, primary_file_id) VALUES(?, ?)",
                    (content_hash, file_id),
                ).lastrowid
                conn.execute(
                    "INSERT INTO jobs(photo_id, stage, status, updated_at) "
                    "VALUES(?, ?, 'running', ?)",
                    (photo_id, STAGE_THUMBNAIL, _now()),
                )
            conn.execute(
                "UPDATE photo_files SET photo_id=?, content_hash=? WHERE id=?",
                (photo_id, content_hash, file_id),
            )
        if is_new:
            self._run_thumbnail_job(photo_id, data)

    def _run_thumbnail_job(self, photo_id: int, data: bytes | None) -> None:
        row = self.db.conn().execute(
            "SELECT p.content_hash, f.path, f.modified_at FROM photos p "
            "LEFT JOIN photo_files f ON f.id = p.primary_file_id WHERE p.id=?",
            (photo_id,),
        ).fetchone()
        error: str | None = None
        info = None
        if row is None or row["path"] is None:
            error = "No file available for this photo"
        else:
            try:
                info = media.process_image(
                    data if data is not None else row["path"],
                    self.config.thumb_path(row["content_hash"], 256),
                    self.config.thumb_path(row["content_hash"], 1024),
                    row["modified_at"],
                    path=row["path"],
                )
            except Exception as e:  # corrupt or unsupported image data
                error = f"{type(e).__name__}: {e}"

        with self.db.write() as conn:
            if info is not None:
                conn.execute(
                    "UPDATE photos SET width=?, height=?, taken_at=?, taken_at_source=?, "
                    "lat=?, lon=?, camera=?, perceptual_hash=?, ready=1 WHERE id=?",
                    (
                        info.width,
                        info.height,
                        info.taken_at,
                        info.taken_at_source,
                        info.lat,
                        info.lon,
                        info.camera,
                        info.perceptual_hash,
                        photo_id,
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO jobs(photo_id, stage, status, updated_at) "
                    "VALUES(?, ?, 'pending', ?)",
                    (photo_id, STAGE_FACES, _now()),
                )
            conn.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE photo_id=? AND stage=?",
                ("done" if info is not None else "error", error, _now(), photo_id, STAGE_THUMBNAIL),
            )

    # ---- faces ---------------------------------------------------------------

    def _process_faces(self) -> None:
        with self.db.write() as conn:
            # Photos scanned before this stage existed get a job too.
            conn.execute(
                "INSERT OR IGNORE INTO jobs(photo_id, stage, status, updated_at) "
                "SELECT id, ?, 'pending', ? FROM photos WHERE ready = 1 AND primary_file_id IS NOT NULL",
                (STAGE_FACES, _now()),
            )
        pending = [
            r[0]
            for r in self.db.conn().execute(
                "SELECT j.photo_id FROM jobs j JOIN photos p ON p.id = j.photo_id "
                "WHERE j.stage = ? AND j.status = 'pending' AND p.primary_file_id IS NOT NULL "
                "ORDER BY p.taken_at DESC",
                (STAGE_FACES,),
            )
        ]
        if not pending:
            return
        if self._engine is None:
            try:
                self._engine = self._engine_factory()
            except face_engines.ModelsMissing as e:
                # Jobs stay pending and run once the models are installed.
                self._faces_note = "Face models are not installed"
                log.warning("%s", e)
                return
            except Exception:
                self._faces_note = "The face engine could not start"
                log.exception("Face engine failed to load")
                return
        self._faces_note = None

        for n, photo_id in enumerate(pending, 1):
            if not self._gate():
                return
            self._run_faces_job(photo_id)
            if n % _PEOPLE_REFRESH_EVERY == 0:
                self.people.refresh()
        self.people.refresh()

    def _run_faces_job(self, photo_id: int) -> None:
        engine = self._engine
        row = self.db.conn().execute(
            "SELECT f.path FROM photos p JOIN photo_files f ON f.id = p.primary_file_id WHERE p.id=?",
            (photo_id,),
        ).fetchone()
        error: str | None = None
        found: list[face_engines.Face] = []
        full = None
        scale = 1.0
        if row is None:
            error = "No file available for this photo"
        else:
            try:
                full = media.load_bgr(row["path"])
                work, scale = face_engines.working_image(full)
                found = engine.analyze(full, work, scale)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

        with self.db.write() as conn:
            if error is None:
                h, w = full.shape[:2]
                for face in found:
                    x1, y1, x2, y2 = (float(v) for v in face.bbox)
                    face_id = conn.execute(
                        "INSERT INTO faces(photo_id, x1, y1, x2, y2, score, size, turn, quality, "
                        "embedding, engine_version) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            photo_id,
                            x1 / w, y1 / h, x2 / w, y2 / h,
                            face.score,
                            face.size(scale),
                            face.turn,
                            int(face_engines.is_learnable(face, scale)),
                            face.embedding.astype("float32").tobytes(),
                            engine.version,
                        ),
                    ).lastrowid
                    dest = self.config.face_crop_path(face_id)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    ok, data = cv2.imencode(".webp", face_engines.crop_face(full, face), [cv2.IMWRITE_WEBP_QUALITY, 85])
                    if ok:
                        data.tofile(str(dest))
            conn.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE photo_id=? AND stage=?",
                ("done" if error is None else "error", error, _now(), photo_id, STAGE_FACES),
            )

    # ---- scenes --------------------------------------------------------------

    def scene_engine(self) -> scenes.SceneEngine:
        """The shared scene engine, loaded on first use. Raises ModelsMissing."""
        with self._scene_lock:
            if self._scene_engine is None:
                self._scene_engine = self._scene_factory()
            return self._scene_engine

    def _process_scenes(self) -> None:
        with self.db.write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO jobs(photo_id, stage, status, updated_at) "
                "SELECT id, ?, 'pending', ? FROM photos WHERE ready = 1 AND primary_file_id IS NOT NULL",
                (STAGE_SCENES, _now()),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO tags(name, category) VALUES(?, ?)", scenes.all_labels()
            )
        conn = self.db.conn()
        pending = conn.execute(
            "SELECT j.photo_id, p.content_hash FROM jobs j JOIN photos p ON p.id = j.photo_id "
            "WHERE j.stage = ? AND j.status = 'pending' AND p.primary_file_id IS NOT NULL "
            "ORDER BY p.taken_at DESC",
            (STAGE_SCENES,),
        ).fetchall()
        if not pending:
            return
        try:
            engine = self.scene_engine()
        except face_engines.ModelsMissing as e:
            self._scenes_note = "The scene model is not installed"
            log.warning("%s", e)
            return
        except Exception:
            self._scenes_note = "The scene engine could not start"
            log.exception("Scene engine failed to load")
            return
        self._scenes_note = None
        tag_ids = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM tags")}
        labels = [tag_ids[name] for name, _ in scenes.all_labels()]

        for start in range(0, len(pending), scenes.BATCH):
            if not self._gate():
                return
            batch = pending[start : start + scenes.BATCH]
            self._run_scene_batch(engine, batch, labels)

    def _run_scene_batch(self, engine, batch, labels) -> None:
        images, ok, failed = [], [], []
        for row in batch:
            try:
                # The 1024 px thumbnail is upright already and far quicker to read than the original.
                with Image.open(self.config.thumb_path(row["content_hash"], 1024)) as img:
                    images.append(img.convert("RGB"))
                ok.append(row["photo_id"])
            except Exception as e:
                failed.append((row["photo_id"], f"{type(e).__name__}: {e}"))
        embeddings = probs = None
        if images:
            try:
                embeddings, probs = engine.embed_images(images)
            except Exception as e:
                failed += [(photo_id, f"{type(e).__name__}: {e}") for photo_id in ok]
                ok = []
        with self.db.write() as conn:
            for i, photo_id in enumerate(ok):
                conn.execute(
                    "INSERT OR REPLACE INTO photo_embeddings(photo_id, embedding, model_version) VALUES(?, ?, ?)",
                    (photo_id, embeddings[i].astype("float32").tobytes(), engine.version),
                )
                conn.execute("DELETE FROM photo_tags WHERE photo_id = ?", (photo_id,))
                # Scores a little under the floor are kept so the floor can be tuned later.
                conn.executemany(
                    "INSERT INTO photo_tags(photo_id, tag_id, score) VALUES(?, ?, ?)",
                    [
                        (photo_id, tag_id, float(score))
                        for tag_id, score in zip(labels, probs[i])
                        if score >= scenes.TAG_FLOOR / 2
                    ],
                )
            conn.executemany(
                "UPDATE jobs SET status='done', error=NULL, updated_at=? WHERE photo_id=? AND stage=?",
                [(_now(), photo_id, STAGE_SCENES) for photo_id in ok],
            )
            conn.executemany(
                "UPDATE jobs SET status='error', error=?, updated_at=? WHERE photo_id=? AND stage=?",
                [(error, _now(), photo_id, STAGE_SCENES) for photo_id, error in failed],
            )

"""Private folder browsing: look through a folder without Pixora keeping anything.

A private session never touches the database, the thumbnail cache, the face or
scene stages, or the log. Small thumbnails are held in memory only; the large
view is rendered from the original each time it is asked for. Closing the
session, opening another one, or stopping the service drops everything.
"""

from __future__ import annotations

import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import media

# Enough for a large folder of small thumbnails (about 15 KB each) to stay modest in memory.
MAX_PHOTOS = 20000
_SKIP_DIRS = frozenset({"$recycle.bin", "system volume information"})


@dataclass
class PrivatePhoto:
    id: int
    path: str
    name: str
    size: int
    info: media.ImageInfo
    thumb: bytes  # the 256 px WebP


@dataclass
class PrivateSession:
    token: str
    root: str
    workers: int
    total: int = 0
    done: int = 0
    errors: int = 0
    truncated: bool = False
    state: str = "discovering"  # discovering | processing | ready | closed
    photos: dict[int, PrivatePhoto] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _closed: threading.Event = field(default_factory=threading.Event)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            self.state = "closed"
            self.photos.clear()

    def snapshot(self) -> list[PrivatePhoto]:
        with self._lock:
            return sorted(self.photos.values(), key=lambda p: (p.info.taken_at, p.id), reverse=True)

    def get(self, photo_id: int) -> PrivatePhoto | None:
        with self._lock:
            return self.photos.get(photo_id)

    def _discover(self) -> list[tuple[str, int, int]]:
        found: list[tuple[str, int, int]] = []
        stack = [self.root]
        while stack and not self.closed:
            try:
                entries = list(os.scandir(stack.pop()))
            except OSError:
                self.errors += 1
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.lower() not in _SKIP_DIRS and not e.name.startswith("."):
                            stack.append(e.path)
                    elif e.is_file() and media.is_supported(e.name):
                        st = e.stat()
                        found.append((e.path, st.st_size, st.st_mtime_ns))
                except OSError:
                    self.errors += 1
                if len(found) >= MAX_PHOTOS:
                    self.truncated = True
                    return found
        return found

    def _process(self, photo_id: int, path: str, size: int, mtime_ns: int) -> None:
        if self.closed:
            return
        try:
            if size <= media.IN_MEMORY_LIMIT:
                with open(path, "rb") as f:
                    source: bytes | str = f.read()
            else:
                source = path
            info, _large, small = media.render_image(source, mtime_ns, path)
            photo = PrivatePhoto(
                id=photo_id,
                path=path,
                name=os.path.basename(path),
                size=size,
                info=info,
                thumb=media.webp_bytes(small),
            )
            with self._lock:
                if not self.closed:
                    self.photos[photo_id] = photo
        except Exception:
            # Deliberately not logged: a log line would record the file's path.
            with self._lock:
                self.errors += 1
        finally:
            with self._lock:
                self.done += 1

    def run(self) -> None:
        files = self._discover()
        with self._lock:
            if self.closed:
                return
            self.total = len(files)
            self.state = "processing"
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="private") as pool:
            for i, (path, size, mtime) in enumerate(files, start=1):
                if self.closed:
                    break
                pool.submit(self._process, i, path, size, mtime)
        with self._lock:
            if not self.closed:
                self.state = "ready"


def render_large(photo: PrivatePhoto) -> bytes:
    """The 1024 px view, rendered from the original on demand and never kept."""
    _info, large, _small = media.render_image(photo.path, 0, photo.path)
    return media.webp_bytes(large)


class PrivateBrowser:
    """Holds at most one private session at a time."""

    def __init__(self, workers: int):
        self._workers = workers
        self._lock = threading.Lock()
        self._session: PrivateSession | None = None

    def open(self, root: str) -> PrivateSession:
        session = PrivateSession(token=secrets.token_urlsafe(16), root=root, workers=self._workers)
        with self._lock:
            if self._session:
                self._session.close()
            self._session = session
        threading.Thread(target=session.run, name="private-scan", daemon=True).start()
        return session

    def get(self, token: str) -> PrivateSession | None:
        with self._lock:
            s = self._session
        if s and not s.closed and secrets.compare_digest(s.token, token):
            return s
        return None

    def close(self, token: str | None = None) -> None:
        with self._lock:
            s = self._session
            if s and (token is None or secrets.compare_digest(s.token, token)):
                s.close()
                self._session = None

"""The typed local API. The UI reaches originals and thumbnails only through here."""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from pydantic import BaseModel

from . import __version__
from .config import THUMB_SIZES, Config
from .db import Database
from .faces import ModelsMissing
from .galaxy import POINT_SIZE, people_layout
from .scanner import ScanManager, repair_primaries
from .scenes import TAG_FLOOR
from .search import SearchIndex

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "tauri.localhost", "testserver"}
_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "tauri://localhost",
    "http://tauri.localhost",
]
_HASH_RE = re.compile(r"^[0-9a-f]{32}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_GALAXY_LIMIT = 20000
_VISIBLE = "ready = 1 AND primary_file_id IS NOT NULL AND duplicate_of_photo_id IS NULL"


class Folder(BaseModel):
    id: int
    path: str
    added_at: str
    file_count: int
    error: str | None = None


class AddFolder(BaseModel):
    path: str


class StageProgress(BaseModel):
    done: int
    total: int


class LibraryCounts(BaseModel):
    photos: int
    files: int
    duplicate_files: int


class FolderError(BaseModel):
    folder_id: int
    error: str


class ScanStatus(BaseModel):
    state: str  # idle | discovering | processing | faces | paused
    faces_note: str | None = None  # why the face stage is not running, if it is not
    scenes_note: str | None = None
    paused: bool
    discovered: int
    stages: dict[str, StageProgress]
    errors: int
    library: LibraryCounts
    folder_errors: list[FolderError]


class TimelineBucket(BaseModel):
    month: str  # YYYY-MM
    count: int


class Timeline(BaseModel):
    total: int
    buckets: list[TimelineBucket]


class CameraFacet(BaseModel):
    camera: str
    count: int


class PersonFacet(BaseModel):
    id: int
    name: str
    count: int


class SceneFacet(BaseModel):
    name: str
    category: str
    count: int


class OccasionFacet(BaseModel):
    name: str
    count: int


class Facets(BaseModel):
    cameras: list[CameraFacet]
    people: list[PersonFacet]
    scenes: list[SceneFacet]
    occasions: list[OccasionFacet]
    search_available: bool
    first_date: str | None  # YYYY-MM-DD
    last_date: str | None


class PhotoSummary(BaseModel):
    id: int
    hash: str
    taken_at: str
    width: int
    height: int


class PhotoFile(BaseModel):
    path: str
    size: int
    missing: bool


class PhotoFace(BaseModel):
    id: int
    # box as fractions of the photo's width and height
    x1: float
    y1: float
    x2: float
    y2: float
    person_id: int | None
    name: str | None
    source: str | None  # 'user' or 'auto'
    suggested_person_id: int | None
    suggested_name: str | None


class PhotoTag(BaseModel):
    name: str
    category: str
    score: float


class PhotoEvent(BaseModel):
    id: int
    title: str
    type: str | None
    start_at: str
    end_at: str
    photo_count: int


class PhotoDetail(PhotoSummary):
    tags: list[PhotoTag]
    event: PhotoEvent | None
    taken_at_source: str
    lat: float | None
    lon: float | None
    camera: str | None
    files: list[PhotoFile]
    faces: list[PhotoFace]


class Person(BaseModel):
    id: int
    name: str
    hidden: bool
    cover_face_id: int | None
    face_count: int
    photo_count: int


class PeopleOverview(BaseModel):
    people: list[Person]
    unnamed_groups: int  # groups of unknown faces waiting for a name


class Question(BaseModel):
    type: str  # 'name' ("Who is this?"), 'confirm' ("Is this also NAME?"), or 'done'
    face_ids: list[int] = []
    sample_face_ids: list[int] = []
    photo_count: int = 0
    person_id: int | None = None
    name: str | None = None


class NameFaces(BaseModel):
    face_ids: list[int]
    name: str


class ConfirmFaces(BaseModel):
    person_id: int
    face_ids: list[int]
    answer: bool


class FaceIds(BaseModel):
    face_ids: list[int]


class PersonUpdate(BaseModel):
    name: str | None = None
    hidden: bool | None = None


class PersonRef(BaseModel):
    person_id: int


class AssignFace(BaseModel):
    name: str


class MapPhoto(BaseModel):
    id: int
    lat: float
    lon: float
    hash: str


class MapCluster(BaseModel):
    id: int
    lat: float
    lon: float
    count: int
    expansion_zoom: int


class MapData(BaseModel):
    photos_with_location: int
    photos_without_location: int
    points: list[MapPhoto | MapCluster]


class NoLocationPhotos(BaseModel):
    photos: list[PhotoSummary]


def _filter_sql(
    camera: str | None, date_from: str | None, date_to: str | None, person_id: int | None = None
) -> tuple[str, list]:
    """The shared filter state, as SQL. Later phases add place and scene."""
    clauses, params = [_VISIBLE], []
    if person_id is not None:
        clauses.append("EXISTS (SELECT 1 FROM faces WHERE faces.photo_id = photos.id AND faces.person_id = ?)")
        params.append(person_id)
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    for value, op in ((date_from, ">="), (date_to, "<")):
        if value:
            if not _DATE_RE.match(value):
                raise HTTPException(400, "Dates must be YYYY-MM-DD")
            clauses.append(f"taken_at {op} ?")
            # "~" sorts after every time of day, making the end date inclusive.
            params.append(value if op == ">=" else value + "~")
    return " AND ".join(clauses), params




class GalaxyFace(BaseModel):
    id: int
    photo_id: int
    hash: str
    taken_at: str
    width: int
    height: int
    person_id: int | None
    scene: str | None
    # Position in People mode.
    x: float
    y: float
    z: float


class GalaxyPerson(BaseModel):
    id: int
    name: str
    face_count: int
    cover_face_id: int | None
    x: float
    y: float
    z: float
    radius: float


class GalaxyLink(BaseModel):
    """Two people and the number of photos that show them together."""

    a: int
    b: int
    photos: int


class GalaxyData(BaseModel):
    faces: list[GalaxyFace]
    people: list[GalaxyPerson]
    links: list[GalaxyLink]
    point_size: float


def create_app(
    config: Config | None = None, autostart: bool = True, engine_factory=None, scene_factory=None
) -> FastAPI:
    config = config or Config.load()
    db = Database(config.db_path)
    scanner = ScanManager(db, config, engine_factory, scene_factory)
    people = scanner.people
    search = SearchIndex(db, scanner.scene_engine)

    def photo_filter(
        camera: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: int | None = None,
        scene: str | None = None,
        occasion: str | None = None,
        q: str | None = None,
    ) -> tuple[str, list]:
        """Every view's shared filter state, as a SQL condition."""
        where, params = _filter_sql(camera, date_from, date_to, person_id)
        if scene:
            where += (
                " AND EXISTS (SELECT 1 FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
                "WHERE pt.photo_id = photos.id AND t.name = ? AND pt.score >= ?)"
            )
            params += [scene, TAG_FLOOR]
        if occasion:
            where += " AND event_id IN (SELECT id FROM events WHERE type = ?)"
            params.append(occasion)
        if q and q.strip():
            # A person's name finds that person; anything else is a description.
            named = db.conn().execute(
                "SELECT id FROM people WHERE name = ? COLLATE NOCASE", (" ".join(q.split()),)
            ).fetchone()
            if named:
                where += " AND EXISTS (SELECT 1 FROM faces WHERE faces.photo_id = photos.id AND faces.person_id = ?)"
                params.append(named["id"])
            else:
                try:
                    ids = search.photo_ids(q)
                except ModelsMissing:
                    raise HTTPException(503, "Search needs the scene model, which is not installed")
                where += " AND photos.id IN (SELECT value FROM json_each(?))"
                params.append(json.dumps(ids))
        return where, params

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        has_folders = db.conn().execute("SELECT 1 FROM folders LIMIT 1").fetchone()
        if autostart and has_folders and not scanner.paused:
            scanner.start(walk=True)
        yield
        scanner.shutdown()

    app = FastAPI(title="Pixora local service", version=__version__, lifespan=lifespan)
    app.state.db = db
    app.state.scanner = scanner
    app.add_middleware(
        CORSMiddleware, allow_origins=_ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Reject requests that reach the loopback port under another hostname.
        host = request.headers.get("host", "").rsplit(":", 1)[0]
        if host not in _ALLOWED_HOSTS:
            return JSONResponse({"detail": "Forbidden host"}, status_code=403)
        return await call_next(request)

    def folder_rows() -> list[Folder]:
        errors = {e["folder_id"]: e["error"] for e in scanner.status()["folder_errors"]}
        rows = db.conn().execute(
            "SELECT f.id, f.path, f.added_at, "
            "(SELECT count(*) FROM photo_files pf WHERE pf.folder_id=f.id AND pf.missing=0) AS n "
            "FROM folders f ORDER BY f.path"
        )
        return [
            Folder(
                id=r["id"],
                path=r["path"],
                added_at=r["added_at"],
                file_count=r["n"],
                error=errors.get(r["id"]),
            )
            for r in rows
        ]

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    # ---- folders -------------------------------------------------------------

    @app.get("/api/folders")
    def list_folders() -> list[Folder]:
        return folder_rows()

    @app.post("/api/folders")
    def add_folder(body: AddFolder) -> list[Folder]:
        raw = body.path.strip().strip('"')
        if not raw or not os.path.isabs(raw):
            raise HTTPException(400, "Enter the full path of a folder")
        path = os.path.abspath(raw)
        if not os.path.isdir(path):
            raise HTTPException(400, "That folder does not exist")

        def inside(child: str, parent: str) -> bool:
            c, p = os.path.normcase(child), os.path.normcase(parent)
            return c == p or c.startswith(p.rstrip("\\/") + os.sep)

        with db.write() as conn:
            existing = conn.execute("SELECT id, path FROM folders").fetchall()
            if any(inside(path, f["path"]) for f in existing):
                raise HTTPException(409, "That folder is already in the library")
            folder_id = conn.execute(
                "INSERT INTO folders(path, added_at) VALUES(?, ?)",
                (path, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ).lastrowid
            # Folders inside the new one are absorbed into it.
            for f in existing:
                if inside(f["path"], path):
                    conn.execute(
                        "UPDATE photo_files SET folder_id=? WHERE folder_id=?", (folder_id, f["id"])
                    )
                    conn.execute("DELETE FROM folders WHERE id=?", (f["id"],))
        scanner.start(walk=True)
        return folder_rows()

    @app.delete("/api/folders/{folder_id}")
    def remove_folder(folder_id: int) -> list[Folder]:
        """Forget a folder. Nothing on disk is touched."""
        with db.write() as conn:
            if not conn.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone():
                raise HTTPException(404, "Folder not found")
            photo_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT photo_id FROM photo_files WHERE folder_id=?", (folder_id,)
                )
            ]
            conn.execute("DELETE FROM photo_files WHERE folder_id=?", (folder_id,))
            conn.execute("DELETE FROM folders WHERE id=?", (folder_id,))
            repair_primaries(conn, photo_ids)
        return folder_rows()

    # ---- scan ----------------------------------------------------------------

    @app.get("/api/scan/status")
    def scan_status() -> ScanStatus:
        return ScanStatus(**scanner.status())

    @app.post("/api/scan/start")
    def scan_start() -> ScanStatus:
        scanner.start(walk=True)
        return ScanStatus(**scanner.status())

    @app.post("/api/scan/pause")
    def scan_pause() -> ScanStatus:
        scanner.pause()
        return ScanStatus(**scanner.status())

    @app.post("/api/scan/resume")
    def scan_resume() -> ScanStatus:
        scanner.resume()
        return ScanStatus(**scanner.status())

    # ---- photos --------------------------------------------------------------

    @app.get("/api/facets")
    def facets() -> Facets:
        conn = db.conn()
        cameras = conn.execute(
            f"SELECT camera, count(*) AS n FROM photos WHERE {_VISIBLE} AND camera IS NOT NULL "
            f"GROUP BY camera ORDER BY n DESC, camera"
        ).fetchall()
        span = conn.execute(
            f"SELECT substr(min(taken_at), 1, 10), substr(max(taken_at), 1, 10) "
            f"FROM photos WHERE {_VISIBLE}"
        ).fetchone()
        named = conn.execute(
            f"SELECT pe.id, pe.name, count(DISTINCT f.photo_id) AS n FROM people pe "
            f"JOIN faces f ON f.person_id = pe.id JOIN photos ON photos.id = f.photo_id "
            f"WHERE pe.hidden = 0 AND {_VISIBLE} GROUP BY pe.id ORDER BY n DESC, pe.name"
        ).fetchall()
        tagged = conn.execute(
            f"SELECT t.name, t.category, count(*) AS n FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            f"JOIN photos ON photos.id = pt.photo_id WHERE pt.score >= ? AND t.category != 'occasions' "
            f"AND {_VISIBLE} GROUP BY t.id ORDER BY n DESC, t.name",
            (TAG_FLOOR,),
        ).fetchall()
        occasions = conn.execute(
            f"SELECT e.type, count(*) AS n FROM events e JOIN photos ON photos.event_id = e.id "
            f"WHERE e.type IS NOT NULL AND {_VISIBLE} GROUP BY e.type ORDER BY n DESC, e.type"
        ).fetchall()
        return Facets(
            cameras=[CameraFacet(camera=r["camera"], count=r["n"]) for r in cameras],
            people=[PersonFacet(id=r["id"], name=r["name"], count=r["n"]) for r in named],
            scenes=[SceneFacet(name=r["name"], category=r["category"], count=r["n"]) for r in tagged],
            occasions=[OccasionFacet(name=r["type"], count=r["n"]) for r in occasions],
            search_available=bool(conn.execute("SELECT 1 FROM photo_embeddings LIMIT 1").fetchone()),
            first_date=span[0],
            last_date=span[1],
        )

    @app.get("/api/timeline")
    def timeline(filtered: tuple[str, list] = Depends(photo_filter)) -> Timeline:
        where, params = filtered
        rows = db.conn().execute(
            f"SELECT substr(taken_at, 1, 7) AS month, count(*) AS n FROM photos "
            f"WHERE {where} GROUP BY month ORDER BY month DESC",
            params,
        ).fetchall()
        buckets = [TimelineBucket(month=r["month"], count=r["n"]) for r in rows]
        return Timeline(total=sum(b.count for b in buckets), buckets=buckets)

    @app.get("/api/photos")
    def photos(month: str, filtered: tuple[str, list] = Depends(photo_filter)) -> list[PhotoSummary]:
        if not _MONTH_RE.match(month):
            raise HTTPException(400, "month must be YYYY-MM")
        where, params = filtered
        rows = db.conn().execute(
            f"SELECT id, content_hash, taken_at, width, height FROM photos "
            f"WHERE {where} AND taken_at >= ? AND taken_at < ? "
            f"ORDER BY taken_at DESC, id DESC",
            (*params, month, month + "~"),
        )
        return [
            PhotoSummary(
                id=r["id"],
                hash=r["content_hash"],
                taken_at=r["taken_at"],
                width=r["width"],
                height=r["height"],
            )
            for r in rows
        ]

    @app.get("/api/photos/{photo_id}")
    def photo_detail(photo_id: int) -> PhotoDetail:
        conn = db.conn()
        r = conn.execute("SELECT * FROM photos WHERE id=? AND ready=1", (photo_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "Photo not found")
        files = conn.execute(
            "SELECT path, size, missing FROM photo_files WHERE photo_id=? "
            "ORDER BY (id = ?) DESC, path",
            (photo_id, r["primary_file_id"]),
        )
        event = conn.execute(
            "SELECT e.*, (SELECT count(*) FROM photos WHERE event_id = e.id) AS n FROM events e WHERE e.id = ?",
            (r["event_id"],),
        ).fetchone()
        tags = conn.execute(
            "SELECT t.name, t.category, pt.score FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            "WHERE pt.photo_id = ? AND pt.score >= ? AND t.category != 'occasions' ORDER BY pt.score DESC",
            (photo_id, TAG_FLOOR),
        )
        return PhotoDetail(
            tags=[PhotoTag(name=t["name"], category=t["category"], score=t["score"]) for t in tags],
            event=PhotoEvent(
                id=event["id"], title=event["title"], type=event["type"],
                start_at=event["start_at"], end_at=event["end_at"], photo_count=event["n"],
            ) if event else None,
            id=r["id"],
            hash=r["content_hash"],
            taken_at=r["taken_at"],
            width=r["width"],
            height=r["height"],
            taken_at_source=r["taken_at_source"],
            lat=r["lat"],
            lon=r["lon"],
            camera=r["camera"],
            files=[PhotoFile(path=f["path"], size=f["size"], missing=bool(f["missing"])) for f in files],
            faces=[
                PhotoFace(
                    id=f["id"], x1=f["x1"], y1=f["y1"], x2=f["x2"], y2=f["y2"],
                    person_id=f["person_id"], name=f["name"], source=f["assignment_source"],
                    suggested_person_id=f["suggested_person_id"], suggested_name=f["suggested_name"],
                )
                for f in conn.execute(
                    "SELECT f.*, p.name, s.name AS suggested_name FROM faces f "
                    "LEFT JOIN people p ON p.id = f.person_id "
                    "LEFT JOIN people s ON s.id = f.suggested_person_id "
                    "WHERE f.photo_id = ? ORDER BY f.x1",
                    (photo_id,),
                )
            ],
        )

    @app.get("/api/photos/{photo_id}/original")
    def photo_original(photo_id: int) -> FileResponse:
        r = db.conn().execute(
            "SELECT f.path FROM photos p JOIN photo_files f ON f.id = p.primary_file_id "
            "WHERE p.id=?",
            (photo_id,),
        ).fetchone()
        if r is None or not os.path.isfile(r["path"]):
            raise HTTPException(404, "Original not available")
        return FileResponse(r["path"], headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/thumbs/{content_hash}/{size}")
    def thumbnail(content_hash: str, size: int) -> FileResponse:
        if not _HASH_RE.match(content_hash) or size not in THUMB_SIZES:
            raise HTTPException(404, "Thumbnail not found")
        path = config.thumb_path(content_hash, size)
        if not path.is_file():
            raise HTTPException(404, "Thumbnail not found")
        # Addressed by content hash, so the bytes behind a URL never change.
        return FileResponse(
            path,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    # ---- people ----------------------------------------------------------------

    def known_faces(face_ids: list[int]) -> list[int]:
        if not face_ids:
            raise HTTPException(400, "No faces given")
        marks = ",".join("?" * len(face_ids))
        found = db.conn().execute(f"SELECT count(*) FROM faces WHERE id IN ({marks})", face_ids).fetchone()[0]
        if found != len(set(face_ids)):
            raise HTTPException(404, "Face not found")
        return face_ids

    def known_person(person_id: int) -> int:
        if not db.conn().execute("SELECT 1 FROM people WHERE id=?", (person_id,)).fetchone():
            raise HTTPException(404, "Person not found")
        return person_id

    @app.get("/api/people")
    def list_people() -> PeopleOverview:
        rows = db.conn().execute(
            "SELECT pe.id, pe.name, pe.hidden, pe.cover_face_id, count(f.id) AS faces, "
            "count(DISTINCT f.photo_id) AS photos FROM people pe "
            "LEFT JOIN faces f ON f.person_id = pe.id GROUP BY pe.id ORDER BY photos DESC, pe.name"
        )
        return PeopleOverview(
            people=[
                Person(
                    id=r["id"], name=r["name"], hidden=bool(r["hidden"]), cover_face_id=r["cover_face_id"],
                    face_count=r["faces"], photo_count=r["photos"],
                )
                for r in rows
            ],
            unnamed_groups=people.pending_groups(),
        )

    @app.get("/api/people/question")
    def people_question() -> Question:
        return Question(**people.next_question())

    @app.post("/api/people/name")
    def people_name(body: NameFaces) -> PersonRef:
        try:
            return PersonRef(person_id=people.name_faces(known_faces(body.face_ids), body.name))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/people/confirm")
    def people_confirm(body: ConfirmFaces) -> Question:
        people.confirm(known_person(body.person_id), known_faces(body.face_ids), body.answer)
        return Question(**people.next_question())

    @app.post("/api/people/dismiss")
    def people_dismiss(body: FaceIds) -> Question:
        people.dismiss(known_faces(body.face_ids))
        return Question(**people.next_question())

    @app.patch("/api/people/{person_id}")
    def update_person(person_id: int, body: PersonUpdate) -> PersonRef:
        known_person(person_id)
        if body.hidden is not None:
            people.set_hidden(person_id, body.hidden)
        if body.name is not None:
            try:
                person_id = people.rename(person_id, body.name)
            except ValueError as e:
                raise HTTPException(400, str(e))
        return PersonRef(person_id=person_id)

    @app.post("/api/faces/{face_id}/assign")
    def assign_face(face_id: int, body: AssignFace) -> PersonRef:
        try:
            return PersonRef(person_id=people.name_faces(known_faces([face_id]), body.name))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/faces/{face_id}/reject")
    def reject_face(face_id: int) -> dict:
        people.reject(known_faces([face_id])[0])
        return {"ok": True}

    @app.get("/api/faces/{face_id}/crop")
    def face_crop(face_id: int) -> FileResponse:
        path = config.face_crop_path(face_id)
        if not path.is_file():
            raise HTTPException(404, "Face not found")
        return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "private, max-age=86400"})


    @app.get("/api/map/data")
    def map_data(
        q: str | None = None,
        scene: str | None = None,
        occasion: str | None = None,
        person_id: int | None = None,
        camera: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        zoom: int = 0,
        min_lat: float = -90,
        max_lat: float = 90,
        min_lon: float = -180,
        max_lon: float = 180,
    ) -> MapData:
        """Get photos with location data, optionally clustered, within bounds."""
        conn = db.conn()

        # Get all photos with location matching filters
        where, params = _filter_sql(camera, date_from, date_to, person_id)
        if q:
            try:
                ids = search.photo_ids(q)
                where += f" AND photos.id IN ({','.join('?' * len(ids))})"
                params.extend(ids)
            except Exception:
                return MapData(photos_with_location=0, photos_without_location=0, points=[])
        if scene:
            where += """ AND EXISTS (SELECT 1 FROM photo_tags
              JOIN tags ON photo_tags.tag_id = tags.id
              WHERE photo_tags.photo_id = photos.id AND tags.name = ?)"""
            params.append(scene)
        if occasion:
            where += """ AND EXISTS (SELECT 1 FROM events
              WHERE events.id = photos.event_id AND events.type = ?)"""
            params.append(occasion)

        where += " AND lat IS NOT NULL AND lon IS NOT NULL"
        where += f" AND lat BETWEEN {min_lat} AND {max_lat}"
        where += f" AND lon BETWEEN {min_lon} AND {max_lon}"

        rows = conn.execute(
            f"SELECT id, hash, lat, lon FROM photos WHERE {where}",
            params,
        ).fetchall()

        points = [MapPhoto(id=r[0], hash=r[1], lat=r[2], lon=r[3]) for r in rows]

        # Count photos without location
        where_no_loc, params_no_loc = _filter_sql(camera, date_from, date_to, person_id)
        if q:
            try:
                ids = search.photo_ids(q)
                where_no_loc += f" AND photos.id IN ({','.join('?' * len(ids))})"
                params_no_loc.extend(ids)
            except Exception:
                where_no_loc += " AND 1=0"
        if scene:
            where_no_loc += """ AND EXISTS (SELECT 1 FROM photo_tags
              JOIN tags ON photo_tags.tag_id = tags.id
              WHERE photo_tags.photo_id = photos.id AND tags.name = ?)"""
            params_no_loc.append(scene)
        if occasion:
            where_no_loc += """ AND EXISTS (SELECT 1 FROM events
              WHERE events.id = photos.event_id AND events.type = ?)"""
            params_no_loc.append(occasion)

        where_no_loc += " AND (lat IS NULL OR lon IS NULL)"

        no_loc_count = conn.execute(
            f"SELECT COUNT(*) FROM photos WHERE {where_no_loc}",
            params_no_loc,
        ).fetchone()[0]

        return MapData(
            photos_with_location=len(points),
            photos_without_location=no_loc_count,
            points=points,
        )

    @app.get("/api/map/no-location")
    def no_location_photos(
        q: str | None = None,
        scene: str | None = None,
        occasion: str | None = None,
        person_id: int | None = None,
        camera: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> NoLocationPhotos:
        """Get photos without location data."""
        conn = db.conn()

        where, params = _filter_sql(camera, date_from, date_to, person_id)
        if q:
            try:
                ids = search.photo_ids(q)
                where += f" AND photos.id IN ({','.join('?' * len(ids))})"
                params.extend(ids)
            except Exception:
                return NoLocationPhotos(photos=[])
        if scene:
            where += """ AND EXISTS (SELECT 1 FROM photo_tags
              JOIN tags ON photo_tags.tag_id = tags.id
              WHERE photo_tags.photo_id = photos.id AND tags.name = ?)"""
            params.append(scene)
        if occasion:
            where += """ AND EXISTS (SELECT 1 FROM events
              WHERE events.id = photos.event_id AND events.type = ?)"""
            params.append(occasion)

        where += " AND (lat IS NULL OR lon IS NULL)"

        rows = conn.execute(
            f"SELECT id, hash, taken_at, width, height FROM photos WHERE {where} ORDER BY taken_at DESC LIMIT 100",
            params,
        ).fetchall()

        photos = [
            PhotoSummary(id=r[0], hash=r[1], taken_at=r[2], width=r[3], height=r[4])
            for r in rows
        ]

        return NoLocationPhotos(photos=photos)



    @app.get("/api/galaxy/data")
    def galaxy_data(filtered: tuple[str, list] = Depends(photo_filter)) -> GalaxyData:
        """Every face in the matching photos, placed for the galaxy."""
        where, params = filtered
        conn = db.conn()
        rows = conn.execute(
            f"SELECT f.id, f.photo_id, f.person_id, f.cluster_id, f.embedding, pe.name, pe.cover_face_id, "
            f"photos.content_hash, photos.taken_at, photos.width, photos.height "
            f"FROM faces f JOIN photos ON photos.id = f.photo_id LEFT JOIN people pe ON pe.id = f.person_id "
            f"WHERE {where} AND f.dismissed = 0 AND coalesce(pe.hidden, 0) = 0 "
            f"ORDER BY f.id LIMIT {_GALAXY_LIMIT}",
            params,
        ).fetchall()
        if not rows:
            return GalaxyData(faces=[], people=[], links=[], point_size=POINT_SIZE)

        embeddings = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        positions, centres = people_layout(
            embeddings, [r["person_id"] for r in rows], [r["cluster_id"] for r in rows]
        )

        # The strongest scene tag of each photo.
        scenes: dict[int, str] = {}
        for t in conn.execute(
            "SELECT pt.photo_id, t.name FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            "WHERE pt.score >= ? AND t.category != 'occasions' ORDER BY pt.score DESC",
            (TAG_FLOOR,),
        ):
            scenes.setdefault(t["photo_id"], t["name"])

        info: dict[int, dict] = {}
        together: dict[int, set[int]] = {}
        for r in rows:
            if r["person_id"] is None:
                continue
            person = info.setdefault(
                r["person_id"], {"name": r["name"], "cover": r["cover_face_id"], "faces": 0}
            )
            person["faces"] += 1
            together.setdefault(r["photo_id"], set()).add(r["person_id"])
        pairs: dict[tuple[int, int], int] = {}
        for group in together.values():
            members = sorted(group)
            for i, first in enumerate(members):
                for second in members[i + 1 :]:
                    pairs[first, second] = pairs.get((first, second), 0) + 1

        return GalaxyData(
            faces=[
                GalaxyFace(
                    id=r["id"],
                    photo_id=r["photo_id"],
                    hash=r["content_hash"],
                    taken_at=r["taken_at"],
                    width=r["width"],
                    height=r["height"],
                    person_id=r["person_id"],
                    scene=scenes.get(r["photo_id"]),
                    x=round(float(x), 3),
                    y=round(float(y), 3),
                    z=round(float(z), 3),
                )
                for r, (x, y, z) in zip(rows, positions)
            ],
            people=[
                GalaxyPerson(
                    id=pid,
                    name=person["name"],
                    face_count=person["faces"],
                    cover_face_id=person["cover"],
                    x=round(float(centres[pid][0][0]), 3),
                    y=round(float(centres[pid][0][1]), 3),
                    z=round(float(centres[pid][0][2]), 3),
                    radius=round(centres[pid][1], 3),
                )
                for pid, person in sorted(info.items(), key=lambda kv: (-kv[1]["faces"], kv[0]))
            ],
            links=[GalaxyLink(a=a, b=b, photos=n) for (a, b), n in sorted(pairs.items())],
            point_size=POINT_SIZE,
        )

    return app

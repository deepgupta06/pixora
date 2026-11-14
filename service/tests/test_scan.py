import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixora_service.api import create_app
from pixora_service.config import Config


def make_image(path: Path, color, size=(640, 480), exif_date: str | None = None, orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    # A gradient stripe so different images do not hash alike.
    for x in range(0, size[0], 7):
        img.putpixel((x, x % size[1]), (255 - color[0], x % 255, 0))
    exif = Image.Exif()
    if exif_date:
        exif.get_ifd(0x8769)[36867] = exif_date
        exif[271] = "Acme"
        exif[272] = "Acme Cam 1"
    if orientation:
        exif[274] = orientation
    img.save(path, exif=exif)


@pytest.fixture
def env(tmp_path):
    config = Config.load(data_dir=tmp_path / "data", workers=2)
    lib = tmp_path / "library"
    lib.mkdir()
    apps = []

    def open_app():
        client = TestClient(create_app(config))
        client.__enter__()
        apps.append(client)
        return client

    yield lib, config, open_app
    for client in apps:
        client.__exit__(None, None, None)


def scan(client, wait=True):
    client.post("/api/scan/start")
    if wait:
        assert client.app.state.scanner.wait_idle(30)


def all_photos(client):
    out = []
    for b in client.get("/api/timeline").json()["buckets"]:
        out += client.get("/api/photos", params={"month": b["month"]}).json()
    return out


def test_scan_reads_metadata_and_writes_thumbnails(env):
    lib, config, open_app = env
    make_image(lib / "a.jpg", (200, 30, 30), exif_date="2021:06:05 10:11:12")
    make_image(lib / "sub" / "b.png", (30, 200, 30))
    make_image(lib / "rot.jpg", (30, 30, 200), size=(800, 400), orientation=6)
    (lib / "notes.txt").write_text("not a photo")
    client = open_app()
    assert client.post("/api/folders", json={"path": str(lib)}).status_code == 200
    assert client.app.state.scanner.wait_idle(30)

    photos = all_photos(client)
    assert len(photos) == 3
    by_date = {p["taken_at"]: p for p in photos}
    a = by_date["2021-06-05T10:11:12"]
    detail = client.get(f"/api/photos/{a['id']}").json()
    assert detail["camera"] == "Acme Cam 1"
    assert detail["taken_at_source"] == "exif"
    assert (detail["width"], detail["height"]) == (640, 480)

    rotated = next(p for p in photos if p["width"] == 400)
    assert rotated["height"] == 800  # EXIF orientation applied

    for size in (256, 1024):
        r = client.get(f"/api/thumbs/{a['hash']}/{size}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    with Image.open(config.thumb_path(rotated["hash"], 256)) as thumb:
        assert thumb.height > thumb.width
    assert client.get(f"/api/photos/{a['id']}/original").status_code == 200


def test_exact_duplicates_show_once_and_keep_all_paths(env):
    lib, _, open_app = env
    make_image(lib / "one" / "a.jpg", (10, 120, 200))
    shutil.copy2(lib / "one" / "a.jpg", lib / "one" / "copy.jpg")
    (lib / "two").mkdir()
    shutil.copy2(lib / "one" / "a.jpg", lib / "two" / "elsewhere.jpg")
    make_image(lib / "two" / "other.jpg", (200, 120, 10))
    client = open_app()
    client.post("/api/folders", json={"path": str(lib)})
    assert client.app.state.scanner.wait_idle(30)

    photos = all_photos(client)
    assert len(photos) == 2
    status = client.get("/api/scan/status").json()
    assert status["library"] == {"photos": 2, "files": 4, "duplicate_files": 2}
    paths = [
        len(client.get(f"/api/photos/{p['id']}").json()["files"]) for p in photos
    ]
    assert sorted(paths) == [1, 3]


def test_rescan_is_incremental_and_tracks_changes(env, monkeypatch):
    lib, _, open_app = env
    make_image(lib / "a.jpg", (1, 2, 3))
    make_image(lib / "b.jpg", (4, 5, 6))
    shutil.copy2(lib / "b.jpg", lib / "b_copy.jpg")
    client = open_app()
    client.post("/api/folders", json={"path": str(lib)})
    assert client.app.state.scanner.wait_idle(30)
    assert len(all_photos(client)) == 2

    from pixora_service import media

    hashed = []
    real = media.read_and_hash
    monkeypatch.setattr(
        media, "read_and_hash", lambda path, size: (hashed.append(path), real(path, size))[1]
    )

    # Nothing changed: nothing is re-read.
    scan(client)
    assert hashed == []

    # One new file, one changed file, one deleted primary with a surviving copy.
    make_image(lib / "c.jpg", (7, 8, 9))
    make_image(lib / "a.jpg", (99, 98, 97), size=(320, 200))
    os.remove(lib / "b.jpg")
    scan(client)
    assert sorted(os.path.basename(p) for p in hashed) == ["a.jpg", "c.jpg"]

    photos = all_photos(client)
    assert len(photos) == 3  # old "a" content is gone, "b" survives through its copy
    assert (320, 200) in {(p["width"], p["height"]) for p in photos}
    b = next(
        d
        for d in (client.get(f"/api/photos/{p['id']}").json() for p in photos)
        if len(d["files"]) == 2
    )
    assert [(os.path.basename(f["path"]), f["missing"]) for f in b["files"]] == [
        ("b_copy.jpg", False),
        ("b.jpg", True),
    ]
    assert client.get(f"/api/photos/{b['id']}/original").status_code == 200

    # Deleting the last copy hides the photo; the scan never deletes anything itself.
    os.remove(lib / "b_copy.jpg")
    scan(client)
    assert len(all_photos(client)) == 2


def test_scan_resumes_after_restart(env):
    lib, _, open_app = env
    for i in range(12):
        make_image(lib / f"p{i}.jpg", (i * 20, 255 - i * 20, i))
    (lib / "broken.jpg").write_bytes(b"this is not a jpeg")

    first = open_app()
    first.post("/api/scan/pause")
    first.post("/api/folders", json={"path": str(lib)})
    status = first.get("/api/scan/status").json()
    assert status["state"] == "paused" and status["library"]["photos"] == 0
    first.__exit__(None, None, None)  # shut down while paused, nothing processed

    second = open_app()
    status = second.get("/api/scan/status").json()
    assert status["state"] == "paused"  # the pause survives a restart
    second.post("/api/scan/resume")
    assert second.app.state.scanner.wait_idle(30)
    status = second.get("/api/scan/status").json()
    assert status["state"] == "idle"
    assert status["library"]["photos"] == 12
    assert status["errors"] == 1  # the broken file is reported, not retried forever
    assert status["stages"]["thumbnail"] == {"done": 12, "total": 12}


def test_folder_rules(env, tmp_path):
    lib, _, open_app = env
    make_image(lib / "inner" / "a.jpg", (5, 5, 5))
    client = open_app()
    assert client.post("/api/folders", json={"path": "relative/path"}).status_code == 400
    assert client.post("/api/folders", json={"path": str(tmp_path / "nope")}).status_code == 400
    client.post("/api/folders", json={"path": str(lib / "inner")})
    assert client.post("/api/folders", json={"path": str(lib / "inner")}).status_code == 409
    assert client.app.state.scanner.wait_idle(30)

    # Adding the parent absorbs the child folder without duplicating its photos.
    folders = client.post("/api/folders", json={"path": str(lib)}).json()
    assert client.app.state.scanner.wait_idle(30)
    assert [f["path"] for f in folders] == [str(lib)]
    assert len(all_photos(client)) == 1

    folder_id = folders[0]["id"]
    assert client.delete(f"/api/folders/{folder_id}").json() == []
    assert all_photos(client) == []
    assert (lib / "inner" / "a.jpg").exists()


def test_rejects_foreign_host(env):
    _, _, open_app = env
    client = open_app()
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 403


def test_raw_files_use_the_embedded_preview(env, monkeypatch):
    """LibRaw is faked here: no camera RAW file is available to the test suite."""
    import io
    import sys
    import types

    lib, config, open_app = env
    preview = io.BytesIO()
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = "2019:03:04 05:06:07"
    Image.new("RGB", (1600, 1200), (40, 90, 160)).save(preview, "JPEG", exif=exif)
    (lib / "shot.nef").write_bytes(b"fake raw container")

    class FakeRaw:
        sizes = types.SimpleNamespace(width=6000, height=4000, flip=6)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_thumb(self):
            return types.SimpleNamespace(format="jpeg", data=preview.getvalue())

    fake = types.SimpleNamespace(
        imread=lambda _: FakeRaw(), ThumbFormat=types.SimpleNamespace(JPEG="jpeg")
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)

    client = open_app()
    client.post("/api/folders", json={"path": str(lib)})
    assert client.app.state.scanner.wait_idle(30)
    (photo,) = all_photos(client)
    assert photo["taken_at"] == "2019-03-04T05:06:07"
    assert (photo["width"], photo["height"]) == (4000, 6000)  # full RAW size, upright
    with Image.open(config.thumb_path(photo["hash"], 1024)) as thumb:
        assert thumb.height > thumb.width  # preview rotated to match


def test_filters_combine(env):
    lib, _, open_app = env
    make_image(lib / "a.jpg", (1, 1, 1), exif_date="2020:01:15 10:00:00")
    make_image(lib / "b.jpg", (2, 2, 2), exif_date="2020:01:31 23:59:59")
    make_image(lib / "c.jpg", (3, 3, 3), exif_date="2021:07:01 08:00:00")
    make_image(lib / "plain.png", (4, 4, 4))  # no EXIF: no camera
    client = open_app()
    client.post("/api/folders", json={"path": str(lib)})
    assert client.app.state.scanner.wait_idle(30)

    facets = client.get("/api/facets").json()
    assert facets["cameras"] == [{"camera": "Acme Cam 1", "count": 3}]
    assert facets["first_date"] == "2020-01-15"

    def total(**params):
        return client.get("/api/timeline", params=params).json()["total"]

    assert total() == 4
    assert total(camera="Acme Cam 1") == 3
    assert total(date_from="2020-01-16", date_to="2020-01-31") == 1  # end date inclusive
    assert total(camera="Acme Cam 1", date_from="2021-01-01") == 1
    assert total(camera="Nope") == 0
    january = client.get(
        "/api/photos", params={"month": "2020-01", "date_to": "2020-01-15"}
    ).json()
    assert [p["taken_at"] for p in january] == ["2020-01-15T10:00:00"]
    assert client.get("/api/timeline", params={"date_from": "soon"}).status_code == 400

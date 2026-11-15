import time

from test_scan import env, make_image  # noqa: F401  (fixture)


def files_under(path):
    return sorted(p.relative_to(path) for p in path.rglob("*") if p.is_file())


def wait_ready(client, token, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/api/private/{token}").json()
        if view["state"] == "ready":
            return view
        time.sleep(0.05)
    raise AssertionError("private session did not finish")


def test_private_browsing_keeps_nothing(env):
    lib, config, open_app = env
    make_image(lib / "a.jpg", (200, 30, 30), exif_date="2021:06:01 10:00:00")
    make_image(lib / "sub" / "b.png", (30, 200, 30))
    (lib / "notes.txt").write_text("not a photo")
    client = open_app()
    before = files_under(config.data_dir)
    db_size = config.db_path.stat().st_size

    res = client.post("/api/private", json={"path": str(lib)})
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    token = res.json()["token"]
    view = wait_ready(client, token)
    assert view["total"] == 2 and view["done"] == 2 and view["errors"] == 0
    assert [p["name"] for p in view["photos"]][-1] == "a.jpg"  # oldest last
    a = next(p for p in view["photos"] if p["name"] == "a.jpg")
    assert a["taken_at"] == "2021-06-01T10:00:00" and a["camera"] == "Acme Cam 1"

    for size in (256, 1024):
        thumb = client.get(f"/api/private/{token}/thumbs/{a['id']}/{size}")
        assert thumb.status_code == 200 and thumb.content[:4] == b"RIFF"
        assert thumb.headers["cache-control"] == "no-store"
    original = client.get(f"/api/private/{token}/original/{a['id']}")
    assert original.content == (lib / "a.jpg").read_bytes()

    # Nothing was written: no thumbnails, no database rows, nothing in the library.
    assert files_under(config.data_dir) == before
    assert config.db_path.stat().st_size == db_size
    assert client.get("/api/timeline").json()["total"] == 0
    assert client.get("/api/folders").json() == []

    assert client.delete(f"/api/private/{token}").status_code == 200
    assert client.get(f"/api/private/{token}").status_code == 404
    assert client.get(f"/api/private/{token}/thumbs/{a['id']}/256").status_code == 404
    assert client.app.state.private._session is None


def test_opening_a_new_private_session_ends_the_old_one(env):
    lib, _config, open_app = env
    make_image(lib / "a.jpg", (10, 10, 200))
    client = open_app()
    first = client.post("/api/private", json={"path": str(lib)}).json()["token"]
    second = client.post("/api/private", json={"path": str(lib)}).json()["token"]
    assert client.get(f"/api/private/{first}").status_code == 404
    assert wait_ready(client, second)["total"] == 1


def test_private_rejects_bad_paths(env):
    lib, _config, open_app = env
    client = open_app()
    assert client.post("/api/private", json={"path": "relative"}).status_code == 400
    assert client.post("/api/private", json={"path": str(lib / "missing")}).status_code == 400
    assert client.get("/api/private/nope").status_code == 404

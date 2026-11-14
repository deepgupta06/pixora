"""Scenes, search, and events, with a fake scene engine that reads colours.

Blue photos are "beach", yellow photos are "birthday"; other colours carry no label.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_scan import make_image

from pixora_service.api import create_app
from pixora_service.config import Config
from pixora_service.scenes import all_labels

BLUE, YELLOW, GREEN = (20, 40, 230), (240, 220, 20), (20, 200, 40)
CONCEPTS = ["blue", "yellow", "green", "other"]
WORDS = {"ocean": "blue", "sea": "blue", "cake": "yellow", "grass": "green"}


def concept(rgb) -> str:
    r, g, b = (int(v) for v in rgb)
    if b > 150 and r < 100:
        return "blue"
    if r > 150 and g > 150 and b < 100:
        return "yellow"
    if g > 150 and r < 100:
        return "green"
    return "other"


def one_hot(name: str) -> np.ndarray:
    v = np.zeros(len(CONCEPTS), dtype=np.float32)
    v[CONCEPTS.index(name)] = 1
    return v


class FakeSceneEngine:
    version = "fake-scenes:1"

    def embed_images(self, images):
        names = [name for name, _ in all_labels()]
        embeddings, probs = [], []
        for img in images:
            c = concept(img.getpixel((img.width // 2, img.height // 2)))
            embeddings.append(one_hot(c))
            row = np.zeros(len(names))
            if c == "blue":
                row[names.index("beach")] = 0.4
            if c == "yellow":
                row[names.index("birthday")] = 0.3
            probs.append(row)
        return np.stack(embeddings), np.stack(probs)

    def embed_text(self, text):
        for word, c in WORDS.items():
            if word in text:
                return one_hot(c)
        return np.zeros(len(CONCEPTS), dtype=np.float32)

    def probability(self, cosine):
        return cosine


@pytest.fixture
def client(tmp_path):
    config = Config.load(data_dir=tmp_path / "data", workers=2)
    lib = tmp_path / "library"
    lib.mkdir()
    with TestClient(create_app(config, scene_factory=FakeSceneEngine)) as c:
        c.lib = lib
        yield c


def add_library(client):
    client.post("/api/folders", json={"path": str(client.lib)})
    assert client.app.state.scanner.wait_idle(30)


def total(client, **params):
    return client.get("/api/timeline", params=params).json()["total"]


def test_scene_tags_filter_and_search(client):
    for n in range(3):
        make_image(client.lib / f"sea{n}.jpg", BLUE, size=(640 + n, 480))
    for n in range(2):
        make_image(client.lib / f"lawn{n}.jpg", GREEN, size=(640 + n, 480))
    add_library(client)

    status = client.get("/api/scan/status").json()
    assert status["stages"]["scenes"] == {"done": 5, "total": 5} and status["scenes_note"] is None

    facets = client.get("/api/facets").json()
    assert facets["scenes"] == [{"name": "beach", "category": "nature", "count": 3}]
    assert facets["search_available"] is True
    assert total(client, scene="beach") == 3 and total(client, scene="forest") == 0

    # Search is by meaning, not by a stored label.
    assert total(client, q="the ocean") == 3
    assert total(client, q="green grass") == 2
    assert total(client, q="a spaceship") == 0
    assert total(client, q="   ") == 5  # an empty query is no filter

    month = client.get("/api/timeline", params={"q": "ocean"}).json()["buckets"][0]["month"]
    found = client.get("/api/photos", params={"month": month, "q": "ocean"}).json()
    assert len(found) == 3
    detail = client.get(f"/api/photos/{found[0]['id']}").json()
    assert [(t["name"], t["category"]) for t in detail["tags"]] == [("beach", "nature")]


def test_events_group_by_time_and_infer_the_occasion(client):
    # One afternoon: mostly yellow "birthday" photos, plus one that shows nothing in particular.
    for n in range(4):
        make_image(client.lib / f"party{n}.jpg", YELLOW, size=(640 + n, 480), exif_date=f"2020:05:01 1{n}:00:00")
    make_image(client.lib / "party_plain.jpg", GREEN, exif_date="2020:05:01 14:30:00")
    # Weeks later: a separate, small event.
    for n in range(2):
        make_image(client.lib / f"later{n}.jpg", YELLOW, size=(700 + n, 480), exif_date=f"2020:06:20 1{n}:00:00")
    add_library(client)

    assert client.get("/api/facets").json()["occasions"] == [{"name": "birthday", "count": 5}]
    # The plain photo belongs to the birthday through its event, not through its own content.
    assert total(client, occasion="birthday") == 5
    assert total(client, occasion="wedding") == 0

    month = client.get("/api/timeline", params={"occasion": "birthday"}).json()["buckets"][0]["month"]
    photos = client.get("/api/photos", params={"month": month, "occasion": "birthday"}).json()
    event = client.get(f"/api/photos/{photos[0]['id']}").json()["event"]
    assert (event["type"], event["photo_count"]) == ("birthday", 5)
    assert (event["start_at"], event["end_at"]) == ("2020-05-01T10:00:00", "2020-05-01T14:30:00")

    # Two photos are an event too, but too few to call it an occasion.
    later = client.get("/api/photos", params={"month": "2020-06"}).json()
    small = client.get(f"/api/photos/{later[0]['id']}").json()["event"]
    assert (small["type"], small["photo_count"]) == (None, 2)


def test_search_by_a_persons_name(client):
    make_image(client.lib / "a.jpg", BLUE)
    make_image(client.lib / "b.jpg", GREEN, size=(641, 480))
    add_library(client)
    conn = client.app.state.db.conn()
    photo_id = conn.execute("SELECT id FROM photos ORDER BY id LIMIT 1").fetchone()[0]
    conn.execute("INSERT INTO people(id, name) VALUES(1, 'Asha Rao')")
    conn.execute(
        "INSERT INTO faces(photo_id, x1, y1, x2, y2, score, size, turn, quality, embedding, engine_version, person_id) "
        "VALUES(?, 0, 0, 1, 1, 1, 100, 0, 1, x'00', 'test', 1)",
        (photo_id,),
    )
    assert total(client, q="asha rao") == 1


def test_without_the_scene_model_search_says_so(tmp_path):
    config = Config.load(data_dir=tmp_path / "data", workers=2)
    (tmp_path / "library").mkdir()
    make_image(tmp_path / "library" / "a.jpg", BLUE)
    with TestClient(create_app(config)) as c:  # real engine, no model files
        c.post("/api/folders", json={"path": str(tmp_path / "library")})
        assert c.app.state.scanner.wait_idle(30)
        status = c.get("/api/scan/status").json()
        assert status["scenes_note"] == "The scene model is not installed"
        assert c.get("/api/facets").json()["search_available"] is False
        assert c.get("/api/timeline").json()["total"] == 1
        assert c.get("/api/timeline", params={"q": "ocean"}).status_code == 503

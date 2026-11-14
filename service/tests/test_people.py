"""People: the scan stage, grouping, the question flow, and corrections.

A fake face engine stands in for the real models. It reads each test image's
colour as an identity, so embeddings are deterministic and need no downloads.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixora_service.api import create_app
from pixora_service.config import Config
from pixora_service.faces import ENGINE_THRESHOLDS, Face, Thresholds

DIM = 64
rng = np.random.default_rng(7)
# One base direction per person; a face is its person's direction plus a little noise.
BASES = {name: v / np.linalg.norm(v) for name, v in zip("ABCD", rng.normal(size=(4, DIM)))}
# "B2" is person B looking different enough to land in the "ask the user" band.
COLOURS = {(250, 10, 10): "A", (10, 250, 10): "B", (10, 10, 250): "C", (250, 250, 10): "D", (10, 250, 250): "B2"}


def embedding(person: str, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=DIM)
    noise /= np.linalg.norm(noise)
    if person == "B2":
        # About 0.375 similar to B's own faces: inside the suggestion band, below grouping.
        other = BASES["D"] - (BASES["D"] @ BASES["B"]) * BASES["B"]
        v = 0.37 * BASES["B"] + 0.89 * other / np.linalg.norm(other) + 0.05 * noise
    else:
        v = BASES[person] + 0.2 * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEngine:
    name = "fake"
    version = "fake:1"
    thresholds = Thresholds(cluster=0.40, auto=0.45, suggest=0.35)

    def analyze(self, full, work, scale):
        h, w = full.shape[:2]
        faces = []
        # Each image is split into left and right halves; a non-grey half is a face.
        for i, x0 in enumerate((0, w // 2)):
            b, g, r = (int(v) for v in full[h // 2, x0 + w // 4])
            person = COLOURS.get((round(r / 10) * 10, round(g / 10) * 10, round(b / 10) * 10))
            if person is None:
                continue
            seed = int(full[0, 0, 0]) * 2 + i  # the corner pixel varies per photo
            box = np.array([x0 + w * 0.1, h * 0.2, x0 + w * 0.4, h * 0.8], dtype=np.float32)
            cx = (box[0] + box[2]) / 2
            kps = np.array([[cx - 20, h * 0.4], [cx + 20, h * 0.4], [cx, h * 0.5], [cx - 15, h * 0.6], [cx + 15, h * 0.6]], dtype=np.float32)
            faces.append(Face(box, 0.9, kps, embedding(person, seed)))
        return faces


def make_photo(path, left, right=None, n=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 480), (128, 128, 128))
    for x0, person in ((0, left), (320, right)):
        if person:
            colour = next(c for c, p in COLOURS.items() if p == person)
            img.paste(colour, (x0, 0, x0 + 320, 480))
    img.putpixel((0, 0), (0, 0, n))  # makes every photo unique and seeds its embeddings
    img.save(path, "PNG")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(ENGINE_THRESHOLDS, FakeEngine.version, FakeEngine.thresholds)
    config = Config.load(data_dir=tmp_path / "data", workers=2)
    lib = tmp_path / "library"
    lib.mkdir()
    with TestClient(create_app(config, engine_factory=FakeEngine)) as c:
        c.lib = lib
        yield c


def scan(client):
    client.post("/api/scan/start")
    assert client.app.state.scanner.wait_idle(30)


def add_library(client):
    client.post("/api/folders", json={"path": str(client.lib)})
    assert client.app.state.scanner.wait_idle(30)


def people(client):
    return {p["name"]: p for p in client.get("/api/people").json()["people"]}


def test_faces_are_found_and_unknown_people_grouped(client):
    for n in range(6):
        make_photo(client.lib / f"ab{n}.png", "A", "B", n)
    for n in range(6, 8):
        make_photo(client.lib / f"c{n}.png", "C", None, n)  # only two faces: too few to ask about
    make_photo(client.lib / "empty.png", None, None, 9)
    add_library(client)

    status = client.get("/api/scan/status").json()
    assert status["stages"]["faces"] == {"done": 9, "total": 9}
    assert client.get("/api/people").json() == {"people": [], "unnamed_groups": 2}

    q = client.get("/api/people/question").json()
    assert q["type"] == "name" and len(q["face_ids"]) == 6 and q["photo_count"] == 6
    assert len(q["sample_face_ids"]) == 6
    assert client.get(f"/api/faces/{q['face_ids'][0]}/crop").headers["content-type"] == "image/webp"


def test_naming_a_group_learns_the_person(client):
    for n in range(5):
        make_photo(client.lib / f"ab{n}.png", "A", "B", n)
    add_library(client)

    q = client.get("/api/people/question").json()
    client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Asha"})
    q = client.get("/api/people/question").json()
    client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Bina"})
    assert client.get("/api/people/question").json()["type"] == "done"
    assert {n: p["face_count"] for n, p in people(client).items()} == {"Asha": 5, "Bina": 5}

    # New photos of known people are recognised with no further questions.
    for n in range(5, 9):
        make_photo(client.lib / f"later{n}.png", "A", "B", n)
    scan(client)
    assert {n: p["face_count"] for n, p in people(client).items()} == {"Asha": 9, "Bina": 9}
    assert client.get("/api/people/question").json()["type"] == "done"

    # The person filter and facets use the result.
    asha = people(client)["Asha"]["id"]
    assert client.get("/api/timeline", params={"person_id": asha}).json()["total"] == 9
    assert [p["name"] for p in client.get("/api/facets").json()["people"]] == ["Asha", "Bina"]


def test_follow_up_questions_and_not_this_person(client):
    for n in range(5):
        make_photo(client.lib / f"b{n}.png", "B", None, n)
    for n in range(5, 9):
        make_photo(client.lib / f"b2_{n}.png", "B2", None, n)  # B, but looking different
    add_library(client)

    q = client.get("/api/people/question").json()
    assert q["type"] == "name" and len(q["face_ids"]) == 5
    client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Bina"})

    # The uncertain group is asked about as a whole: one question, not four.
    q = client.get("/api/people/question").json()
    assert q["type"] == "confirm" and q["name"] == "Bina" and len(q["face_ids"]) == 4
    after = client.post("/api/people/confirm", json={"person_id": q["person_id"], "face_ids": q["face_ids"], "answer": True}).json()
    assert after["type"] == "done"
    assert people(client)["Bina"]["face_count"] == 9


def test_answering_no_is_remembered(client):
    for n in range(5):
        make_photo(client.lib / f"b{n}.png", "B", None, n)
    for n in range(5, 9):
        make_photo(client.lib / f"b2_{n}.png", "B2", None, n)
    add_library(client)
    q = client.get("/api/people/question").json()
    bina = client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Bina"}).json()["person_id"]
    q = client.get("/api/people/question").json()
    after = client.post("/api/people/confirm", json={"person_id": bina, "face_ids": q["face_ids"], "answer": False}).json()

    # Not Bina, so the group is offered for a name of its own, and never as Bina again.
    assert after["type"] == "name" and sorted(after["face_ids"]) == sorted(q["face_ids"])
    scan(client)
    assert client.get("/api/people/question").json()["type"] == "name"
    assert people(client)["Bina"]["face_count"] == 5


def test_at_most_three_follow_ups_per_person(client):
    for n in range(5):
        make_photo(client.lib / f"b{n}.png", "B", None, n)
    add_library(client)
    q = client.get("/api/people/question").json()
    bina = client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Bina"}).json()["person_id"]

    # Five single uncertain faces arrive; only three questions may be asked.
    for n in range(10, 15):
        make_photo(client.lib / f"b2_{n}.png", "B2", None, n)
    scan(client)
    asked = 0
    conn = client.app.state.db.conn()
    conn.execute("UPDATE faces SET cluster_id=NULL, dismissed=1 WHERE person_id IS NULL")  # keep them as loose faces
    while (q := client.get("/api/people/question").json())["type"] == "confirm":
        assert len(q["face_ids"]) == 1
        client.post("/api/people/confirm", json={"person_id": bina, "face_ids": q["face_ids"], "answer": False})
        conn.execute("UPDATE faces SET cluster_id=NULL, dismissed=1 WHERE person_id IS NULL")
        asked += 1
        assert asked <= 3
    assert asked <= 3


def test_corrections_win_and_merges_are_explicit(client):
    for n in range(5):
        make_photo(client.lib / f"ab{n}.png", "A", "B", n)
    add_library(client)
    for name in ("Asha", "Bina"):
        q = client.get("/api/people/question").json()
        client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": name})
    make_photo(client.lib / "new.png", "A", None, 20)
    scan(client)

    photo = next(
        d for d in (client.get(f"/api/photos/{p['id']}").json() for p in all_photos(client)) if len(d["faces"]) == 1
    )
    face = photo["faces"][0]
    assert (face["name"], face["source"]) == ("Asha", "auto")
    assert 0 <= face["x1"] < face["x2"] <= 1

    # "Not Asha": the face leaves Asha and is not put back by the next scan.
    client.post(f"/api/faces/{face['id']}/reject")
    scan(client)
    assert client.get(f"/api/photos/{photo['id']}").json()["faces"][0]["person_id"] is None
    assert people(client)["Asha"]["face_count"] == 5

    # The user names it directly; that overrules the earlier "not".
    client.post(f"/api/faces/{face['id']}/assign", json={"name": "asha"})
    again = client.get(f"/api/photos/{photo['id']}").json()["faces"][0]
    assert (again["name"], again["source"]) == ("Asha", "user")

    # Two named people merge only when the user renames one to the other.
    assert set(people(client)) == {"Asha", "Bina"}
    bina = people(client)["Bina"]["id"]
    merged = client.patch(f"/api/people/{bina}", json={"name": "Asha"}).json()["person_id"]
    assert merged == people(client)["Asha"]["id"] and people(client)["Asha"]["face_count"] == 11
    assert client.patch(f"/api/people/{merged}", json={"name": "  "}).status_code == 400


def test_skipping_a_group_stops_the_question(client):
    for n in range(4):
        make_photo(client.lib / f"c{n}.png", "C", None, n)
    add_library(client)
    q = client.get("/api/people/question").json()
    assert client.post("/api/people/dismiss", json={"face_ids": q["face_ids"]}).json()["type"] == "done"
    scan(client)
    assert client.get("/api/people/question").json()["type"] == "done"
    assert client.post("/api/people/name", json={"face_ids": [999999], "name": "X"}).status_code == 404


def test_missing_models_leave_the_library_usable(tmp_path):
    config = Config.load(data_dir=tmp_path / "data", workers=2)
    (tmp_path / "library").mkdir()
    make_photo(tmp_path / "library" / "a.png", "A", None, 1)
    with TestClient(create_app(config)) as c:  # real engine, no model files
        c.post("/api/folders", json={"path": str(tmp_path / "library")})
        assert c.app.state.scanner.wait_idle(30)
        status = c.get("/api/scan/status").json()
        assert status["library"]["photos"] == 1
        assert status["faces_note"] == "Face models are not installed"
        assert status["stages"]["faces"] == {"done": 0, "total": 1}


def all_photos(client):
    out = []
    for b in client.get("/api/timeline").json()["buckets"]:
        out += client.get("/api/photos", params={"month": b["month"]}).json()
    return out

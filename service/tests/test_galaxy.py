"""Galaxy: the People-mode layout and the data endpoint."""

import numpy as np
from fastapi.testclient import TestClient

from pixora_service.api import create_app
from pixora_service.config import Config
from pixora_service.galaxy import people_layout

from test_people import FakeEngine, make_photo


def test_layout_keeps_each_person_together_and_people_apart():
    rng = np.random.default_rng(3)
    bases = rng.normal(size=(3, 32))
    counts = [40, 12, 1]
    embeddings = np.concatenate([b + 0.1 * rng.normal(size=(n, 32)) for b, n in zip(bases, counts)])
    person_ids = [p for p, n in zip((1, 2, 3), counts) for _ in range(n)]
    # Two more faces: one in an unnamed group, one on its own.
    embeddings = np.concatenate([embeddings, rng.normal(size=(2, 32))]).astype(np.float32)
    positions, people = people_layout(embeddings, person_ids + [None, None], [None] * 53 + [9, None])

    assert positions.shape == (55, 3) and np.isfinite(positions).all()
    for person, (centre, radius) in people.items():
        mine = positions[[i for i, p in enumerate(person_ids) if p == person]]
        assert (np.linalg.norm(mine - centre, axis=1) <= radius + 1e-6).all()
    centres = {p: c for p, (c, _) in people.items()}
    for a in people:
        for b in people:
            if a < b:
                assert np.linalg.norm(centres[a] - centres[b]) >= people[a][1] + people[b][1]
    # Unnamed faces sit outside every named cluster.
    core = max(np.linalg.norm(c) + r for c, r in people.values())
    assert (np.linalg.norm(positions[53:], axis=1) > core).all()


def test_galaxy_data(tmp_path):
    lib = tmp_path / "lib"
    make_photo(lib / "a1.jpg", "A", "B", n=1)
    make_photo(lib / "a2.jpg", "A", "B", n=2)
    make_photo(lib / "a3.jpg", "A", n=3)
    config = Config.load(data_dir=tmp_path / "data", workers=1)
    app = create_app(config, autostart=False, engine_factory=FakeEngine)
    with TestClient(app) as client:
        assert client.get("/api/galaxy/data").json()["faces"] == []
        client.post("/api/folders", json={"path": str(lib)})
        app.state.scanner.wait_idle(30)
        data = client.get("/api/galaxy/data").json()
        assert len(data["faces"]) == 5 and data["people"] == []

        ids = [f["id"] for f in client.get("/api/photos/%d" % data["faces"][0]["photo_id"]).json()["faces"]]
        client.post("/api/faces/%d/assign" % ids[0], json={"name": "Asha"})
        data = client.get("/api/galaxy/data").json()
        assert [p["name"] for p in data["people"]] == ["Asha"]
        named = [f for f in data["faces"] if f["person_id"] is not None]
        assert len(named) == data["people"][0]["face_count"] >= 1
        assert {"hash", "taken_at", "width", "height", "x", "y", "z"} <= set(data["faces"][0])

        # The shared filters apply.
        one = client.get("/api/galaxy/data", params={"person_id": data["people"][0]["id"]}).json()
        assert 0 < len(one["faces"]) <= 5
        assert client.get("/api/galaxy/data", params={"date_from": "2999-01-01"}).json()["faces"] == []

"""Editing and generation: everything around the models.

A fake engine stands in for the models and records what it was asked, so
these tests need no downloads and no graphics card.
"""

import base64
import io
import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixora_service import compose, identity, model_store
from pixora_service.api import create_app
from pixora_service.config import Config
from pixora_service.identity import Candidate, find_mentions, pick_references
from pixora_service.studio import Cancelled

from test_people import FakeEngine as FakeFaces
from test_people import ENGINE_THRESHOLDS, make_photo


class FakeStudioEngine:
    def __init__(self):
        self.calls: list[tuple] = []
        self.unloaded = False
        self.gate: threading.Event | None = None  # when set, painting waits for it

    def _flat(self, size, colour, progress):
        if self.gate is not None:
            while not self.gate.wait(0.01):
                progress.check()
        progress.report(0.5)
        return Image.new("RGB", size, colour)

    def matte(self, image, progress):
        self.calls.append(("matte", image.size))
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (0, 0, image.width // 2, image.height))  # the left half is the subject
        return mask

    def upscale(self, image, scale, progress):
        self.calls.append(("upscale", image.size, scale))
        return image.resize((round(image.width * scale), round(image.height * scale)))

    def restore_faces(self, image, strength, progress):
        self.calls.append(("restore_faces", strength))
        return image

    def inpaint(self, image, mask, prompt, negative, seed, progress):
        self.calls.append(("inpaint", image.size, mask.size, prompt, seed))
        return self._flat(image.size, (0, 0, 255), progress)

    def restyle(self, image, prompt, negative, style, strength, seed, progress):
        self.calls.append(("restyle", image.size, prompt, style.id, strength, seed))
        return self._flat(image.size, (255, seed % 256, 255), progress)

    def generate(self, prompt, negative, width, height, seed, style, references, progress):
        self.calls.append(("generate", prompt, (width, height), seed, style.id if style else None, references))
        return self._flat((width, height), (seed % 256, 100, 100), progress)

    def unload(self):
        self.unloaded = True


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(ENGINE_THRESHOLDS, FakeFaces.version, FakeFaces.thresholds)
    config = Config.load(data_dir=tmp_path / "data", workers=2, creations_dir=tmp_path / "creations")
    lib = tmp_path / "library"
    lib.mkdir()
    engine = FakeStudioEngine()
    with TestClient(create_app(config, engine_factory=FakeFaces, studio_factory=lambda: engine)) as c:
        c.lib, c.engine, c.config = lib, engine, config
        yield c


def scan(client):
    client.post("/api/folders", json={"path": str(client.lib)})
    assert client.app.state.scanner.wait_idle(60)


def photo_ids(client):
    months = client.get("/api/timeline").json()["buckets"]
    return [p["id"] for b in months for p in client.get(f"/api/photos?month={b['month']}").json()]


def run(client, **body):
    r = client.post("/api/studio/jobs", json=body)
    assert r.status_code == 200, r.text
    job = client.app.state.studio.wait(r.json()["id"])
    assert client.app.state.scanner.wait_idle(60)
    return client.get(f"/api/studio/jobs/{job.id}").json()


@pytest.fixture
def one_photo(client):
    make_photo(client.lib / "a.png", "A", n=1)
    scan(client)
    (photo_id,) = photo_ids(client)
    return photo_id


# ---- names in prompts -----------------------------------------------------------

PEOPLE = [
    Candidate(3, "Amit", 30, 198),
    Candidate(37, "Amit Kumar", 370, 61),
    Candidate(2, "Deep(Me)", 20, 1341),
    Candidate(22, "Piu Di", 220, 236),
    Candidate(23, "Butti Di", 230, 228),
]


def test_first_name_shared_by_two_people_is_ambiguous():
    (m,) = find_mentions("Create an image of Amit playing guitar on stage", PEOPLE)
    assert (m.text, m.status) == ("Amit", "ambiguous")
    # The exact name is offered first.
    assert [c.name for c in m.candidates] == ["Amit", "Amit Kumar"]


def test_full_name_wins_over_the_shorter_one():
    (m,) = find_mentions("amit kumar playing guitar", PEOPLE)
    assert (m.text, m.status, [c.person_id for c in m.candidates]) == ("amit kumar", "match", [37])


def test_first_name_and_bracketed_note():
    prompt = "A portrait of Deep and Piu at the beach"
    deep, piu = find_mentions(prompt, PEOPLE)
    assert (deep.status, deep.candidates[0].name) == ("match", "Deep(Me)")
    assert (piu.status, piu.candidates[0].name) == ("match", "Piu Di")
    assert prompt[deep.start : deep.end] == "Deep"


def test_a_shared_last_word_is_not_a_mention():
    assert find_mentions("di is a short word", PEOPLE) == []


def test_unknown_name_is_reported_not_guessed():
    mentions = find_mentions("Create an image of Rahul Sharma playing guitar in Paris", PEOPLE)
    assert [(m.text, m.status) for m in mentions] == [("Rahul Sharma", "unknown")]
    assert mentions[0].candidates == []


def test_no_names_in_an_ordinary_prompt():
    assert find_mentions("A red bicycle leaning on a wall. Sunny day", PEOPLE) == []


def test_prompt_for_the_model_has_no_names():
    prompt = "Amit playing guitar"
    assert identity.describe_without_names(prompt, find_mentions(prompt, PEOPLE)) == "a person playing guitar"


# ---- reference faces ------------------------------------------------------------


def face_row(i, vec, **over):
    row = {
        "id": i, "photo_id": i, "embedding": np.asarray(vec, dtype=np.float32).tobytes(), "quality": 1,
        "size": 100.0, "turn": 0.05, "score": 0.9, "assignment_source": "user",
    }
    row.update(over)
    return row


def test_references_prefer_confirmed_clear_large_faces():
    a = [1, 0, 0]
    rows = [
        face_row(1, a, size=60.0),
        face_row(2, a, size=300.0, assignment_source="auto"),
        face_row(3, a, size=200.0),
        face_row(4, a, size=400.0, quality=0, turn=0.45),
        face_row(5, a, size=120.0),
    ]
    assert [rows[i]["id"] for i in pick_references(rows, 3)] == [3, 5, 1]


def test_references_leave_out_a_face_unlike_the_rest():
    rows = [face_row(i, [1, 0.05 * i, 0]) for i in range(1, 5)]
    rows.append(face_row(9, [0, 1, 0], size=900.0))  # huge, but someone else
    assert 9 not in [rows[i]["id"] for i in pick_references(rows, 4)]


def test_references_take_one_face_per_photo():
    rows = [face_row(1, [1, 0, 0], photo_id=7), face_row(2, [1, 0, 0], photo_id=7), face_row(3, [1, 0, 0], photo_id=8)]
    assert sorted(rows[i]["photo_id"] for i in pick_references(rows, 3)) == [7, 8]


def test_no_faces_no_references():
    assert pick_references([]) == []


# ---- image arithmetic -----------------------------------------------------------


def test_working_size_keeps_shape_and_divides_by_eight():
    assert compose.working_size(4608, 3456) == (1024, 768)
    w, h = compose.working_size(1000, 333)
    assert w % 8 == 0 and h % 8 == 0 and w == 1024


def test_mask_box_surrounds_the_painted_area_with_context():
    mask = Image.new("L", (2000, 1500), 0)
    mask.paste(255, (900, 700, 1000, 800))
    x1, y1, x2, y2 = compose.mask_box(mask)
    assert (x2 - x1, y2 - y1) == (512, 512)
    assert x1 <= 900 and y1 <= 700 and x2 >= 1000 and y2 >= 800
    assert compose.mask_box(Image.new("L", (64, 64), 0)) is None


def test_mask_box_stays_inside_a_small_photo():
    mask = Image.new("L", (300, 200), 0)
    mask.paste(255, (280, 180, 300, 200))
    assert compose.mask_box(mask) == (0, 0, 300, 200)


def test_extended_canvas_masks_only_the_new_area():
    canvas, mask = compose.extended_canvas(Image.new("RGB", (100, 80), (200, 0, 0)), 50, 0, 0, 0, overlap=10)
    assert canvas.size == mask.size == (150, 80)
    assert mask.getpixel((10, 40)) == 255 and mask.getpixel((55, 40)) == 255  # new area and the join
    assert mask.getpixel((100, 40)) == 0 and mask.getpixel((149, 40)) == 0  # untouched sides stay


def test_keep_original_restores_the_photo_exactly_away_from_the_join():
    original = Image.new("RGB", (100, 80), (200, 0, 0))
    out = compose.keep_original(Image.new("RGB", (150, 80), (0, 0, 255)), original, (50, 0), (True, False, False, False), 10)
    assert out.getpixel((149, 40)) == (200, 0, 0) and out.getpixel((70, 0)) == (200, 0, 0)
    assert out.getpixel((10, 40)) == (0, 0, 255)


# ---- jobs -----------------------------------------------------------------------


def test_remove_background_saves_a_new_transparent_file(client, one_photo):
    original = (client.lib / "a.png").read_bytes()
    job = run(client, operation="remove_background", photo_id=one_photo)
    assert job["state"] == "done" and job["progress"] == 1
    (result,) = job["results"]
    assert result["origin"] == "edited" and result["id"] != one_photo

    # The original is untouched and the result is a separate file in the creations folder.
    assert (client.lib / "a.png").read_bytes() == original
    detail = client.get(f"/api/photos/{result['id']}").json()
    (saved,) = detail["files"]
    assert saved["path"].startswith(str(client.config.creations_dir)) and saved["path"].endswith(".png")
    with Image.open(saved["path"]) as img:
        assert img.mode == "RGBA" and img.size == (640, 480)
        assert img.getpixel((10, 10))[3] == 255 and img.getpixel((630, 10))[3] == 0

    # It is in the library, marked, and remembers where it came from.
    assert set(photo_ids(client)) == {one_photo, result["id"]}
    assert detail["origin"] == "edited"
    assert detail["creation"]["operation"] == "remove_background"
    assert detail["creation"]["source"]["id"] == one_photo
    assert client.get(f"/api/photos/{one_photo}").json()["origin"] is None


def test_results_never_overwrite_each_other(client, one_photo):
    first = run(client, operation="remove_background", photo_id=one_photo)
    second = run(client, operation="enhance", photo_id=one_photo)
    paths = {
        client.get(f"/api/photos/{j['results'][0]['id']}").json()["files"][0]["path"] for j in (first, second)
    }
    assert len(paths) == 2


def test_replace_background_keeps_the_subject_pixels(client, one_photo):
    job = run(client, operation="replace_background", photo_id=one_photo, prompt="a beach at sunset")
    path = client.get(f"/api/photos/{job['results'][0]['id']}").json()["files"][0]["path"]
    with Image.open(path) as img:
        r, g, b = img.convert("RGB").getpixel((100, 240))  # subject: the photo's own red
        assert r > 200 and b < 60
        r, g, b = img.convert("RGB").getpixel((600, 240))  # background: what the model painted
        assert b > 200 and r < 60
    kinds = [c[0] for c in client.engine.calls]
    assert kinds == ["matte", "inpaint"]
    assert client.engine.calls[1][3] == "a beach at sunset"


def test_replace_background_needs_a_description(client, one_photo):
    r = client.post("/api/studio/jobs", json={"operation": "replace_background", "photo_id": one_photo})
    assert r.status_code == 400


def test_enhance_enlarges(client, one_photo):
    job = run(client, operation="enhance", photo_id=one_photo)
    assert (job["results"][0]["width"], job["results"][0]["height"]) == (2560, 1920)
    assert [c[0] for c in client.engine.calls] == ["upscale"]


def test_restore_fixes_faces_first(client, one_photo):
    run(client, operation="restore", photo_id=one_photo)
    assert [c[0] for c in client.engine.calls] == ["restore_faces", "upscale"]


def test_stylize_makes_variations_with_consecutive_seeds(client, one_photo):
    job = run(client, operation="stylize", photo_id=one_photo, style="anime", count=2, seed=41, strength=0.5)
    assert len(job["results"]) == 2
    calls = client.engine.calls
    assert [(c[3], c[4], c[5]) for c in calls] == [("anime", 0.5, 41), ("anime", 0.5, 42)]
    assert calls[0][1] == (1024, 768)  # worked on at the model's size, same shape as the photo
    seeds = [client.get(f"/api/photos/{r['id']}").json()["creation"]["params"]["seed"] for r in job["results"]]
    assert seeds == [41, 42]


def test_stylize_rejects_unknown_and_generate_only_styles(client, one_photo):
    for style in ("nope", "photo", None):
        r = client.post("/api/studio/jobs", json={"operation": "stylize", "photo_id": one_photo, "style": style})
        assert r.status_code == 400


def painted(box, size=(64, 48)):
    """A mask as the UI sends it: opaque strokes on a transparent canvas."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    img.paste((255, 255, 255, 255), box)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_inpaint_changes_only_the_painted_area(client, one_photo):
    job = run(client, operation="inpaint", photo_id=one_photo, prompt="a hat", mask=painted((40, 20, 50, 30)))
    detail = client.get(f"/api/photos/{job['results'][0]['id']}").json()
    with Image.open(detail["files"][0]["path"]) as img:
        assert img.size == (640, 480)
        assert img.getpixel((450, 250))[2] > 200  # inside the painted area
        assert img.getpixel((100, 100))[0] > 200 and img.getpixel((100, 100))[2] < 60  # far from it
    assert "mask" not in detail["creation"]["params"]


def test_inpaint_without_a_prompt_erases(client, one_photo):
    run(client, operation="inpaint", photo_id=one_photo, mask=painted((40, 20, 50, 30)))
    assert "empty background" in client.engine.calls[0][3]


def test_inpaint_needs_a_painted_area(client, one_photo):
    assert client.post("/api/studio/jobs", json={"operation": "inpaint", "photo_id": one_photo}).status_code == 400
    job = run(client, operation="inpaint", photo_id=one_photo, mask=painted((0, 0, 0, 0)))
    assert job["state"] == "error" and job["results"] == []


def test_extend_grows_the_canvas_and_keeps_the_photo(client, one_photo):
    job = run(client, operation="extend", photo_id=one_photo, right=0.5)
    result = job["results"][0]
    assert (result["width"], result["height"]) == (960, 480)
    with Image.open(client.get(f"/api/photos/{result['id']}").json()["files"][0]["path"]) as img:
        assert img.getpixel((100, 240))[0] > 200  # the photo
        assert img.getpixel((900, 240))[2] > 200  # the new area
    assert client.post("/api/studio/jobs", json={"operation": "extend", "photo_id": one_photo}).status_code == 400


def test_generate_records_settings_for_reuse(client):
    job = run(
        client, operation="generate", prompt="a red bicycle", size="wide", style="cinematic",
        negative_prompt="rain", seed=7, count=2,
    )
    assert [(r["width"], r["height"], r["origin"]) for r in job["results"]] == [(1344, 768, "generated")] * 2
    history = client.get("/api/studio/history?kind=generated").json()
    assert [c["photo"]["id"] for c in history] == [r["id"] for r in reversed(job["results"])]
    params = history[0]["params"]
    assert (params["prompt"], params["size"], params["style"], params["negative_prompt"], params["seed"]) == (
        "a red bicycle", "wide", "cinematic", "rain", 8,
    )
    assert history[0]["source"] is None
    call = client.engine.calls[0]
    assert call[1].startswith("a red bicycle, cinematic") and call[4] == "cinematic" and call[5] == []


def test_generate_needs_a_prompt_and_a_known_size(client):
    assert client.post("/api/studio/jobs", json={"operation": "generate", "prompt": " "}).status_code == 400
    assert client.post("/api/studio/jobs", json={"operation": "generate", "prompt": "x", "size": "huge"}).status_code == 400
    assert client.post("/api/studio/jobs", json={"operation": "bake"}).status_code == 400


def test_generate_a_named_person_uses_their_reference_faces(client):
    for n in range(4):
        make_photo(client.lib / f"a{n}.png", "A", n=n)
    make_photo(client.lib / "c.png", "C", n=9)
    scan(client)
    q = client.get("/api/people/question").json()
    amit = client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Amit"}).json()["person_id"]

    people = client.post("/api/studio/people", json={"prompt": "Amit playing guitar on stage"}).json()
    (mention,) = people["mentions"]
    assert mention["status"] == "match" and mention["candidates"][0]["person_id"] == amit

    job = run(client, operation="generate", prompt="Amit playing guitar on stage", person_id=amit)
    assert job["state"] == "done"
    call = client.engine.calls[0]
    assert call[1].startswith("a person playing guitar on stage")
    references = call[5]
    assert 1 <= len(references) <= identity.MAX_REFERENCES
    faces_of_amit = {
        f["id"] for pid in photo_ids(client) for f in client.get(f"/api/photos/{pid}").json()["faces"]
        if f["person_id"] == amit
    }
    assert {r.face_id for r in references} <= faces_of_amit
    assert all(r.path and r.path.startswith(str(client.lib)) for r in references)
    params = client.get(f"/api/photos/{job['results'][0]['id']}").json()["creation"]["params"]
    assert params["person_id"] == amit and params["person_name"] == "Amit"


def test_generated_pictures_of_a_person_are_never_used_as_references(client):
    for n in range(3):
        make_photo(client.lib / f"a{n}.png", "A", n=n)
    scan(client)
    q = client.get("/api/people/question").json()
    amit = client.post("/api/people/name", json={"face_ids": q["face_ids"], "name": "Amit"}).json()["person_id"]
    db = client.app.state.db
    real = {r.photo_id for r in identity.references_for(db, amit)}
    # A made picture that shows the same face, filed under the same person.
    made = client.config.creations_dir / "fake.png"
    make_photo(made, "A", n=50)
    photo_id = client.app.state.scanner.import_file(str(made), str(made.parent))
    with db.write() as conn:
        conn.execute("UPDATE photos SET origin='generated' WHERE id=?", (photo_id,))
    client.app.state.scanner.start(walk=False)
    assert client.app.state.scanner.wait_idle(60)
    assert db.conn().execute("SELECT count(*) FROM faces WHERE photo_id=? AND person_id=?", (photo_id, amit)).fetchone()[0]
    assert {r.photo_id for r in identity.references_for(db, amit)} == real


def test_generate_for_an_unknown_or_faceless_person_is_refused(client):
    r = client.post("/api/studio/jobs", json={"operation": "generate", "prompt": "x", "person_id": 99})
    assert r.status_code == 404
    with client.app.state.db.write() as conn:
        nobody = conn.execute("INSERT INTO people(name) VALUES('Nobody')").lastrowid
    r = client.post("/api/studio/jobs", json={"operation": "generate", "prompt": "x", "person_id": nobody})
    assert r.status_code == 400 and "Nobody" in r.json()["detail"]


def test_cancel_stops_a_running_job_and_saves_nothing(client, one_photo):
    client.engine.gate = threading.Event()
    job_id = client.post(
        "/api/studio/jobs", json={"operation": "stylize", "photo_id": one_photo, "style": "anime"}
    ).json()["id"]
    queued = client.post(
        "/api/studio/jobs", json={"operation": "generate", "prompt": "a tree"}
    ).json()
    assert queued["state"] == "queued"
    studio = client.app.state.studio
    import time

    deadline = time.monotonic() + 10
    while studio.get(job_id).state != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    # While a job runs, the scan stands aside.
    assert not client.app.state.scanner._free.is_set()

    assert client.post(f"/api/studio/jobs/{queued['id']}/cancel").json()["state"] == "cancelled"
    client.post(f"/api/studio/jobs/{job_id}/cancel")
    job = studio.wait(job_id)
    assert job.state == "cancelled" and job.result_ids == []
    assert client.app.state.scanner._free.is_set()
    assert photo_ids(client) == [one_photo]
    assert not client.config.creations_dir.exists() or not any(client.config.creations_dir.iterdir())
    assert client.post("/api/studio/jobs/999/cancel").status_code == 404


def test_a_failing_model_reports_an_error(client, one_photo):
    def boom(*args):
        raise RuntimeError("out of memory")

    client.engine.upscale = boom
    job = run(client, operation="enhance", photo_id=one_photo)
    assert job["state"] == "error" and "out of memory" in job["error"] and job["results"] == []


def test_unknown_photo(client):
    assert client.post("/api/studio/jobs", json={"operation": "enhance", "photo_id": 123}).status_code == 404


def test_progress_raises_once_cancelled():
    from pixora_service.studio import Job, Progress

    job = Job(1, "enhance", {})
    progress = Progress(job).part(0.5, 1.0)
    progress.report(0.5, "Halfway")
    assert (job.progress, job.message) == (0.75, "Halfway")
    job.cancel.set()
    with pytest.raises(Cancelled):
        progress.report(0.6)


# ---- models -----------------------------------------------------------------------


def test_missing_models_are_named_before_a_job_starts(tmp_path):
    config = Config.load(
        data_dir=tmp_path / "data", workers=1, models_dir=tmp_path / "models", creations_dir=tmp_path / "creations"
    )
    with TestClient(create_app(config, engine_factory=FakeFaces)) as c:
        status = c.get("/api/studio/status").json()
        assert status["features"]["generate"] == ["sdxl", "sdxl-vae"]
        assert not any(m["installed"] for m in status["models"])
        assert {s["id"] for s in status["styles"] if s["restyle"]} >= {"anime", "3d"}
        r = c.post("/api/studio/jobs", json={"operation": "generate", "prompt": "a tree", "style": "anime"})
        assert r.status_code == 409 and "Stable Diffusion XL" in r.json()["detail"] and "Anime" in r.json()["detail"]


def test_a_model_is_installed_only_when_every_file_is_there(tmp_path):
    folder = model_store.model_dir(tmp_path, "sdxl-vae")
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    assert not model_store.is_installed(tmp_path, "sdxl-vae")
    (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert model_store.is_installed(tmp_path, "sdxl-vae")
    assert model_store.missing_for(tmp_path, "generate") == ["sdxl"]
    assert json.dumps(sorted(model_store.FEATURES))  # every feature names known models
    assert all(m in model_store.MODELS for needs in model_store.FEATURES.values() for m in needs)

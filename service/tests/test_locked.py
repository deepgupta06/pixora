"""Locked pictures: an incognito picture encrypted so that only Pixora, given the key, can read it."""

import base64
import io
import logging
import sys
import tempfile

import pytest
from PIL import Image, UnidentifiedImageError

from pixora_service import locked
from pixora_service.studio import Studio
from test_studio import client, library_state, run  # noqa: F401  (client is a fixture)

pytest.importorskip("cryptography")

PROMPT = "a very private description"
SETTINGS = {"prompt": PROMPT, "seed": 7, "size": "square", "person_name": "Amit"}


def picture(colour=(200, 120, 40), size=(64, 48)):
    return Image.new("RGB", size, colour)


def png_of(image):
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


# ---- the file ---------------------------------------------------------------------


def test_a_locked_picture_opens_with_its_key_and_gives_back_everything():
    key = locked.new_key()
    image = picture()
    data = locked.seal(image, SETTINGS, key)
    png, settings = locked.unseal(data, key)
    assert settings == SETTINGS
    assert png == png_of(image)
    assert Image.open(io.BytesIO(png)).tobytes() == image.tobytes()


def test_nothing_about_the_picture_can_be_read_from_the_file():
    image = picture()
    data = locked.seal(image, SETTINGS, locked.new_key())
    assert data[:6] == locked.MAGIC and data[6] == locked.VERSION
    for secret in (PROMPT.encode(), b"Amit", png_of(image), b"square"):
        assert secret not in data
    with pytest.raises(UnidentifiedImageError):  # no image viewer can show it, whatever it is named
        Image.open(io.BytesIO(data))


def test_every_file_has_its_own_nonce_and_a_random_name():
    key = locked.new_key()
    a, b = locked.seal(picture(), SETTINGS, key), locked.seal(picture(), SETTINGS, key)
    assert a != b and a[7:19] != b[7:19]
    names = {locked.random_name() for _ in range(20)}
    assert len(names) == 20 and all(n.endswith(".pxlock") and len(n) == 16 + 7 for n in names)


def test_the_wrong_key_is_refused():
    data = locked.seal(picture(), SETTINGS, locked.new_key())
    with pytest.raises(locked.LockedError) as e:
        locked.unseal(data, locked.new_key())
    assert str(e.value) == locked.WRONG


@pytest.mark.parametrize("where", [0, 5, 6, 7, 18, 19, 40, -17, -1], ids=lambda w: f"byte {w}")
def test_a_file_changed_in_any_part_is_refused(where):
    key = locked.new_key()
    data = bytearray(locked.seal(picture(), SETTINGS, key))
    data[where] ^= 0x01  # the magic, version, nonce, ciphertext and tag are all covered
    with pytest.raises(locked.LockedError) as e:
        locked.unseal(bytes(data), key)
    assert str(e.value) == locked.WRONG


def test_a_cut_short_or_extended_file_or_another_kind_of_file_is_refused():
    key = locked.new_key()
    data = locked.seal(picture(), SETTINGS, key)
    for bad in (data[:-1], data + b"x", data[:19], b"", b"PXLOCK", png_of(picture())):
        with pytest.raises(locked.LockedError) as e:
            locked.unseal(bad, key)
        assert str(e.value) == locked.WRONG


def test_the_key_is_shown_in_groups_and_read_back_in_any_form():
    key = locked.new_key()
    text = locked.format_key(key)
    groups = text.split("-")
    assert len(key) == 32 and len(groups) == 13 and all(len(g) == 4 for g in groups)
    assert set(text.replace("-", "")) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    assert locked.parse_key(text) == key
    assert locked.parse_key(text.lower()) == key
    assert locked.parse_key(text.replace("-", " ")) == key
    assert locked.parse_key("  " + text.replace("-", "") + "\n") == key
    assert locked.new_key() != key


def test_a_zero_or_one_typed_for_the_letters_they_look_like_still_opens():
    key = bytes(32)  # its text is all A, so use a key whose text has an O and an I
    key = locked.parse_key("OOOO-IIII-" + "A" * 44)
    assert locked.parse_key("0000-1111-" + "A" * 44) == key


@pytest.mark.parametrize("text", ["", "ABCD", "A" * 51, "A" * 53, "!" * 52, "9" * 52, "é" * 52])
def test_something_that_is_not_a_key_is_said_so(text):
    with pytest.raises(locked.LockedError) as e:
        locked.parse_key(text)
    assert str(e.value) == locked.NOT_A_KEY


def test_without_the_encryption_library_the_user_is_told_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives.ciphers.aead", None)
    with pytest.raises(locked.LockedError) as e:
        locked.seal(picture(), SETTINGS, locked.new_key())
    assert e.value.status == 503 and "studio" in str(e.value)


# ---- through the service ----------------------------------------------------------


def held(client, count=1):
    job = run(client, operation="generate", prompt=PROMPT, incognito=True, count=count, seed=5)
    assert job["state"] == "done" and len(job["held"]) == count
    return job


def forbid(*_a, **_k):
    raise AssertionError("a locked picture must not be written anywhere")


def test_locking_and_opening_leave_nothing_behind_and_the_key_is_kept_nowhere(client, monkeypatch, caplog, capsys):
    job = held(client, count=2)
    before = library_state(client)
    scanner = client.app.state.scanner
    started = []
    monkeypatch.setattr(scanner, "start", lambda *a, **k: started.append(1))  # no face or scene scan
    monkeypatch.setattr(Studio, "save", forbid)  # nothing that makes a library photo
    monkeypatch.setattr(Studio, "keep", forbid)
    for name in ("NamedTemporaryFile", "SpooledTemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, forbid)  # no temporary file, encrypted or not
    caplog.set_level(logging.DEBUG)

    r = client.post(f"/api/studio/jobs/{job['id']}/held/0/lock")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store, no-cache, must-revalidate" and r.headers["pragma"] == "no-cache"
    body = r.json()
    data = base64.b64decode(body["data"])
    assert body["name"].endswith(".pxlock") and data[:6] == locked.MAGIC
    assert PROMPT.encode() not in data

    # No row, thumbnail or file anywhere, no scan, and the held picture waits for the save to succeed.
    assert library_state(client) == before and started == []
    assert client.get(f"/api/studio/jobs/{job['id']}/held/0").status_code == 200
    # A second lock is a different key and a different file.
    again = client.post(f"/api/studio/jobs/{job['id']}/held/0/lock").json()
    assert again["key"] != body["key"] and again["data"] != body["data"]

    # Once it is written, the picture is discarded exactly as a discard does.
    assert client.delete(f"/api/studio/jobs/{job['id']}/held/0").status_code == 200
    assert client.get(f"/api/studio/jobs/{job['id']}/held/0").status_code == 404

    opened = client.post("/api/studio/locked/open", json={"data": body["data"], "key": body["key"].lower()})
    assert opened.status_code == 200
    assert opened.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    shown = opened.json()
    assert Image.open(io.BytesIO(base64.b64decode(shown["image"]))).size == (1024, 1024)
    assert shown["settings"]["prompt"] == PROMPT and shown["settings"]["seed"] == 5
    assert library_state(client) == before and started == []

    out, err = capsys.readouterr()
    for text in (caplog.text, out, err):
        for form in (body["key"], body["key"].replace("-", ""), again["key"], again["key"].replace("-", "")):
            assert form not in text
        assert PROMPT not in text


def test_opening_with_a_wrong_key_or_a_damaged_file_says_so_and_shows_nothing(client):
    job = held(client)
    body = client.post(f"/api/studio/jobs/{job['id']}/held/0/lock").json()
    other = locked.format_key(locked.new_key())
    damaged = bytearray(base64.b64decode(body["data"]))
    damaged[-1] ^= 1

    def open_(data, key):
        return client.post("/api/studio/locked/open", json={"data": data, "key": key})

    for r in (
        open_(body["data"], other),
        open_(base64.b64encode(bytes(damaged)).decode(), body["key"]),
        open_("this is not base64!", body["key"]),
        open_(base64.b64encode(b"just some other file").decode(), body["key"]),
    ):
        assert r.status_code == 400 and r.json() == {"detail": locked.WRONG}
        assert "image" not in r.text
    r = open_(body["data"], "ABCD-EFGH")
    assert r.status_code == 400 and r.json() == {"detail": locked.NOT_A_KEY}


def test_only_a_picture_that_is_still_held_can_be_locked(client):
    run(client, operation="generate", prompt="a red bicycle")  # a normal one, saved to the library
    assert client.post("/api/studio/jobs/1/held/0/lock").status_code == 404
    assert client.post("/api/studio/jobs/999/held/0/lock").status_code == 404
    job = held(client)
    assert client.post(f"/api/studio/jobs/{job['id']}/held/7/lock").status_code == 404
    client.delete(f"/api/studio/jobs/{job['id']}/held")
    assert client.post(f"/api/studio/jobs/{job['id']}/held/0/lock").status_code == 404


def test_without_the_encryption_library_locking_says_what_to_install(client, monkeypatch):
    job = held(client)
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives.ciphers.aead", None)
    r = client.post(f"/api/studio/jobs/{job['id']}/held/0/lock")
    assert r.status_code == 503 and "studio" in r.json()["detail"]
    assert client.get(f"/api/studio/jobs/{job['id']}/held/0").status_code == 200  # nothing was lost

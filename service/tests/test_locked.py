"""Locked pictures: incognito pictures encrypted together so that only Pixora, given the key, can read them."""

import base64
import io
import json
import logging
import secrets
import struct
import sys
import tempfile

import pytest
from PIL import Image, UnidentifiedImageError

from pixora_service import locked
from pixora_service.studio import Studio
from test_studio import client, library_state, run  # noqa: F401  (client is a fixture)

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

PROMPT = "a very private description"


def picture(colour=(200, 120, 40), size=(64, 48)):
    return Image.new("RGB", size, colour)


def pictures(n=3):
    return [picture((40 * (i + 1), 90, 200 - 40 * i), (64 + 8 * i, 48)) for i in range(n)]


def png_of(image):
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def plaintext_of(data, key):
    """What is inside a locked file, for checking what is and is not kept there."""
    return AESGCM(key).decrypt(data[7:19], data[19:], data[:19])


# ---- the file ---------------------------------------------------------------------


def test_several_pictures_open_together_with_the_one_key_in_their_order():
    key = locked.new_key()
    images = pictures(3)
    data = locked.seal(images, key)
    opened = locked.unseal(data, key)
    assert opened == [png_of(i) for i in images]
    assert [Image.open(io.BytesIO(p)).size for p in opened] == [i.size for i in images]


def test_one_picture_is_a_locked_file_too():
    key = locked.new_key()
    assert locked.unseal(locked.seal([picture()], key), key) == [png_of(picture())]


def test_only_the_pictures_are_kept_no_description_and_no_seed():
    key = locked.new_key()
    images = pictures(2)
    plain = plaintext_of(locked.seal(images, key), key)
    # a count, then each picture with its length: nothing else, so nothing about how it was made
    assert plain == struct.pack(">I", 2) + b"".join(struct.pack(">I", len(png_of(i))) + png_of(i) for i in images)
    assert b"prompt" not in plain and b"seed" not in plain
    assert all(b"tEXt" not in png_of(i) and b"iTXt" not in png_of(i) for i in images)  # nor in the PNG


def test_nothing_about_the_pictures_can_be_read_from_the_file():
    images = pictures(3)
    data = locked.seal(images, locked.new_key())
    assert data[:6] == locked.MAGIC and data[6] == locked.VERSION == 2
    for image in images:
        assert png_of(image) not in data
    with pytest.raises(UnidentifiedImageError):  # no image viewer can show it, whatever it is named
        Image.open(io.BytesIO(data))


def test_every_file_has_its_own_nonce_and_a_random_name():
    key = locked.new_key()
    a, b = locked.seal(pictures(2), key), locked.seal(pictures(2), key)
    assert a != b and a[7:19] != b[7:19]
    names = {locked.random_name() for _ in range(20)}
    assert len(names) == 20 and all(n.endswith(".pxlock") and len(n) == 16 + 7 for n in names)


def test_the_wrong_key_is_refused():
    data = locked.seal(pictures(2), locked.new_key())
    with pytest.raises(locked.LockedError) as e:
        locked.unseal(data, locked.new_key())
    assert str(e.value) == locked.WRONG


@pytest.mark.parametrize("where", [0, 5, 6, 7, 18, 19, 40, -17, -1], ids=lambda w: f"byte {w}")
def test_a_file_changed_in_any_part_is_refused(where):
    key = locked.new_key()
    data = bytearray(locked.seal(pictures(2), key))
    data[where] ^= 0x01  # the magic, version, nonce, ciphertext and tag are all covered
    with pytest.raises(locked.LockedError) as e:
        locked.unseal(bytes(data), key)
    assert str(e.value) == locked.WRONG


def test_a_cut_short_or_extended_file_or_another_kind_of_file_is_refused():
    key = locked.new_key()
    data = locked.seal(pictures(2), key)
    for bad in (data[:-1], data + b"x", data[:19], b"", b"PXLOCK", png_of(picture())):
        with pytest.raises(locked.LockedError) as e:
            locked.unseal(bad, key)
        assert str(e.value) == locked.WRONG


def sealed_by_hand(plain, version=2):
    """A file with a header and encryption that are right around whatever plaintext is given."""
    key, nonce = locked.new_key(), secrets.token_bytes(12)
    header = locked.MAGIC + bytes([version]) + nonce
    return key, header + AESGCM(key).encrypt(nonce, plain, header)


@pytest.mark.parametrize(
    "plain",
    [
        struct.pack(">I", 2) + struct.pack(">I", 3) + b"abc",  # says two pictures, has one
        struct.pack(">I", 1) + struct.pack(">I", 99) + b"abc",  # a picture longer than the file
        struct.pack(">I", 1) + struct.pack(">I", 3) + b"abc" + b"extra",  # something after the last
        struct.pack(">I", 4_000_000_000),  # a count that cannot be true
        b"\x00",  # too short to say anything
        struct.pack(">I", 0),  # no pictures at all
    ],
    ids=["too few", "too long", "left over", "huge count", "tiny", "none"],
)
def test_a_file_that_is_right_to_decrypt_but_wrong_inside_is_refused(plain):
    key, data = sealed_by_hand(plain)
    with pytest.raises(locked.LockedError) as e:
        locked.unseal(data, key)
    assert str(e.value) == locked.WRONG


def test_the_first_form_of_the_feature_still_opens_and_its_settings_are_not_read():
    png = png_of(picture())
    settings = json.dumps({"prompt": PROMPT, "seed": 7}).encode()
    key, data = sealed_by_hand(struct.pack(">I", len(settings)) + settings + png, version=1)
    assert locked.unseal(data, key) == [png]


def test_a_file_holds_at_least_one_picture_and_at_most_a_hundred():
    key = locked.new_key()
    with pytest.raises(locked.LockedError, match="at least one"):
        locked.seal([], key)
    tiny = Image.new("RGB", (2, 2))
    assert len(locked.unseal(locked.seal([tiny] * locked.MAX_PICTURES, key), key)) == locked.MAX_PICTURES
    with pytest.raises(locked.LockedError, match="At most"):
        locked.seal([tiny] * (locked.MAX_PICTURES + 1), key)


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
        locked.seal(pictures(1), locked.new_key())
    assert e.value.status == 503 and "studio" in str(e.value)


# ---- through the service ----------------------------------------------------------


def held(client, count=1, prompt=PROMPT):
    job = run(client, operation="generate", prompt=prompt, incognito=True, count=count, seed=5)
    assert job["state"] == "done" and len(job["held"]) == count
    return job


def refs(*pairs):
    return {"pictures": [{"job_id": j, "index": i} for j, i in pairs]}


def forbid(*_a, **_k):
    raise AssertionError("a locked picture must not be written anywhere")


def test_locking_and_opening_leave_nothing_behind_and_the_key_is_kept_nowhere(client, monkeypatch, caplog, capsys):
    first, second = held(client, count=2), held(client, count=1, prompt="another private description")
    before = library_state(client)
    scanner = client.app.state.scanner
    started = []
    monkeypatch.setattr(scanner, "start", lambda *a, **k: started.append(1))  # no face or scene scan
    monkeypatch.setattr(Studio, "save", forbid)  # nothing that makes a library photo
    monkeypatch.setattr(Studio, "keep", forbid)
    for name in ("NamedTemporaryFile", "SpooledTemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, forbid)  # no temporary file, encrypted or not
    caplog.set_level(logging.DEBUG)

    # Three pictures from two jobs, under one key, in one file.
    wanted = refs((first["id"], 0), (first["id"], 1), (second["id"], 0))
    r = client.post("/api/studio/locked/lock", json=wanted)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store, no-cache, must-revalidate" and r.headers["pragma"] == "no-cache"
    body = r.json()
    data = base64.b64decode(body["data"])
    assert body["count"] == 3 and body["name"].endswith(".pxlock") and data[:6] == locked.MAGIC
    assert PROMPT.encode() not in data

    # No row, thumbnail or file anywhere, no scan, and the pictures wait for the save to succeed.
    assert library_state(client) == before and started == []
    for job, index in ((first, 0), (first, 1), (second, 0)):
        assert client.get(f"/api/studio/jobs/{job['id']}/held/{index}").status_code == 200
    # Locking again is another key and another file.
    again = client.post("/api/studio/locked/lock", json=wanted).json()
    assert again["key"] != body["key"] and again["data"] != body["data"]

    # Once the file is written, the pictures are discarded exactly as a discard does.
    for job, index in ((first, 0), (first, 1), (second, 0)):
        assert client.delete(f"/api/studio/jobs/{job['id']}/held/{index}").status_code == 200
        assert client.get(f"/api/studio/jobs/{job['id']}/held/{index}").status_code == 404

    opened = client.post("/api/studio/locked/open", json={"data": body["data"], "key": body["key"].lower()})
    assert opened.status_code == 200
    assert opened.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    shown = opened.json()
    assert list(shown) == ["images"]  # the pictures and nothing about them
    assert [Image.open(io.BytesIO(base64.b64decode(p))).size for p in shown["images"]] == [(1024, 1024)] * 3
    assert library_state(client) == before and started == []

    out, err = capsys.readouterr()
    for text in (caplog.text, out, err):
        for form in (body["key"], body["key"].replace("-", ""), again["key"], again["key"].replace("-", "")):
            assert form not in text
        assert PROMPT not in text and "another private description" not in text


def test_the_same_picture_asked_for_twice_is_locked_once(client):
    job = held(client)
    body = client.post("/api/studio/locked/lock", json=refs((job["id"], 0), (job["id"], 0))).json()
    assert body["count"] == 1


def test_opening_with_a_wrong_key_or_a_damaged_file_says_so_and_shows_nothing(client):
    job = held(client, count=2)
    body = client.post("/api/studio/locked/lock", json=refs((job["id"], 0), (job["id"], 1))).json()
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
        assert "no-store" in r.headers["cache-control"] and "images" not in r.text
    r = open_(body["data"], "ABCD-EFGH")
    assert r.status_code == 400 and r.json() == {"detail": locked.NOT_A_KEY}


def test_only_pictures_that_are_still_held_can_be_locked_and_none_is_locked_if_one_is_missing(client):
    run(client, operation="generate", prompt="a red bicycle")  # a normal one, saved to the library
    assert client.post("/api/studio/locked/lock", json=refs((1, 0))).status_code == 404
    assert client.post("/api/studio/locked/lock", json=refs((999, 0))).status_code == 404
    job = held(client)
    assert client.post("/api/studio/locked/lock", json=refs((job["id"], 0), (job["id"], 7))).status_code == 404
    assert client.get(f"/api/studio/jobs/{job['id']}/held/0").status_code == 200  # still there
    client.delete(f"/api/studio/jobs/{job['id']}/held")
    assert client.post("/api/studio/locked/lock", json=refs((job["id"], 0))).status_code == 404


def test_locking_nothing_is_refused(client):
    r = client.post("/api/studio/locked/lock", json=refs())
    assert r.status_code == 400 and "at least one" in r.json()["detail"]


def test_without_the_encryption_library_locking_says_what_to_install(client, monkeypatch):
    job = held(client)
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives.ciphers.aead", None)
    r = client.post("/api/studio/locked/lock", json=refs((job["id"], 0)))
    assert r.status_code == 503 and "studio" in r.json()["detail"]
    assert client.get(f"/api/studio/jobs/{job['id']}/held/0").status_code == 200  # nothing was lost

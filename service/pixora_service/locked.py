"""Locked pictures: a picture encrypted so that only Pixora, given the key, can read it.

A locked picture is one file, named at random and ending in `.pxlock`:

    offset  size  what
    0       6     the bytes "PXLOCK"
    6       1     the format version, 1
    7       12    a nonce, random for every file
    19      ...   the AES-256-GCM ciphertext, followed by its 16-byte tag

The 19 header bytes are the associated data of the encryption, so changing any of them
makes the file fail to open, as changing the ciphertext does. What is encrypted is the
picture as a PNG together with the settings it was made with, the description included:

    4 bytes   the length of the settings, big-endian
    ...       the settings, as UTF-8 JSON
    ...       the PNG

So nothing about the picture can be read without the key, except roughly how big the file is.

The key is 256 random bits made for each save. It is handed over once, as base32 in groups
of four, and kept nowhere: not on disk, not in the database, not in the log. A key that is
lost cannot be recovered, and neither can the picture. Because the key is random and
used for one file, there is no password to stretch, and a nonce can never be reused with it.

Nothing here touches the disk or the library. Locking and unlocking happen in memory.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import secrets
import struct

from PIL import Image

MAGIC = b"PXLOCK"
VERSION = 1
SUFFIX = ".pxlock"
KEY_BYTES = 32
NONCE_BYTES = 12
_HEADER = len(MAGIC) + 1 + NONCE_BYTES
_TAG = 16
_KEY_LETTERS = 52  # 256 bits in base32

WRONG = "Wrong key or damaged file"
NOT_A_KEY = "That is not a Pixora key. It is 52 letters and digits, in 13 groups of four."


class LockedError(Exception):
    """Something to tell the user. The message never includes the key or the picture."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _cipher(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise LockedError('Locked pictures need the studio packages: pip install -e ".[studio]"', 503) from e
    if len(key) != KEY_BYTES:
        raise LockedError(NOT_A_KEY)
    return AESGCM(key)


# ---- the key ----------------------------------------------------------------------


def new_key() -> bytes:
    return secrets.token_bytes(KEY_BYTES)


def format_key(key: bytes) -> str:
    """The key as the user is shown it: 13 groups of four letters and digits."""
    text = base64.b32encode(key).decode("ascii").rstrip("=")
    return "-".join(text[i : i + 4] for i in range(0, len(text), 4))


def parse_key(text: str) -> bytes:
    """The key from what the user typed or pasted: any case, with or without separators."""
    letters = "".join(ch for ch in text.upper() if ch.isalnum())
    # The alphabet has no 0 or 1, and they are what O and I get mistaken for.
    letters = letters.replace("0", "O").replace("1", "I")
    if len(letters) != _KEY_LETTERS:
        raise LockedError(NOT_A_KEY)
    try:
        key = base64.b32decode(letters + "====")
    except (binascii.Error, ValueError):
        raise LockedError(NOT_A_KEY) from None
    if len(key) != KEY_BYTES:
        raise LockedError(NOT_A_KEY)
    return key


def random_name() -> str:
    """A file name that says nothing about what is inside."""
    return secrets.token_hex(8) + SUFFIX


# ---- the file ---------------------------------------------------------------------


def seal(image: Image.Image, settings: dict, key: bytes) -> bytes:
    """The picture and its settings as the bytes of a locked file."""
    png = io.BytesIO()
    image.save(png, "PNG")
    meta = json.dumps(settings, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    plain = struct.pack(">I", len(meta)) + meta + png.getvalue()
    nonce = secrets.token_bytes(NONCE_BYTES)
    header = MAGIC + bytes([VERSION]) + nonce
    return header + _cipher(key).encrypt(nonce, plain, header)


def unseal(data: bytes, key: bytes) -> tuple[bytes, dict]:
    """The PNG and the settings in a locked file.

    A wrong key, a file that is not a locked picture, and a file that was changed in any way
    all give the same answer, which says no more than it has to.
    """
    if len(data) < _HEADER + _TAG + 4 or data[: len(MAGIC)] != MAGIC or data[len(MAGIC)] != VERSION:
        raise LockedError(WRONG)
    cipher = _cipher(key)
    from cryptography.exceptions import InvalidTag

    header, body = data[:_HEADER], data[_HEADER:]
    try:
        plain = cipher.decrypt(header[-NONCE_BYTES:], body, header)
    except InvalidTag:
        raise LockedError(WRONG) from None
    (size,) = struct.unpack(">I", plain[:4])
    if 4 + size > len(plain):
        raise LockedError(WRONG)
    try:
        settings = json.loads(plain[4 : 4 + size].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise LockedError(WRONG) from None
    return plain[4 + size :], settings

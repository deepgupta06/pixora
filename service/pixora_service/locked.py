"""Locked pictures: pictures encrypted so that only Pixora, given the key, can read them.

A locked file holds one or more pictures under one key. It is named at random and ends in
`.pxlock`:

    offset  size  what
    0       6     the bytes "PXLOCK"
    6       1     the format version, 2
    7       12    a nonce, random for every file
    19      ...   the AES-256-GCM ciphertext, followed by its 16-byte tag

The 19 header bytes are the associated data of the encryption, so changing any of them
makes the file fail to open, as changing the ciphertext does. What is encrypted is only the
pictures, each as a PNG, and nothing about how they were made: no description, no seed:

    4 bytes   how many pictures, big-endian
    then, for each picture:
    4 bytes   the length of the PNG, big-endian
    ...       the PNG

So nothing about the pictures can be read without the key, except roughly how big the file is.

Version 1, the first form of this feature, held one picture and its settings. Those files
still open; the settings in them are not read.

The key is 256 random bits made for each save. It is handed over once, as base32 in groups
of four, and kept nowhere: not on disk, not in the database, not in the log. A key that is
lost cannot be recovered, and neither can the pictures. Because the key is random and
used for one file, there is no password to stretch, and a nonce can never be reused with it.

Nothing here touches the disk or the library. Locking and unlocking happen in memory.
"""

from __future__ import annotations

import base64
import binascii
import io
import secrets
import struct

from PIL import Image

MAGIC = b"PXLOCK"
VERSION = 2
SUFFIX = ".pxlock"
KEY_BYTES = 32
NONCE_BYTES = 12
MAX_PICTURES = 100
_HEADER = len(MAGIC) + 1 + NONCE_BYTES
_TAG = 16
_KEY_LETTERS = 52  # 256 bits in base32

WRONG = "Wrong key or damaged file"
NOT_A_KEY = "That is not a Pixora key. It is 52 letters and digits, in 13 groups of four."


class LockedError(Exception):
    """Something to tell the user. The message never includes the key or a picture."""

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


def seal(images: list[Image.Image], key: bytes) -> bytes:
    """The pictures as the bytes of one locked file."""
    if not images:
        raise LockedError("Choose at least one picture to lock")
    if len(images) > MAX_PICTURES:
        raise LockedError(f"At most {MAX_PICTURES} pictures fit in one locked file")
    cipher = _cipher(key)
    parts = [struct.pack(">I", len(images))]
    for image in images:
        png = io.BytesIO()
        image.save(png, "PNG")
        parts += [struct.pack(">I", png.tell()), png.getvalue()]
    nonce = secrets.token_bytes(NONCE_BYTES)
    header = MAGIC + bytes([VERSION]) + nonce
    return header + cipher.encrypt(nonce, b"".join(parts), header)


def unseal(data: bytes, key: bytes) -> list[bytes]:
    """The pictures in a locked file, each as the bytes of a PNG.

    A wrong key, a file that is not a locked file, and a file that was changed in any way
    all give the same answer, which says no more than it has to.
    """
    if len(data) < _HEADER + _TAG + 4 or data[: len(MAGIC)] != MAGIC or data[len(MAGIC)] not in (1, 2):
        raise LockedError(WRONG)
    version = data[len(MAGIC)]
    cipher = _cipher(key)
    from cryptography.exceptions import InvalidTag

    header, body = data[:_HEADER], data[_HEADER:]
    try:
        plain = cipher.decrypt(header[-NONCE_BYTES:], body, header)
    except InvalidTag:
        raise LockedError(WRONG) from None
    (first,) = struct.unpack(">I", plain[:4])
    if version == 1:
        # One picture after its settings. The settings are not kept any more, so not read.
        if 4 + first > len(plain):
            raise LockedError(WRONG)
        return [plain[4 + first :]]
    if first > (len(plain) - 4) // 4:
        raise LockedError(WRONG)
    pictures, at = [], 4
    for _ in range(first):
        if at + 4 > len(plain):
            raise LockedError(WRONG)
        (size,) = struct.unpack(">I", plain[at : at + 4])
        at += 4
        if at + size > len(plain):
            raise LockedError(WRONG)
        pictures.append(plain[at : at + size])
        at += size
    if not pictures or at != len(plain):  # a locked file always holds one, and nothing else
        raise LockedError(WRONG)
    return pictures

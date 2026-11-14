"""Who a prompt is about, and which of that person's faces to show the model.

Nothing here guesses silently: a name that fits more than one person, or a
name nobody in the library has, is reported so the UI can ask.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from .db import Database
from .people import USER

MAX_REFERENCES = 5
# Reference faces further than this from the person's typical face are left
# out: they are the likeliest to be someone else filed under the name.
_OUTLIER_MARGIN = 0.15

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
# Capitalised words that are not names of people.
_NOT_NAMES = frozenset(
    """i a an the my create make generate draw paint show image picture photo portrait of in on at with and
    monday tuesday wednesday thursday friday saturday sunday january february march april may june july
    august september october november december christmas diwali holi eid india paris london tokyo
    new york mumbai delhi kolkata goa taj mahal eiffel tower mount everest""".split()
)


@dataclass
class Candidate:
    person_id: int
    name: str
    cover_face_id: int | None
    photo_count: int


@dataclass
class Mention:
    """A stretch of the prompt that names someone."""

    text: str
    start: int
    end: int
    # 'match': exactly one person fits. 'ambiguous': the user must choose.
    # 'unknown': it reads like a name, but nobody in the library has it.
    status: str
    candidates: list[Candidate] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def name_words(name: str) -> list[str]:
    """A person's name as words, without bracketed notes: "Deep(Me)" is "deep"."""
    return _words(re.sub(r"\([^)]*\)", " ", name)) or _words(name)


def find_mentions(prompt: str, people: list[Candidate]) -> list[Mention]:
    """Find the people a prompt names.

    A full name always beats a shorter one ("Amit Kumar" is not "Amit"). A
    first name on its own matches everyone who has it, so "Amit" is ambiguous
    while both Amit and Amit Kumar exist.
    """
    named = [(name_words(p.name), p) for p in people]
    named = [(words, p) for words, p in named if words]
    tokens = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(prompt)]
    lower = [t[0].lower() for t in tokens]

    mentions: list[Mention] = []
    i = 0
    while i < len(tokens):
        # The longest full name that starts here.
        exact = [(w, p) for w, p in named if lower[i : i + len(w)] == w]
        length = max((len(w) for w, _ in exact), default=0)
        exact_people = [p for w, p in exact if len(w) == length]
        span = length or 1
        # Longer names that begin with the same words.
        longer = [p for w, p in named if len(w) > span and w[:span] == lower[i : i + span]]
        candidates = exact_people + [p for p in longer if p not in exact_people]
        if candidates:
            text_start, text_end = tokens[i][1], tokens[i + span - 1][2]
            mentions.append(
                Mention(
                    text=prompt[text_start:text_end],
                    start=text_start,
                    end=text_end,
                    status="match" if len(candidates) == 1 else "ambiguous",
                    candidates=candidates,
                )
            )
            i += span
            continue
        word, start, end = tokens[i]
        sentence_start = i == 0 or prompt[tokens[i - 1][2] : start].strip()[:1] in (".", "!", "?", ":")
        if word[0].isupper() and not word.isupper() and not sentence_start and word.lower() not in _NOT_NAMES:
            previous = mentions[-1] if mentions else None
            if previous and previous.status == "unknown" and not prompt[previous.end : start].strip():
                # Two capitalised words in a row are one name.
                previous.text, previous.end = prompt[previous.start : end], end
            else:
                mentions.append(Mention(text=word, start=start, end=end, status="unknown"))
        i += 1
    return mentions


def describe_without_names(prompt: str, mentions: list[Mention], replacement: str = "a person") -> str:
    """The prompt as the image model should read it: names mean nothing to it."""
    out, last = [], 0
    for m in sorted(mentions, key=lambda m: m.start):
        if m.status == "unknown":
            continue
        out.append(prompt[last : m.start])
        out.append(replacement)
        last = m.end
    out.append(prompt[last:])
    return "".join(out)


def list_candidates(db: Database) -> list[Candidate]:
    rows = db.conn().execute(
        "SELECT pe.id, pe.name, pe.cover_face_id, count(DISTINCT f.photo_id) AS photos FROM people pe "
        "LEFT JOIN faces f ON f.person_id = pe.id WHERE pe.hidden = 0 GROUP BY pe.id ORDER BY photos DESC, pe.name"
    )
    return [Candidate(r["id"], r["name"], r["cover_face_id"], r["photos"]) for r in rows]


@dataclass
class Reference:
    face_id: int
    photo_id: int
    path: str | None  # the photo's file, when it is on disk
    box: tuple[float, float, float, float]  # fractions of the photo's width and height
    embedding: np.ndarray


def pick_references(rows: list[sqlite3.Row] | list[dict], k: int = MAX_REFERENCES) -> list[int]:
    """Choose the faces that best show what a person looks like.

    `rows` need id, embedding (float32 bytes), quality, size, turn, score and
    assignment_source. Returns indexes into `rows`, best first.

    Faces the user confirmed come before automatic matches, clear frontal
    faces before the rest, and larger before smaller. Faces unlike the
    person's typical face are dropped when enough others remain.
    """
    if not rows:
        return []
    x = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    trusted = [i for i, r in enumerate(rows) if r["assignment_source"] == USER] or list(range(len(rows)))
    centre = x[trusted].mean(axis=0)
    centre /= max(float(np.linalg.norm(centre)), 1e-9)
    typical = x @ centre
    cutoff = float(np.median(typical[trusted])) - _OUTLIER_MARGIN

    def rank(i: int) -> tuple:
        r = rows[i]
        return (
            typical[i] >= cutoff,
            r["assignment_source"] == USER,
            bool(r["quality"]),
            # Big and frontal: what an identity model reads best.
            float(r["size"]) * (1.0 - min(float(r["turn"]), 0.5)) * float(r["score"]),
        )

    order = sorted(range(len(rows)), key=rank, reverse=True)
    picked: list[int] = []
    seen_photos: set[int] = set()
    for i in order:
        # One face per photo, so the references show the person on different days.
        photo = rows[i]["photo_id"]
        if photo in seen_photos:
            continue
        seen_photos.add(photo)
        picked.append(i)
        if len(picked) == k:
            break
    return picked


def references_for(db: Database, person_id: int, k: int = MAX_REFERENCES) -> list[Reference]:
    rows = db.conn().execute(
        "SELECT f.id, f.photo_id, f.x1, f.y1, f.x2, f.y2, f.embedding, f.quality, f.size, f.turn, f.score, "
        "f.assignment_source, pf.path, pf.missing FROM faces f JOIN photos p ON p.id = f.photo_id "
        "LEFT JOIN photo_files pf ON pf.id = p.primary_file_id "
        # Pictures Pixora made of this person are never evidence of what they look like.
        "WHERE f.person_id = ? AND f.dismissed = 0 AND p.origin IS NULL",
        (person_id,),
    ).fetchall()
    return [
        Reference(
            face_id=rows[i]["id"],
            photo_id=rows[i]["photo_id"],
            path=rows[i]["path"] if rows[i]["path"] and not rows[i]["missing"] else None,
            box=(rows[i]["x1"], rows[i]["y1"], rows[i]["x2"], rows[i]["y2"]),
            embedding=np.frombuffer(rows[i]["embedding"], dtype=np.float32),
        )
        for i in pick_references(rows, k)
    ]

"""Person recognition: grouping unknown faces, matching known people, and the
short question flow that teaches Pixora a new person.

Rules from the plan:
  - A new person is learned from one name and at most three follow-up questions.
  - User answers always win over automatic results.
  - "Not this person" is stored as a constraint and never forgotten.
  - Two named people are never merged automatically.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .db import Database
from .faces import ENGINE_THRESHOLDS, Thresholds

MAX_FOLLOW_UPS = 3
MAX_PROTOTYPES = 12
MIN_CLUSTER = 3
SAMPLE_FACES = 6
# Grouping compares every pair of unknown faces, so it is capped. The largest
# faces are grouped first; the rest wait until some have been named.
MAX_CLUSTER_FACES = 8000

USER = "user"  # confirmed by the user, directly or by naming a group
AUTO = "auto"


def _matrix(rows: list[sqlite3.Row]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])


def _spread_sample(x: np.ndarray, k: int) -> list[int]:
    """Pick up to k rows that cover the variety in x, starting from the most typical."""
    if len(x) <= k:
        return list(range(len(x)))
    centre = x.mean(axis=0)
    centre /= max(float(np.linalg.norm(centre)), 1e-9)
    picked = [int(np.argmax(x @ centre))]
    while len(picked) < k:
        nearest = (x @ x[picked].T).max(axis=1)
        nearest[picked] = np.inf
        picked.append(int(np.argmin(nearest)))
    return picked


class People:
    def __init__(self, db: Database):
        self.db = db

    # ---- prototypes and matching ---------------------------------------------

    def _rebuild_prototypes(self, conn: sqlite3.Connection, person_id: int) -> None:
        """Prototypes come only from faces the user confirmed, never from guesses."""
        conn.execute("DELETE FROM person_prototypes WHERE person_id=?", (person_id,))
        rows = conn.execute(
            "SELECT id, embedding, engine_version, quality FROM faces "
            "WHERE person_id=? AND assignment_source=? ORDER BY quality DESC, size DESC",
            (person_id, USER),
        ).fetchall()
        by_engine: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for r in rows:
            by_engine[r["engine_version"]].append(r)
        cover = None
        for version, group in by_engine.items():
            # Learn from clear, frontal faces when there are any.
            good = [r for r in group if r["quality"]] or group
            x = _matrix(good)
            picked = _spread_sample(x, MAX_PROTOTYPES)
            conn.executemany(
                "INSERT INTO person_prototypes(person_id, embedding, engine_version) VALUES(?, ?, ?)",
                [(person_id, good[i]["embedding"], version) for i in picked],
            )
            cover = cover or good[picked[0]]["id"]
        current = conn.execute("SELECT cover_face_id FROM people WHERE id=?", (person_id,)).fetchone()
        valid = current and conn.execute(
            "SELECT 1 FROM faces WHERE id=? AND person_id=?", (current["cover_face_id"], person_id)
        ).fetchone()
        if not valid:
            conn.execute("UPDATE people SET cover_face_id=? WHERE id=?", (cover, person_id))

    def _match_unassigned(self, conn: sqlite3.Connection) -> None:
        """Assign or suggest a known person for every face that has none."""
        conn.execute("UPDATE faces SET suggested_person_id=NULL WHERE person_id IS NULL")
        versions = [r[0] for r in conn.execute("SELECT DISTINCT engine_version FROM person_prototypes")]
        for version in versions:
            thresholds = ENGINE_THRESHOLDS.get(version)
            if thresholds is None:
                continue
            protos = conn.execute(
                "SELECT person_id, embedding FROM person_prototypes WHERE engine_version=?", (version,)
            ).fetchall()
            faces = conn.execute(
                "SELECT id, embedding FROM faces WHERE person_id IS NULL AND engine_version=?", (version,)
            ).fetchall()
            if not protos or not faces:
                continue
            people = sorted({p["person_id"] for p in protos})
            sims = _matrix(faces) @ _matrix(protos).T
            owner = np.array([p["person_id"] for p in protos])
            # A person's score is the average similarity to their prototypes.
            scores = np.stack([sims[:, owner == pid].mean(axis=1) for pid in people], axis=1)

            column = {pid: i for i, pid in enumerate(people)}
            row = {f["id"]: i for i, f in enumerate(faces)}
            for c in conn.execute("SELECT face_id, person_id FROM face_constraints WHERE type='not'"):
                if c["face_id"] in row and c["person_id"] in column:
                    scores[row[c["face_id"]], column[c["person_id"]]] = -1.0

            best = scores.argmax(axis=1)
            best_score = scores.max(axis=1)
            assign, suggest = [], []
            for i, face in enumerate(faces):
                score = float(best_score[i])
                if score >= thresholds.auto:
                    assign.append((people[best[i]], AUTO, round(score, 4), face["id"]))
                elif score >= thresholds.suggest:
                    suggest.append((people[best[i]], round(score, 4), face["id"]))
            conn.executemany(
                "UPDATE faces SET person_id=?, assignment_source=?, confidence=?, cluster_id=NULL WHERE id=?",
                assign,
            )
            conn.executemany("UPDATE faces SET suggested_person_id=?, confidence=? WHERE id=?", suggest)

    def _recluster(self, conn: sqlite3.Connection) -> None:
        """Group the unknown faces that are good enough to learn a person from."""
        conn.execute("UPDATE faces SET cluster_id=NULL WHERE cluster_id IS NOT NULL")
        next_id = 1
        for (version,) in conn.execute("SELECT DISTINCT engine_version FROM faces").fetchall():
            thresholds: Thresholds | None = ENGINE_THRESHOLDS.get(version)
            if thresholds is None:
                continue
            faces = conn.execute(
                "SELECT id, embedding FROM faces WHERE person_id IS NULL AND quality=1 AND dismissed=0 "
                "AND engine_version=? ORDER BY size DESC LIMIT ?",
                (version, MAX_CLUSTER_FACES),
            ).fetchall()
            if len(faces) < MIN_CLUSTER:
                continue
            x = _matrix(faces).astype(np.float64)
            distance = np.clip(1 - x @ x.T, 0, 2)
            np.fill_diagonal(distance, 0)
            labels = fcluster(
                linkage(squareform(distance, checks=False), method="average"),
                t=1 - thresholds.cluster,
                criterion="distance",
            )
            sizes = Counter(labels)
            renumber: dict[int, int] = {}
            updates = []
            for face, label in zip(faces, labels):
                if sizes[label] < MIN_CLUSTER:
                    continue
                if label not in renumber:
                    renumber[label] = next_id
                    next_id += 1
                updates.append((renumber[label], face["id"]))
            conn.executemany("UPDATE faces SET cluster_id=? WHERE id=?", updates)

    def refresh(self) -> None:
        """Run after new faces arrive or after any answer from the user."""
        with self.db.write() as conn:
            self._match_unassigned(conn)
            self._recluster(conn)

    # ---- the question flow -----------------------------------------------------

    def _sample(self, conn: sqlite3.Connection, face_ids: list[int]) -> list[int]:
        marks = ",".join("?" * len(face_ids))
        rows = conn.execute(
            f"SELECT id FROM faces WHERE id IN ({marks}) ORDER BY quality DESC, size DESC LIMIT ?",
            (*face_ids, SAMPLE_FACES),
        )
        return [r["id"] for r in rows]

    def next_question(self) -> dict:
        conn = self.db.conn()
        # Follow-ups first: "Is this also NAME?", for the group of uncertain faces
        # that would teach the most.
        rows = conn.execute(
            "SELECT f.id, f.suggested_person_id AS person_id, f.cluster_id, p.name "
            "FROM faces f JOIN people p ON p.id = f.suggested_person_id "
            "WHERE f.person_id IS NULL AND f.dismissed = 0 AND p.questions_asked < ?",
            (MAX_FOLLOW_UPS,),
        ).fetchall()
        groups: dict[tuple, list[int]] = defaultdict(list)
        names: dict[int, str] = {}
        for r in rows:
            # Faces from one group are asked about together; loose faces one at a time.
            key = (r["person_id"], r["cluster_id"] if r["cluster_id"] is not None else -r["id"])
            groups[key].append(r["id"])
            names[r["person_id"]] = r["name"]
        if groups:
            (person_id, _), face_ids = max(groups.items(), key=lambda item: (len(item[1]), -item[0][0]))
            return {
                "type": "confirm",
                "person_id": person_id,
                "name": names[person_id],
                "face_ids": face_ids,
                "sample_face_ids": self._sample(conn, face_ids),
                "photo_count": self._photo_count(conn, face_ids),
            }

        # Otherwise "Who is this?", largest group first.
        largest = conn.execute(
            "SELECT cluster_id, count(*) AS n FROM faces WHERE cluster_id IS NOT NULL "
            "GROUP BY cluster_id ORDER BY n DESC, cluster_id LIMIT 1"
        ).fetchone()
        if largest is None:
            return {"type": "done"}
        face_ids = [r["id"] for r in conn.execute("SELECT id FROM faces WHERE cluster_id=?", (largest["cluster_id"],))]
        return {
            "type": "name",
            "face_ids": face_ids,
            "sample_face_ids": self._sample(conn, face_ids),
            "photo_count": self._photo_count(conn, face_ids),
        }

    def _photo_count(self, conn: sqlite3.Connection, face_ids: list[int]) -> int:
        marks = ",".join("?" * len(face_ids))
        return conn.execute(f"SELECT count(DISTINCT photo_id) FROM faces WHERE id IN ({marks})", face_ids).fetchone()[0]

    def pending_groups(self) -> int:
        return self.db.conn().execute("SELECT count(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL").fetchone()[0]

    # ---- answers and corrections -----------------------------------------------

    def _person_named(self, conn: sqlite3.Connection, name: str) -> int:
        name = " ".join(name.split())
        if not name:
            raise ValueError("A name is required")
        row = conn.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        if row:
            return row["id"]
        return conn.execute("INSERT INTO people(name) VALUES(?)", (name,)).lastrowid

    def _assign(self, conn: sqlite3.Connection, face_ids: list[int], person_id: int) -> set[int]:
        """Give faces to a person on the user's word. Returns every person affected."""
        marks = ",".join("?" * len(face_ids))
        affected = {r[0] for r in conn.execute(f"SELECT DISTINCT person_id FROM faces WHERE id IN ({marks}) AND person_id IS NOT NULL", face_ids)}
        conn.execute(
            f"UPDATE faces SET person_id=?, assignment_source=?, confidence=NULL, suggested_person_id=NULL, "
            f"cluster_id=NULL WHERE id IN ({marks})",
            (person_id, USER, *face_ids),
        )
        # Saying who it is overrules an earlier "not this person".
        conn.execute(f"DELETE FROM face_constraints WHERE person_id=? AND face_id IN ({marks})", (person_id, *face_ids))
        return affected | {person_id}

    def _finish(self, conn: sqlite3.Connection, people: set[int]) -> None:
        for person_id in people:
            self._rebuild_prototypes(conn, person_id)
        self._match_unassigned(conn)
        self._recluster(conn)

    def name_faces(self, face_ids: list[int], name: str) -> int:
        """Answer "Who is this?". An existing name adds the faces to that person."""
        with self.db.write() as conn:
            person_id = self._person_named(conn, name)
            self._finish(conn, self._assign(conn, face_ids, person_id))
            return person_id

    def confirm(self, person_id: int, face_ids: list[int], yes: bool) -> None:
        """Answer "Is this also NAME?"."""
        with self.db.write() as conn:
            conn.execute("UPDATE people SET questions_asked = questions_asked + 1 WHERE id=?", (person_id,))
            if yes:
                people = self._assign(conn, face_ids, person_id)
            else:
                conn.executemany(
                    "INSERT OR IGNORE INTO face_constraints(face_id, person_id, type) VALUES(?, ?, 'not')",
                    [(fid, person_id) for fid in face_ids],
                )
                people = set()
            self._finish(conn, people)

    def dismiss(self, face_ids: list[int]) -> None:
        """Skip a group: stop asking who these faces are."""
        with self.db.write() as conn:
            marks = ",".join("?" * len(face_ids))
            conn.execute(f"UPDATE faces SET dismissed=1, cluster_id=NULL WHERE id IN ({marks})", face_ids)
            self._recluster(conn)

    def reject(self, face_id: int) -> None:
        """"Not this person": remember it and look for someone else."""
        with self.db.write() as conn:
            row = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
            if row is None or row["person_id"] is None:
                return
            conn.execute(
                "INSERT OR IGNORE INTO face_constraints(face_id, person_id, type) VALUES(?, ?, 'not')",
                (face_id, row["person_id"]),
            )
            conn.execute(
                "UPDATE faces SET person_id=NULL, assignment_source=NULL, confidence=NULL WHERE id=?", (face_id,)
            )
            self._finish(conn, {row["person_id"]})

    def rename(self, person_id: int, name: str) -> int:
        """Rename a person. Renaming to an existing name merges into that person."""
        with self.db.write() as conn:
            name = " ".join(name.split())
            if not name:
                raise ValueError("A name is required")
            other = conn.execute(
                "SELECT id FROM people WHERE name = ? COLLATE NOCASE AND id != ?", (name, person_id)
            ).fetchone()
            if other is None:
                conn.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))
                return person_id
            # A merge only ever happens here, on the user's explicit request.
            target = other["id"]
            conn.execute("UPDATE faces SET person_id=? WHERE person_id=?", (target, person_id))
            conn.execute("UPDATE faces SET suggested_person_id=NULL WHERE suggested_person_id=?", (person_id,))
            conn.execute("UPDATE OR IGNORE face_constraints SET person_id=? WHERE person_id=?", (target, person_id))
            conn.execute("DELETE FROM face_constraints WHERE person_id=?", (person_id,))
            conn.execute("DELETE FROM person_prototypes WHERE person_id=?", (person_id,))
            conn.execute("DELETE FROM people WHERE id=?", (person_id,))
            self._finish(conn, {target})
            return target

    def set_hidden(self, person_id: int, hidden: bool) -> None:
        with self.db.write() as conn:
            conn.execute("UPDATE people SET hidden=? WHERE id=?", (int(hidden), person_id))

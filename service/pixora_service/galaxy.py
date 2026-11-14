"""Galaxy layout: where each face sits in 3D in People mode.

Named people form the core: their centres come from how alike their average
faces are, so similar-looking people sit near each other. Groups of unnamed
faces orbit on a shell around the core, and faces that belong to no group
form a sparse outer halo.
"""

from __future__ import annotations

import numpy as np

POINT_SIZE = 0.4  # world units; the interface draws points at this size
_GAP = 1.2  # clear space between neighbouring clusters
_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))


def cluster_radius(count: int) -> float:
    """Radius of a ball that holds `count` points without crowding."""
    return 0.2 + POINT_SIZE * float(np.cbrt(count))


def _sphere(n: int) -> np.ndarray:
    """n evenly spread unit directions."""
    i = np.arange(n) + 0.5
    y = 1 - 2 * i / max(n, 1)
    ring = np.sqrt(np.clip(1 - y * y, 0, 1))
    return np.stack([ring * np.cos(_GOLDEN * i), y, ring * np.sin(_GOLDEN * i)], axis=1)


def _project(x: np.ndarray) -> np.ndarray:
    """The three strongest directions of variation in x, as coordinates."""
    out = np.zeros((len(x), 3))
    if len(x) < 2:
        return out
    centred = x - x.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    k = min(3, len(s))
    out[:, :k] = u[:, :k] * s[:k]
    return out


def _fill_ball(embeddings: np.ndarray, radius: float) -> np.ndarray:
    """Local positions: direction from the faces' differences, spread evenly through a ball."""
    n = len(embeddings)
    if n == 1:
        return np.zeros((1, 3))
    coords = _project(embeddings.astype(np.float64))
    norms = np.linalg.norm(coords, axis=1)
    fallback = _sphere(n)
    flat = norms < 1e-9
    directions = np.where(flat[:, None], fallback, coords / np.where(flat, 1, norms)[:, None])
    # The most typical face sits in the middle; the rank keeps the ball evenly filled.
    rank = np.empty(n)
    rank[np.argsort(norms, kind="stable")] = np.arange(n)
    return directions * (radius * np.cbrt((rank + 0.5) / n))[:, None]


def _separate(centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Push overlapping clusters apart, keeping their arrangement."""
    centres = centres.copy()
    n = len(centres)
    fallback = _sphere(n)
    for _ in range(300):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = centres[j] - centres[i]
                dist = float(np.linalg.norm(delta))
                need = radii[i] + radii[j] + _GAP
                if dist >= need:
                    continue
                direction = delta / dist if dist > 1e-9 else fallback[j] - fallback[i]
                direction /= np.linalg.norm(direction)
                push = (need - dist) / 2 + 1e-3
                centres[i] -= direction * push
                centres[j] += direction * push
                moved = True
        if not moved:
            break
    return centres


def people_layout(
    embeddings: np.ndarray, person_ids: list[int | None], cluster_ids: list[int | None]
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, float]]]:
    """Positions for every face, and each person's (centre, radius)."""
    n = len(person_ids)
    positions = np.zeros((n, 3))
    named: dict[int, list[int]] = {}
    unnamed: dict[int, list[int]] = {}
    loose: list[int] = []
    for i, (person, cluster) in enumerate(zip(person_ids, cluster_ids)):
        if person is not None:
            named.setdefault(person, []).append(i)
        elif cluster is not None:
            unnamed.setdefault(cluster, []).append(i)
        else:
            loose.append(i)

    people: dict[int, tuple[np.ndarray, float]] = {}
    core = 0.0
    if named:
        ids = sorted(named)
        radii = np.array([cluster_radius(len(named[p])) for p in ids])
        centroids = np.stack([embeddings[named[p]].mean(axis=0) for p in ids])
        centres = _project(centroids.astype(np.float64))
        spread = np.linalg.norm(centres, axis=1).max()
        if spread > 1e-9:
            # Start a little tighter than the clusters need; separation then opens it up.
            centres *= radii.sum() / spread / 4
        centres = _separate(centres, radii)
        centres -= centres.mean(axis=0)
        for p, centre, radius in zip(ids, centres, radii):
            positions[named[p]] = centre + _fill_ball(embeddings[named[p]], radius)
            people[p] = (centre, float(radius))
        core = float((np.linalg.norm(centres, axis=1) + radii).max())

    shell = core
    if unnamed:
        ids = sorted(unnamed, key=lambda c: (-len(unnamed[c]), c))
        radii = np.array([cluster_radius(len(unnamed[c])) for c in ids])
        # Far enough out that the groups fit side by side on the shell.
        room = float(np.sqrt(((radii + _GAP) ** 2).sum() / 2))
        shell = max(core + radii.max() + _GAP, room)
        for c, direction, radius in zip(ids, _sphere(len(ids)), radii):
            positions[unnamed[c]] = direction * shell + _fill_ball(embeddings[unnamed[c]], radius)
        shell += float(radii.max())

    if loose:
        halo = max(shell + 2 * _GAP, POINT_SIZE * float(np.sqrt(len(loose))))
        positions[loose] = _sphere(len(loose)) * halo

    return positions, people

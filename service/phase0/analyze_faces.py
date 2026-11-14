"""Phase 0, step 2: compare the two face engines without ground-truth labels.

    python -m phase0.analyze_faces --data-dir .data-real

Label-free evidence used:
  - Two faces in the same photo are different people, which gives real
    impostor pairs for each engine.
  - Each engine's clusters act as reference labels for the *other* engine, so
    no engine is graded against its own output.

Writes <data-dir>/phase0/faces_analysis.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

ENGINES = ("insightface", "opencv-zoo")
SWEEP = (0.35, 0.40, 0.45, 0.50, 0.55)
# Similarity at which clusters stop merging, chosen from the sweep.
CLUSTER_THRESHOLD = {"insightface": 0.40, "opencv-zoo": 0.45}
# Faces used to learn people: not tiny, not strongly turned (plan, person recognition step 1).
MIN_SIZE = 40
MAX_TURN = 0.35


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def in_cluster(face: dict) -> bool:
    return face["cluster"] is not None and face["cluster"] >= 0


def percentiles(values: np.ndarray, points) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {f"p{p:g}": round(float(np.percentile(values, p)), 3) for p in points}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    args = parser.parse_args()
    out = args.data_dir / "phase0"

    data = json.loads((out / "faces.json").read_text(encoding="utf-8"))
    npz = np.load(out / "faces.npz")
    faces = {e: [f for f in data["faces"] if f["engine"] == e] for e in ENGINES}
    emb = {e: npz[e] for e in ENGINES}
    for e in ENGINES:
        for f in faces[e]:
            f["good"] = f["size"] >= MIN_SIZE and f["turn"] <= MAX_TURN
            f["cluster"] = None  # None: not clustered (filtered out); -1: noise

    # ---- cluster each engine's good faces -----------------------------------
    result: dict = {"filters": {"min_size": MIN_SIZE, "max_turn": MAX_TURN}, "engines": {}}
    for e in ENGINES:
        good = [f for f in faces[e] if f["good"]]
        x = emb[e][[f["index"] for f in good]].astype(np.float64)
        sims = x @ x.T
        distance = np.clip(1 - sims, 0, 2)
        photo = np.array([f["photo_id"] for f in good])

        # The plan names HDBSCAN. It is kept here for the record: on this library
        # it breaks one dense, frequently photographed person into many pieces.
        hdbscan = HDBSCAN(
            min_cluster_size=args.min_cluster_size, min_samples=2, metric="precomputed", copy=True
        ).fit_predict(distance)

        # Average-linkage merging up to a similarity threshold, swept to show
        # how sensitive each engine is to the threshold.
        sweep = []
        for threshold in SWEEP:
            raw = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1 - threshold
            ).fit_predict(distance)
            counts = Counter(int(v) for v in raw)
            same = (raw[:, None] == raw[None, :]) & (photo[:, None] == photo[None, :])
            sweep.append(
                {
                    "threshold": threshold,
                    "people": sum(1 for v in counts.values() if v >= args.min_cluster_size),
                    "faces_ungrouped": sum(v for v in counts.values() if v < args.min_cluster_size),
                    "largest": sorted(counts.values(), reverse=True)[:5],
                    # Two faces from one photo in the same cluster is a certain mistake.
                    "same_photo_conflicts": int((same.sum() - len(raw)) // 2),
                }
            )
            if threshold == CLUSTER_THRESHOLD[e]:
                # Groups too small to be a person are reported as noise (-1).
                labels = np.array([int(v) if counts[int(v)] >= args.min_cluster_size else -1 for v in raw])
        for f, label in zip(good, labels):
            f["cluster"] = int(label)
        sizes = sorted(Counter(int(v) for v in labels if v >= 0).values(), reverse=True)
        hdbscan_sizes = sorted(Counter(int(v) for v in hdbscan if v >= 0).values(), reverse=True)

        # Same-photo pairs are guaranteed impostors.
        upper = np.triu(np.ones_like(sims, dtype=bool), 1)
        impostor = sims[upper & (photo[:, None] == photo[None, :])]
        result["engines"][e] = {
            "version": data["summary"]["engines"][e]["version"],
            "ms_per_photo": data["summary"]["engines"][e]["ms_per_photo"],
            "faces": len(faces[e]),
            "good_faces": len(good),
            "clusters": len(sizes),
            "cluster_sizes": sizes,
            "clustered": int((labels >= 0).sum()),
            "noise": int((labels < 0).sum()),
            "cluster_threshold": CLUSTER_THRESHOLD[e],
            "threshold_sweep": sweep,
            "hdbscan": {
                "clusters": len(hdbscan_sizes),
                "largest": hdbscan_sizes[:5],
                "noise": int((hdbscan < 0).sum()),
            },
            "same_photo_impostor_pairs": int(len(impostor)),
            "same_photo_impostor_similarity": percentiles(impostor, (50, 90, 99, 99.9, 100)),
        }

    # ---- match detections between engines -----------------------------------
    by_photo = {e: defaultdict(list) for e in ENGINES}
    for e in ENGINES:
        for f in faces[e]:
            by_photo[e][f["photo_id"]].append(f)
    a, b = ENGINES
    pairs: list[tuple[dict, dict]] = []
    for photo_id in set(by_photo[a]) | set(by_photo[b]):
        candidates = sorted(
            ((iou(fa["bbox"], fb["bbox"]), i, j) for i, fa in enumerate(by_photo[a][photo_id]) for j, fb in enumerate(by_photo[b][photo_id])),
            reverse=True,
        )
        used_a, used_b = set(), set()
        for score, i, j in candidates:
            if score < 0.4 or i in used_a or j in used_b:
                continue
            used_a.add(i), used_b.add(j)
            fa, fb = by_photo[a][photo_id][i], by_photo[b][photo_id][j]
            fa["match"], fb["match"] = fb["index"], fa["index"]
            pairs.append((fa, fb))
    result["detection"] = {
        "found_by_both": len(pairs),
        f"only_{a}": len(faces[a]) - len(pairs),
        f"only_{b}": len(faces[b]) - len(pairs),
        f"only_{a}_good": sum(1 for f in faces[a] if "match" not in f and f["good"]),
        f"only_{b}_good": sum(1 for f in faces[b] if "match" not in f and f["good"]),
    }

    # ---- agreement between the two clusterings ------------------------------
    both = [(fa, fb) for fa, fb in pairs if in_cluster(fa) and in_cluster(fb)]
    result["cluster_agreement"] = {
        "faces_clustered_by_both": len(both),
        "adjusted_rand_index": round(float(adjusted_rand_score([p[0]["cluster"] for p in both], [p[1]["cluster"] for p in both])), 3) if both else None,
    }

    # ---- each engine's similarities, judged by the other engine's clusters ---
    for me, other, pick in ((a, b, lambda p: (p[0], p[1])), (b, a, lambda p: (p[1], p[0]))):
        judged = [pick(p) for p in pairs]
        judged = [(mine, ref) for mine, ref in judged if mine["good"] and in_cluster(ref)]
        x = emb[me][[mine["index"] for mine, _ in judged]].astype(np.float64)
        sims = x @ x.T
        ref = np.array([r["cluster"] for _, r in judged])
        photo = np.array([m["photo_id"] for m, _ in judged])
        upper = np.triu(np.ones_like(sims, dtype=bool), 1) & (photo[:, None] != photo[None, :])
        genuine = sims[upper & (ref[:, None] == ref[None, :])]
        impostor = sims[upper & (ref[:, None] != ref[None, :])]
        info = result["engines"][me]
        info["judged_by_other_engine"] = {
            "genuine_pairs": int(len(genuine)),
            "impostor_pairs": int(len(impostor)),
            "genuine_similarity": percentiles(genuine, (1, 5, 25, 50)),
            "impostor_similarity": percentiles(impostor, (50, 99, 99.9, 100)),
        }
        # How many genuine pairs clear thresholds set by false-accept targets.
        table = []
        for target in (1.0, 0.1, 0.01):
            if len(impostor) == 0 or len(genuine) == 0:
                break
            threshold = float(np.percentile(impostor, 100 - target))
            table.append({"false_accept_pct": target, "threshold": round(threshold, 3), "genuine_accepted_pct": round(float((genuine >= threshold).mean() * 100), 1)})
        info["judged_by_other_engine"]["operating_points"] = table
        # The firmest evidence: thresholds from same-photo pairs, which are certain impostors.
        info["same_photo_operating_points"] = [
            {
                "threshold_from": name,
                "threshold": threshold,
                "genuine_accepted_pct": round(float((genuine >= threshold).mean() * 100), 1),
            }
            for name, threshold in info["same_photo_impostor_similarity"].items()
            if name in ("p99", "p100") and len(genuine)
        ]

    (out / "faces_analysis.json").write_text(json.dumps({"result": result, "faces": faces}), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

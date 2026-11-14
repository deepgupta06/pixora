"""Phase 0, step 4: build a local review page from the analysis results.

    python -m phase0.report --data-dir .data-real

Writes <data-dir>/phase0/review.html. It shows personal photos, so it stays on
this computer next to the data it was built from.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

# Below this, a label is treated as "does not apply" on the review page.
SCENE_FLOOR = 0.002

CSS = """
body{font:14px 'Segoe UI',system-ui,sans-serif;background:#0f1114;color:#e6e8ec;margin:0;padding:24px;max-width:1500px}
h1{font-size:22px} h2{font-size:18px;margin-top:36px;border-top:1px solid #2c313a;padding-top:20px} h3{font-size:14px;margin:18px 0 6px}
p,li{color:#b6bcc8;line-height:1.5;max-width:900px} .cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.group{margin-bottom:14px} .group b{color:#e6e8ec} .faces{display:flex;flex-wrap:wrap;gap:2px;margin-top:4px}
.faces img{width:56px;height:56px;border-radius:3px;object-fit:cover} .photos img{height:120px;border-radius:3px}
.photos figure{margin:0;position:relative} .photos figcaption{position:absolute;left:4px;bottom:4px;background:#000a;padding:0 5px;border-radius:3px;font-size:12px}
table{border-collapse:collapse;margin:8px 0} td,th{border:1px solid #2c313a;padding:4px 10px;text-align:right} th:first-child,td:first-child{text-align:left}
.muted{color:#8b93a1}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
"""


def face_groups(faces: list[dict]) -> tuple[list[list[dict]], list[dict], list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    ungrouped, filtered = [], []
    for f in faces:
        if f["cluster"] is None:
            filtered.append(f)
        elif f["cluster"] < 0:
            ungrouped.append(f)
        else:
            groups[f["cluster"]].append(f)
    return sorted(groups.values(), key=len, reverse=True), ungrouped, filtered


def face_strip(faces: list[dict]) -> str:
    return '<div class="faces">' + "".join(f'<img loading="lazy" src="{f["crop"]}" alt="">' for f in faces) + "</div>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.data_dir / "phase0"
    analysis = json.loads((out / "faces_analysis.json").read_text(encoding="utf-8"))
    scenes = json.loads((out / "scenes.json").read_text(encoding="utf-8"))
    result = analysis["result"]

    parts = [f"<!doctype html><meta charset='utf-8'><title>Pixora Phase 0 review</title><style>{CSS}</style>"]
    parts.append("<h1>Pixora Phase 0 review</h1>")
    parts.append(
        "<p>Each row below should be one person. Look for rows that mix two people, and for one person "
        "spread over several rows. The two columns are the two face engines on the same photos.</p>"
    )

    parts.append("<table><tr><th></th>" + "".join(f"<th>{html.escape(e)}</th>" for e in result["engines"]) + "</tr>")
    rows = [
        ("Faces detected", "faces"),
        ("Faces good enough to learn from", "good_faces"),
        ("People found (3+ faces)", "clusters"),
        ("Faces left ungrouped", "noise"),
        ("Time per photo (ms)", "ms_per_photo"),
    ]
    for label, key in rows:
        parts.append(f"<tr><td>{label}</td>" + "".join(f"<td>{info[key]}</td>" for info in result["engines"].values()) + "</tr>")
    parts.append("</table>")

    parts.append("<h2>People</h2><div class='cols'>")
    for engine, faces in analysis["faces"].items():
        groups, ungrouped, _ = face_groups(faces)
        parts.append(f"<div><h3>{html.escape(engine)}</h3>")
        for n, group in enumerate(groups, 1):
            photos = len({f["photo_id"] for f in group})
            parts.append(f"<div class='group'><b>Person {n}</b> <span class='muted'>{len(group)} faces in {photos} photos</span>{face_strip(group)}</div>")
        parts.append(f"<div class='group'><b>Ungrouped</b> <span class='muted'>{len(ungrouped)} faces seen fewer than 3 times</span>{face_strip(ungrouped)}</div>")
        parts.append("</div>")
    parts.append("</div>")

    parts.append("<h2>Faces only one engine found</h2><div class='cols'>")
    for engine, faces in analysis["faces"].items():
        only = sorted((f for f in faces if "match" not in f), key=lambda f: -f["size"])
        parts.append(f"<div><h3>{html.escape(engine)} only: {len(only)}</h3>{face_strip(only)}</div>")
    parts.append("</div>")

    parts.append("<h2>Scenes and occasions</h2>")
    parts.append(
        f"<p>Photos are listed under every label scoring at least {SCENE_FLOOR}, best first, with the score shown. "
        "Scores are small by nature for this model; the order matters more than the number.</p>"
    )
    labels = [label for group in scenes["labels"].values() for label in group]
    for i, label in enumerate(labels):
        hits = sorted((p for p in scenes["photos"] if p["probs"][i] >= SCENE_FLOOR), key=lambda p: -p["probs"][i])
        parts.append(f"<h3>{html.escape(label)} <span class='muted'>{len(hits)} photos</span></h3><div class='faces photos'>")
        for p in hits[:40]:
            h = p["hash"]
            parts.append(f"<figure><img loading='lazy' src='../thumbs/256/{h[:2]}/{h}.webp' alt=''><figcaption>{p['probs'][i]:.3f}</figcaption></figure>")
        parts.append("</div>")
    unlabelled = sum(1 for p in scenes["photos"] if max(p["probs"]) < SCENE_FLOOR)
    parts.append(f"<p>{unlabelled} of {len(scenes['photos'])} photos have no label at or above the floor.</p>")

    (out / "review.html").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out / "review.html", "| unlabelled photos:", unlabelled)
    for i, label in enumerate(labels):
        print(f"{label:11} {sum(1 for p in scenes['photos'] if p['probs'][i] >= SCENE_FLOOR):3}")


if __name__ == "__main__":
    main()

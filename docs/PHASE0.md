# Phase 0 results: model test

First run, on one real folder. The numbers below come from
`service/phase0/` and can be reproduced with the commands at the end.

## Test set

- 251 JPEG files, 163 unique photos after exact-duplicate removal
- Taken 2010 to 2014, mostly one compact camera plus several phones
- No GPS data
- Mostly group photos of the same circle of young adults

This is smaller and narrower than the 500 to 1,000 photos the plan asks for.
There are no children, no older people, and no long age span, so the face
results are a first reading and need repeating on a more varied folder.

## Face engines

| | InsightFace `buffalo_l` | YuNet + SFace |
|---|---|---|
| Faces detected | 727 | 607 |
| Faces good enough to learn from | 451 | 444 |
| People found (3 or more faces) | 23 | 24 |
| Faces left ungrouped | 28 | 37 |
| Same-photo conflicts in clusters | 0 | 0 |
| Time per photo | 50 ms (GPU) | 36 ms (CPU) |

"Good enough to learn from" means at least 40 px across at detection size and
not strongly turned. Timing excludes the 57 ms it takes to decode each photo.

**Detection.** 601 faces were found by both engines. InsightFace found 126
more, but only 6 of them pass the quality filter; the rest are tiny or turned
faces that would not be used to learn a person anyway. YuNet found 6 that
InsightFace missed.

**Grouping.** On the 404 faces both engines grouped, the two sets of people
agree almost exactly (adjusted Rand index 0.996).

**Separation.** There are no labels, so two label-free checks were used: two
faces in one photo are certainly different people, and each engine's groups
were used as the reference for the other engine.

| | InsightFace | SFace |
|---|---|---|
| Highest similarity between two people in one photo | 0.346 | 0.496 |
| Same-person pairs above that value | 99.3% | 82.7% |
| Threshold for 0.1% false accepts across photos | 0.351 | 0.474 |
| Same-person pairs accepted at that threshold | 99.2% | 86.4% |
| Thresholds giving identical people | 0.35 to 0.45 | 0.40 to 0.45 |

InsightFace separates people with a wide margin. SFace reaches the same
grouping but with a narrower one: below 0.40 it starts merging different
people (5 same-photo conflicts at 0.35) and above 0.45 it starts splitting one
person into several.

### Finding: HDBSCAN is the wrong default

The plan names HDBSCAN for grouping. With both engines it broke the most
photographed person into pieces: for InsightFace, 100 faces of one person
became seven small clusters plus 40 faces marked as noise, although every pair
of those faces has a similarity of 0.45 or more. Average-linkage merging up to
a similarity threshold gave clean groups with both engines and is what the
figures above use.

### Decision

- **Default engine: InsightFace `buffalo_l`**, chosen by the project owner for
  its wider separation margin. YuNet + SFace found the same people on this set
  and stays available behind the `FaceEngine` interface.
- **Licence consequence.** The pretrained `buffalo_l` models are licensed for
  non-commercial research use only. Pixora must not bundle them: they are
  downloaded on first run with the licence shown, as the plan's model policy
  already requires. Anyone using Pixora commercially needs the YuNet + SFace
  engine or their own licence from InsightFace. Confirm the current terms
  before release.
- **Grouping: average-linkage clustering**, not HDBSCAN.
- **Starting thresholds:**

| | InsightFace (default) | SFace |
|---|---|---|
| Group unknown faces up to | 0.40 | 0.45 |
| Assign automatically above | 0.45 | 0.50 |
| Queue as a suggestion | 0.35 to 0.45 | 0.40 to 0.50 |
| Keep as unknown below | 0.35 | 0.40 |

The thresholds come from one small folder and need repeating on a larger, more
varied one. Relatives who look alike would test both engines harder than this
set does.

## Scene and occasion labels

Model: SigLIP 2 ViT-B/16 at 224 px, 8 ms per photo on the GPU, 768-number
embedding per photo.

- **The model works but the scores are tiny.** The best score in the whole set
  was 0.028. Taking the top label for every photo gives nonsense for most
  photos ("snow" for people standing by a white wall), so a label needs a
  minimum score.
- **With a floor of 0.002, 133 of 163 photos get no label.** That is the right
  answer for most of them: they are friends posing, which the label set does
  not describe.
- **Places and nature labels were right when they fired** (mall, lake, beach).
- **Occasion labels were weak on single photos.** "Party", "birthday" and
  "festival" fired on ordinary group photos. This supports the plan's choice
  to decide occasions per event, not per photo.
- **The label set misses what this library is mostly made of.** Labels such as
  group photo, portrait, hostel or classroom, and water park would cover far
  more photos than the current nineteen.

The floor of 0.002 is a guess from this one folder and should be set per label
on more photos. The model's logits equal 112.67 times the cosine similarity
minus 16.77, so the product can score stored image embeddings against cached
label embeddings without running the text model again.

## Other findings

- InsightFace's detector fails on DirectML with a variable input size; fixing
  the input size at session creation solves it.
- The scan pipeline handled all 251 real files with no errors.

## Reproducing

```bash
cd service
.venv/Scripts/python -m pip install -e ".[phase0]"
.venv/Scripts/python scripts/scan_folder.py <photo folder> --data-dir .data-real
.venv/Scripts/python -m phase0.run_faces --data-dir .data-real --models .models
.venv/Scripts/python -m phase0.analyze_faces --data-dir .data-real
.venv/Scripts/python -m phase0.run_scenes --data-dir .data-real --models .models
.venv/Scripts/python -m phase0.report --data-dir .data-real
```

Models go in `service/.models` (not committed): `buffalo_l/` from the
InsightFace releases, `opencv_zoo/` with the YuNet and SFace files, and
`siglip2/` from `onnx-community/siglip2-base-patch16-224-ONNX`. The review page
is written to `<data dir>/phase0/review.html` and stays local because it shows
the photos.

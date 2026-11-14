# Pixora Product and Technical Plan

Pixora is a lightweight, private, open-source desktop photo gallery. It scans local folders, recognises people and scenes on the user's own computer, and offers three ways to browse: Smart Grid, Interactive Map, and People Galaxy.

## Goals

- Run all AI locally. Photos and face data never leave the computer.
- Never modify original photos.
- Keep the gallery usable throughout syncing, with AI results filling in during a background scan.
- Have no fixed library-size limit. Use 20,000 photos as the v1 performance test size.
- Show each unique image only once, even when it exists in several folders.
- Learn a new person from one name and at most two or three follow-up questions.
- Target Windows desktop first, and keep the UI separate from local file access so a web version remains possible.

## Out of scope for v1

- Photo captioning
- Video analysis
- Cloud sync and accounts
- Web and mobile versions
- Photo editing

## Architecture

| Layer | Choice |
|---|---|
| Desktop shell | Tauri 2 |
| UI | React, TypeScript, Vite |
| Galaxy | Three.js with instanced sprites |
| Map | MapLibre GL, Supercluster |
| Grid | TanStack Virtual, justified layout |
| AI service | Python FastAPI sidecar, ONNX Runtime |
| Inference | DirectML by default, CPU fallback |
| Database | SQLite, `sqlite-vec` |

The UI talks to the local service through a typed API. Originals, thumbnails, and model files are reached only through that service, so a future web version can replace the transport without rewriting the views.

## Scan pipeline

1. Walk the selected folders and find new or changed files using path, size, and modified time.
2. Calculate a content hash. If the hash is already known, record the extra file path without adding another library item.
3. Read EXIF date, GPS, orientation, and camera.
4. Generate 256 px and 1024 px WebP thumbnails. Support JPEG, PNG, WebP, HEIC, and embedded RAW previews.
5. Show the photo in the grid as soon as its thumbnail is ready.
6. Calculate a perceptual hash and collapse confirmed near-duplicates.
7. Detect faces, filter by quality, align, and create embeddings.
8. Match faces to known people or cluster them as unknown.
9. Create image embeddings and scene tags.
10. Group photos into events and infer occasions.
11. Update map data and Galaxy layouts.

Each stage is a resumable job that can be paused. Later scans process only new or changed files. A folder watcher is added in the release phase.

### Background sync behaviour

- The first sync never blocks the UI. Photos appear in the grid progressively.
- Each view works with the results available so far and updates as more arrive.
- A compact status indicator shows progress by stage, with pause and resume controls.
- Closing the app and reopening it resumes from the saved job state.
- Photos the user is viewing move to the front of the AI queue.
- AI work yields to the interface and uses limited resources while the user is active.
- The UI requests paged and viewport-sized data, so its memory use does not grow with the whole library. The Galaxy uses level of detail and thumbnail loading on demand.

### Duplicate handling

| Case | Detection | Result |
|---|---|---|
| Same file in several folders | Identical content hash | Automatically shown once; all paths are retained |
| Resized or recompressed copy | Perceptual hash, confirmed by image-embedding similarity | Automatically shown once, using the highest-quality copy as primary |
| Burst or similar shots | Similar but not confirmed as the same image | Kept as separate photos |

Duplicates are hidden, not deleted. Pixora never removes a file during sync. An optional review screen can send user-selected copies to the Recycle Bin. AI runs once for each unique image, and thresholds are validated in Phase 0 against bursts and edited copies.

## AI models

| Task | Model | Notes |
|---|---|---|
| Face detection and recognition, option A | InsightFace `buffalo_l` (RetinaFace, ArcFace R50) | High accuracy; pretrained models are non-commercial |
| Face detection and recognition, option B | YuNet and SFace from OpenCV Zoo | Simpler licensing; confirm the accuracy gap in Phase 0 |
| Face grouping | HDBSCAN | Number of people need not be known |
| Scene tags and search | SigLIP 2 ViT-B/16 | Zero-shot labels and natural-language search |
| Place names | GeoNames offline data | Attribution required |
| Galaxy layout | UMAP | Reduces embeddings to 3D |

### Open-source model policy

- Do not commit model weights to the repository.
- Download models on first run, with checksum verification and the model's licence shown.
- Put face models behind a `FaceEngine` interface.
- Store the engine and model version with every embedding. Changing engines requires reprocessing faces.
- Confirm each model's current licence before release.
- Choose the default engine from Phase 0 results. Prefer YuNet and SFace if their accuracy is good enough.

## Person recognition

1. Ignore tiny, blurry, or strongly turned faces when learning a person.
2. Cluster the remaining faces.
3. Ask "Who is this?" once per cluster, largest first, showing about six faces.
4. Ask up to two or three "Is this also NAME?" questions, choosing the most informative uncertain faces.
5. Store several prototype embeddings per person to cover ageing, glasses, and facial hair.
6. Compare new faces with those prototypes.

Starting thresholds, to be calibrated separately for each engine:

| Similarity | Action |
|---|---|
| Above 0.50 | Assign automatically |
| 0.35 to 0.50 | Queue as a suggestion |
| Below 0.35 | Keep as unknown and cluster |

User corrections always win over automatic results. A "not this person" answer is stored as a constraint. Two named people are never merged automatically.

## Scene and occasion detection

Each photo receives scores against a curated label set covering:

- Nature: mountain, beach, forest, lake, desert, snow
- Places: mall, restaurant, home, office, street, temple, stadium
- Occasions: wedding, birthday, festival, graduation, concert, party

Events are built from gaps in capture time and location. Occasions are inferred at event level by combining the scene scores of all photos in that event, because a single photo is often weak evidence.

## Views

### Smart Grid

- Justified grid grouped by date
- Filters for person, place, scene, occasion, date, and camera, which can be combined
- Natural-language search
- Full-screen viewer with metadata and people

### Interactive Map

- Thumbnail clusters that split smoothly when zooming
- Click to fly to a place and open its photos
- Timeline playback of trips
- "No location" tray, with optional user-confirmed location suggestions based on nearby capture times
- Online tiles by default and an offline PMTiles download, with the privacy trade-off explained

### People Galaxy

Approved concept. Each point is a detected face.

- **People mode:** one coloured cluster per person, with cluster positions derived from person centroids and local positions from that person's faces
- **Scenes mode:** the same points move into scene groups
- **Timeline mode:** the same points spread by year
- Lines between people show how often they appear together
- Zoom levels: dots, then names and cover faces, then thumbnails, then the full photo
- Rotation by dragging, hover details, and animated transitions
- New faces fade into place; full recalculations are aligned to the previous layout and animated

All views share one filter state.

## Data model

```text
photos(id, primary_file_id, content_hash, perceptual_hash, duplicate_of_photo_id, taken_at, lat, lon, width, height, camera, event_id)
photo_files(id, photo_id, path, size, modified_at, content_hash, missing)
faces(id, photo_id, bbox, quality, embedding, engine_version, person_id, assignment_source, confidence)
people(id, name, cover_face_id, hidden)
person_prototypes(person_id, embedding, engine_version)
face_constraints(face_id, person_id, type)
tags(id, name, category)
photo_tags(photo_id, tag_id, score)
photo_embeddings(photo_id, embedding, model_version)
events(id, title, type, start_at, end_at)
layouts(item_id, mode, x, y, z)
jobs(id, photo_id, stage, status, error, updated_at)
```

## Roadmap

| Phase | Deliverable | Exit check | Estimate |
|---|---|---|---|
| 0. Model test | Compare face engines and scene labels on 500 to 1,000 real photos | Default engine and starting thresholds selected | 2 to 3 days |
| 1. Foundation | Tauri app, background sync, duplicate handling, EXIF, thumbnails, grid | 20,000-photo test library is browsable during sync, shows exact duplicates once, resumes after restart, and rescans incrementally | 1 to 2 weeks |
| 2. People | Detection, clustering, naming, corrections, filters | New person learned with at most three follow-up questions | 2 weeks |
| 3. Scenes | Tags, events, occasions, search | Scene filters and search are useful on real photos | 1 week |
| 4. Map | Clusters, animation, offline places, no-location tray | Map stays interactive with the full library | 1 week |
| 5. Galaxy | 3D view, modes, relationship lines, zoom levels | Smooth on the full library; points open the right photos | 2 to 3 weeks |
| 6. Release | Folder watcher, installer, model downloads, licences | Clean install works on a fresh Windows PC | 1 to 2 weeks |

Estimated total for one developer working full-time: 7 to 11 weeks.

## Development environment

Reference PC: Intel Core Ultra 7 265K, 31 GB RAM, NVIDIA GeForce RTX 5060 Ti.

Before Phase 1:

- Install the Rust toolchain, which Tauri requires.
- Create a project-specific Python environment and pin a Python version supported by the selected ONNX Runtime package.

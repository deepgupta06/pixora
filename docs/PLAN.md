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

## Editing and generation

Added after v1 planning. The rules above still hold: everything runs locally and
originals are never modified.

- Edits and new pictures are jobs: one at a time, with progress and cancel. The
  background scan stands aside while a job has the graphics card.
- Every result is a new file in the creations folder, added to the library with
  `photos.origin` set to `edited` or `generated`, and a `creations` row holding the
  operation, the source photo, and the settings (prompt, style, seed) so it can
  be compared with its source and made again.
- A prompt can name a person. Names are matched against People; one clear match
  is used, several matches are put to the user, and an unknown name is reported.
  The person's clearest confirmed faces are the references. Pictures Pixora
  made are never used as references.
- A picture of a known person is made to hold together. The scene is composed
  from words, so its neck, hands and hair would belong to an invented stranger
  once the face is painted again as the person. Three things close the gap:
  the words give the person's age, sex and hair colour (read from their photos
  on this computer); the skin outside the repainted face is moved by the same
  change in tone the face went through, limited to the person's shape found
  with the background model when it is installed; and the user can choose a
  build (slim to larger, in plain words), because no photo of a face shows the body. The
  person's own skin tone is deliberately not put into words: it reads as pale
  in one photo and dark in the next with the flash and white balance, and a
  wrong word would make the picture someone else. Pictures of a real person
  are asked to leave out nudity and cleavage.
- A picture Pixora made can be deleted from the viewer, after a confirmation.
  Its file goes to the Recycle Bin, and its previews, face crops and every
  library record of it are removed. Edits made from it are kept without a
  source. The user's own photos can never be deleted this way: the service
  refuses a photo with no `origin`, or one with a copy outside the creations
  folder.
- Photo numbers can be reused after a delete, so the full-size image address
  includes the content hash; the browser never shows a deleted picture's
  saved copy in place of a new one.
- Incognito, for new pictures: nothing is written to disk or to the database,
  and the description is kept out of the log. The results are held in the
  service's memory, shown with "do not store" so the browser keeps no copy, and
  each is saved or discarded by the user. Save opens the system's save
  window and writes the picture, under a neutral name, to the folder the user
  chooses; it is not added to the library, and once written it is no longer
  held. A discarded one, and the words typed for it, are forgotten. (The
  service can also file a held picture in the library; the app does not
  offer that.) Whatever is still held is gone when the service stops. A
  named person works as usual, since their reference photos are only read.
  While an incognito job runs, warnings are turned off everywhere, because the
  image libraries print the part of a long description they cut off, and set
  their own log levels after the job has started.
- Locked pictures: incognito pictures can be saved locked, one or several together,
  as one `.pxlock` file with a random name that only Pixora can open, and only
  with a key. The user ticks the pictures to lock (or locks one) and the
  service makes a random 256-bit key for the file, encrypts the pictures (each
  as PNG) with AES-256-GCM, and returns the file bytes and the key once. Only
  the pictures are kept: no description, no seed, no other settings. The user
  is shown the key as 13 groups of four base32 characters, with a Copy button
  and a warning that a lost key means lost pictures, and must say they have
  saved it. Pixora keeps the key nowhere: not on disk, not in the database,
  not in the log. The file is magic `PXLOCK`, a version byte (2), a fresh
  96-bit nonce, then ciphertext and tag; the header is the associated data, so
  a changed byte anywhere, or a wrong key, gives one answer: "Wrong key or
  damaged file". Files of the first form (version 1: one picture and its
  settings) still open; their settings are not read. A locked save never goes
  through `Studio.save` or `keep`: no library row, thumbnail, scan, temporary
  file or log line. The held pictures are discarded once the file is written,
  not before, so cancelling the save window loses nothing.
- The incognito picture view: in incognito mode the Create screen shows, above
  the form, the incognito pictures one large with Previous and Next (also the
  arrow keys), their thumbnails below, and for the one shown: Save and Discard.
  Locking is always of the ticked pictures: the ticks are on the thumbnails and
  in a "Select" box beside the picture shown, "Select all" is at the top, and
  "Lock selected" puts the ticked pictures in one file under one key (one
  picture alone is locked by ticking it). There is no lock button for a single
  picture, so a lock is not made one key per picture by accident. The picture
  is kept small enough that the thumbnails and these buttons are on the first
  screen at the app's window size. The view can be maximized to fill the window.
  The list of earlier pictures is not shown in incognito mode. Pictures held by
  the service are shown from it with "do not store"; thumbnails are small
  versions made on request. The view also appears while pictures are held, even
  if the incognito box is unticked, so none sits in memory out of sight.
- Opening a locked file: "Open locked pictures" in that view takes a `.pxlock`
  file and its key; the service decrypts in memory and sends the pictures, with
  a small JPEG of each, as `no-store, no-cache`. They join the view, marked with
  a lock, as blob addresses in the page's memory only: nothing is written, and
  they are forgotten when the screen is left, or with "Close locked pictures".
  Their pictures cannot be dragged out or saved from a menu; "Export as normal
  image" is a separate, deliberate step. This protects the file against other
  programs reading it. It does not protect against someone who has the key,
  against a screenshot, or against a program that can read this computer's
  memory while pictures are open. Not done: registering `.pxlock` with the
  desktop shell so double-clicking opens Pixora; separate Windows windows.
- The models sit behind `StudioEngine`, as face models sit behind `FaceEngine`.

| Task | Model | Notes |
|---|---|---|
| Background removal | BiRefNet (ONNX) | MIT |
| Background change, brush, extend | SDXL inpainting 0.1 | Open RAIL++-M |
| Quality | Real-ESRGAN x4, GFPGAN v1.4 | BSD-3-Clause, Apache 2.0 |
| New pictures, styles | Juggernaut XL v9 (a photographic SDXL fine-tune; plain SDXL base drew poor hands and eyes), ControlNet Canny, one LoRA per style. Each face is painted a second time at full size. | Open RAIL-M; LoRA licences vary |
| Likeness of a person | InstantID on SDXL, InsightFace antelopev2 embedding | Apache 2.0; the embedding model is non-commercial |

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
photos(id, primary_file_id, content_hash, perceptual_hash, duplicate_of_photo_id, taken_at, lat, lon, width, height, camera, event_id, origin)
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
creations(id, photo_id, kind, operation, source_photo_id, params, created_at)
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

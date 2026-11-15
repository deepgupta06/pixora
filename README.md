# pixora

A lightweight, private, open-source desktop photo gallery. All AI runs locally and
original photos are never modified. See [docs/PLAN.md](docs/PLAN.md) for the product
and technical plan.

## Status

Phase 0 (model test) results are in [docs/PHASE0.md](docs/PHASE0.md).

Phase 2 (People): faces are found in a background stage after thumbnails, unknown
faces are grouped, and the People view asks "Who is this?" once per group and at
most three follow-up questions per person. Corrections are made in the photo viewer.

Phase 3 (Scenes): a background stage gives each photo a SigLIP 2 embedding and
scene tags, photos are grouped into events with an inferred occasion, and the
grid has a search box plus scene and occasion filters. Search needs
`siglip2/onnx/model.onnx` and `siglip2/tokenizer.json` (from
`onnx-community/siglip2-base-patch16-224-ONNX`) in the models folder.

Editing and generation: the photo viewer has an Edit mode (remove or change the
background, increase quality, restore an old photo, redraw in a style, paint
over something to remove or replace it, extend the picture), and the Create view
makes new pictures from a description. A description can name someone from
People, and the picture then resembles them; a name that fits several people is
asked about, never guessed. Every result is a new file in
`Pictures\Pixora Creations` (override with `--creations-dir` or
`PIXORA_CREATIONS_DIR`), added to the library and marked Edited or Generated,
with the settings that made it. Originals are never changed, and nothing but
the model downloads touches the network.

This needs an NVIDIA graphics card (tested on 16 GB) and the optional packages:

```bash
cd service
.venv/Scripts/python -m pip install -e ".[studio]" --extra-index-url https://download.pytorch.org/whl/cu128
```

The models (about 21 GB in all) go in `<models folder>/studio`. The app offers
to download what a feature needs the first time it is used, showing each
model's size and licence; `python scripts/download_studio_models.py` fetches
everything at once. They are listed in `service/pixora_service/model_store.py`.
Likeness of a person uses InstantID with the InsightFace `antelopev2` embedding
model, which, like `buffalo_l`, is licensed for non-commercial research use only.

Face models are not in the repository. Until the in-app download exists (release
phase), put them in the models folder: `<data dir>/models`, or the folder given by
`--models-dir` / `PIXORA_MODELS_DIR`. The default engine needs
`buffalo_l/det_10g.onnx` and `buffalo_l/w600k_r50.onnx` from the InsightFace
releases; those pretrained models are licensed for non-commercial use only.
Without them the app works normally and reports that face models are not installed.

Phase 1 (Foundation):

- `service/` — the local Python service: background scan, exact-duplicate handling,
  EXIF, WebP thumbnails, and the typed API.
- `app/` — the React UI: Smart Grid, filters, sync status, folder management, photo viewer.
- `app/src-tauri/` — the Tauri 2 desktop shell. It hosts the UI and starts and stops
  the local service.

## Running the desktop app

Needs the Rust toolchain and the Microsoft C++ Build Tools, plus the service's
virtual environment (see below).

```bash
cd app
npm install
npm run tauri dev
```

The shell starts the service from `service/.venv` and stops it on exit. Packaging
the service into the installer is part of the release phase.

## Running in a browser

Service (Python 3.12):

```bash
cd service
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pixora_service
```

It listens on `127.0.0.1:8732` only and keeps its database and thumbnails in
`%LOCALAPPDATA%\Pixora` (override with `--data-dir` or `PIXORA_DATA_DIR`).

App:

```bash
cd app
npm install
npm run dev
```

Then open http://localhost:5173 and add a folder.

## Tests

```bash
cd service
.venv/Scripts/python -m pytest
```

`service/scripts/make_test_library.py` generates a synthetic library for
performance testing, for example `--count 20000`.

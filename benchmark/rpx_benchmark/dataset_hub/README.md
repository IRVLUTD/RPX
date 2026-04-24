# RPX Dataset Hub — team workflow + TODOs

This subpackage turns the on-disk RPX captures (`test_dataset_aggregated/`)
into the HuggingFace dataset at
[`IRVLUTD/RPX`](https://huggingface.co/datasets/IRVLUTD/RPX). It implements
selective task-based downloads with cache-aware delta reuse, and is
the upload path the team runs from the system that holds the real
~890 GB of captures.

**Current status**: end-to-end pipeline works on synthetic data
(mock → pack → manifest → upload dry-run → download). 400 tests green.
What's left before v1 ships to HF is the file-staging work (splits,
dataset card, preview) and one real run on the target system — see
[TODOs](#todos) below.

---

## 0. Pre-reqs

```bash
pip install -e 'benchmark[hub]'          # huggingface_hub + pyarrow
hf auth login                            # push access to IRVLUTD org
```

Verify:
```bash
hf whoami                                # → your HF username
cd benchmark && pytest -q tests/test_dataset_hub_*.py      # → 80+ green
```

---

## 1. The pipeline in one minute

```
              ┌─────────┐                 ┌──────────┐
capture tree  │         │  tar shards     │          │  HF repo
(test_dataset │  pack   │  + manifest +   │  upload  │  IRVLUTD/RPX
_aggregated/) │         │  objects_meta/  │          │
              └─────────┘                 └──────────┘
                  ↑                            │
                  │ reads                      │ snapshot_download
              ┌─────────┐                      ↓
              │  scan   │                 ┌──────────┐
              └─────────┘                 │ download │  local cache,
                                          │ --task   │  just the
                                          │ --split  │  tars the
                                          └──────────┘  task needs
```

**Three CLI entry points you'll actually run:**

```bash
# Walk the captures on disk, see how much data there is.
python -m rpx_benchmark.dataset_hub.cli scan /path/to/test_dataset_aggregated

# Turn the captures into HF-shaped tar shards + manifest + objects_meta/.
python -m rpx_benchmark.dataset_hub.cli pack \
    --src /path/to/test_dataset_aggregated \
    --staging /path/to/staging
python -m rpx_benchmark.dataset_hub.cli manifest \
    --src /path/to/test_dataset_aggregated \
    --staging /path/to/staging \
    --splits benchmark/data/splits/scene_splits.json

# Push to the HF dataset repo (resumable; takes hours at 890 GB).
python -m rpx_benchmark.dataset_hub.cli upload \
    --staging /path/to/staging \
    --repo-id IRVLUTD/RPX
```

**User-facing command** (what the team and collaborators will call):

```bash
python -m rpx_benchmark.dataset_hub.cli download \
    --task segmentation --split easy
# → fetches only the RGB + masks tars for easy-tier MOS scenes.
# Re-running with --task relative_pose --split easy reuses the RGB
# tars from the cache and only fetches the cam_pose tars as the delta.
```

**Mock** lets you exercise the whole chain locally in ~1 s:

```bash
python -m rpx_benchmark.dataset_hub.cli mock --out /tmp/rpx_mock
python -m rpx_benchmark.dataset_hub.cli pack \
    --src /tmp/rpx_mock --staging /tmp/rpx_stage
# Inspect the staging tree — it's the HF repo shape.
find /tmp/rpx_stage -type f | head
```

---

## 2. The on-disk source layout (what the team arranges)

See `benchmark/templates/rpx_capture_wireframe/README.md` for the full
spec. Short version:

```
test_dataset_aggregated/
├── mos/                      # multi-object scenes (100)
│   ├── scene1/
│   │   ├── 0/ 1/ 2/          # three phases: clutter, interaction, clean
│   │   │   ├── rgb/ depth/ fisheye/{left,right}/ cam_pose/
│   │   │   └── sam2/
│   │   │       ├── masks/
│   │   │       ├── bbox_overlay/ …                 # viz, opt-in
│   │   │       ├── mask_to_object.json             # {mask_id → object_id}
│   │   │       ├── verified_masks.txt
│   │   │       └── iter1_faulty.txt … iter4_faulty.txt
│   │   └── ...
│   └── ... scene100/
└── sos/                      # single-object scenes (220)
    ├── tape_and_holder/
    │   ├── questionnaire.txt                        # FewSOL-style Q&A
    │   └── 0/                                       # only one "phase"
    │       └── ... same modalities as MOS ...
    └── ... 220 objects ...
```

**Three rules the team must honour** (the packer + downloader depend on
them):

1. **SOS dir name == canonical `object_id`.** Use the same string in MOS
   `mask_to_object.json` values.
2. **`questionnaire.txt` lives at the SOS scene root**, not inside a
   phase. MOS scenes never carry one.
3. **Frame names `00000.png`, 5-digit zero-padded, aligned across
   modalities.** 250 frames per phase.

---

## 3. HF repo layout the pipeline produces

```
IRVLUTD/RPX/
├── manifest/
│   ├── frames_v1.parquet     # per-frame metadata (always pulled, small)
│   └── current.json          # {label_versions: {masks: v1, ...}}
├── splits/                   # ← TODO 1 adds these
│   ├── scene_splits.json
│   ├── easy.txt  medium.txt  hard.txt
├── scenes/<scene_id>/<phase>/
│   ├── rgb.tar  depth.tar  fisheye.tar  cam_pose.tar
│   └── labels/{masks,masks_aux,sam2_meta,vqa}/v1.tar
├── objects/<object_id>/0/
│   └── (same modality tars)
├── objects_meta/
│   ├── _index.json
│   └── <object_id>/questionnaire.json     # parsed from FewSOL text
├── preview/                  # ← TODO 4 adds these
├── README.md                 # ← TODO 2 adds this (dataset card)
└── rpx_croissant.json        # ← TODO 3 adds this
```

---

## 4. Where the code lives

| Module | Purpose | Entry point |
|---|---|---|
| `mock.py` | Generate synthetic capture tree | `cli mock` |
| `scanner.py` | Inventory the capture tree | `cli scan` |
| `packer.py` | Capture tree → per-(scene, phase, modality) tars + `objects_meta/` | `cli pack` |
| `manifest.py` | Per-frame Parquet + `current.json` | `cli manifest` |
| `uploader.py` | HF push (resumable) | `cli upload` |
| `downloader.py` | Selective pull by (task, split) | `cli download` |
| `recipes.py` | Task → modality recipes | lib only |
| `cli.py` | `python -m rpx_benchmark.dataset_hub.cli <cmd>` | — |

---

## TODOs

Five items before v1 ships to HF, ordered by dependency.

### TODO 1 — Ship the splits files into staging (small, blocking)

The manifest's `split` column is only populated when we pass
`--splits scene_splits.json` to the `manifest` CLI (already supported).
We also need the splits files at the HF repo root for downstream
tooling. Two tiny file copies during staging.

- **Where**: new `staging.py` module in `dataset_hub/` + new CLI
  subcommand `cli stage-splits`.
- **What**:
  - Copy `benchmark/data/splits/scene_splits.json` →
    `<staging>/splits/scene_splits.json`
  - Copy `easy.txt`, `medium.txt`, `hard.txt` → `<staging>/splits/`
  - Done.
- **Acceptance**:
  - `staging/splits/{easy,medium,hard}.txt` and `scene_splits.json`
    exist after running the new subcommand.
  - A test that passes the resulting `scene_splits.json` to
    `build_frame_manifest(..., splits=...)` and confirms the `split`
    column gets populated.
- **Estimated effort**: ~30 min including tests.

### TODO 2 — Generate the HuggingFace dataset card

HF Dataset Viewer needs a `README.md` with YAML frontmatter at the
repo root. Template is already at
`benchmark/templates/hf_dataset_card.md` — needs to be populated with
real metadata (num_scenes, total_bytes, modalities, task_categories,
splits, license).

- **Where**: new `dataset_card.py` in `dataset_hub/` + `cli dataset-card`.
- **What**:
  - Fill the template using the `ScanResult` + manifest totals from
    `manifest.py`.
  - Render and write to `<staging>/README.md`.
  - YAML frontmatter must include:
    ```yaml
    task_categories: [image-segmentation, depth-estimation,
                      visual-question-answering, ...]
    tags: [robotics, manipulation, rgb-d, realsense]
    size_categories: [100B<n<1T]
    license: <pick one: cc-by-4.0 or mit or apache-2.0>
    ```
  - Include a quick-start `snapshot_download` example in the card.
- **Acceptance**:
  - `staging/README.md` exists, parses as valid YAML frontmatter + MD.
  - Upload dry-run lists it as the top-level file.
- **Estimated effort**: ~1 h including a representative rendering test.

### TODO 3 — Copy Croissant metadata into staging + reconcile

`paper-submission/neurips-2026/croissant/rpx_croissant.json` already
exists. We need to (a) copy it into `<staging>/rpx_croissant.json`
and (b) reconcile its field paths against the new tar layout (it was
drafted against the earlier loose-file layout).

- **Where**: extend `staging.py` with `stage_croissant()`.
- **What**:
  - Copy the JSON into staging root.
  - Walk the JSON's `distribution` and `recordSet` entries, rewrite
    any paths that still reference the old `rgb/*.png` layout to the
    new `rgb.tar` shard paths.
- **Acceptance**:
  - `staging/rpx_croissant.json` exists.
  - A test that parses the JSON and confirms every file path it
    references resolves to an actual file or glob in `<staging>/`.
- **Estimated effort**: ~1 h (mostly the reconciliation).

### TODO 4 — Preview subset for the HF Dataset Viewer

Pick ~50 representative frames (mix of easy/medium/hard, 1–2 SOS)
and stage them under `<staging>/preview/`. HF Dataset Viewer renders
these as a thumbnail strip, which is the "window shopping" experience.

- **Where**: new `preview.py` in `dataset_hub/` + `cli preview`.
- **What**:
  - Curate frames: the CLI takes a `--config preview.yaml` listing
    `(scene_id, phase, frame_idx)` tuples, or a `--auto` mode that
    samples 1 frame per tertile per 5 MOS scenes + 10 SOS scenes.
  - For each curated frame: decode the RGB from its tar shard,
    overlay the mask, resize to 512×512, write to
    `<staging>/preview/<nnn>_<scene_id>_<phase>_<frame>.jpg`.
  - Also write `<staging>/preview/thumbnails.parquet` with embedded
    JPEG bytes for the viewer's native thumbnail panel (HF viewer
    auto-detects Parquet with image columns).
- **Acceptance**:
  - `staging/preview/` contains ~50 JPEGs + the Parquet.
  - Total preview dir size < 50 MB (so the viewer loads fast).
- **Estimated effort**: ~2 h (includes the mask-overlay renderer).

### TODO 5 — Python loader that decodes (task, split) → iterator of samples

The downloader currently stops at "tars on disk". Users want a
Python loop that yields decoded frames. Plumb this through using
`rpx_benchmark.loader.RPXDataset` (already exists) or a new
`DatasetHubLoader`.

- **Where**: new `loader.py` in `dataset_hub/`.
- **What**:
  - Given `DownloadResult`, open each relevant tar with
    `tarfile.open(..., "r|")` for streaming, yield
    `Sample(rgb=PIL, depth=np.ndarray, masks=..., cam_pose=...)`.
  - Honour the manifest's `has_<modality>` column — skip frames
    where any required modality is missing.
  - Support iteration ordering: `"sequential"` (by `(scene, phase,
    frame_idx)`) or `"shuffle(seed=N)"`.
- **Acceptance**:
  - On the mock dataset, a loader loop yields the expected number
    of samples, each with the correct modalities populated.
  - Works for both MOS (with `split`) and SOS (no split) recipes.
  - A torch/jax-agnostic iterator that can be wrapped into a
    `DataLoader` in three lines.
- **Estimated effort**: ~3 h including tests.

---

## 5. Running on the real captures (once the above is done)

On the system holding the real ~890 GB:

```bash
# 0. One-time setup
pip install -e 'benchmark[hub]'
hf auth login

# 1. Sanity-check the layout
python -m rpx_benchmark.dataset_hub.cli scan /data/test_dataset_aggregated
# → "100 scenes, 220 objects, 780k files, 890 GB"
# → check: no stray entries in the "skipped" list

# 2. Pack to a local staging dir (will need ~2 TB scratch)
python -m rpx_benchmark.dataset_hub.cli pack \
    --src /data/test_dataset_aggregated \
    --staging /scratch/rpx_stage

# 3. Build the manifest with split assignments
python -m rpx_benchmark.dataset_hub.cli manifest \
    --src /data/test_dataset_aggregated \
    --staging /scratch/rpx_stage \
    --splits benchmark/data/splits/scene_splits.json

# 4. (TODOs 1-4) Stage splits, dataset card, croissant, preview
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging /scratch/rpx_stage   # TODO 1
python -m rpx_benchmark.dataset_hub.cli dataset-card    --staging /scratch/rpx_stage   # TODO 2
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging /scratch/rpx_stage   # TODO 3
python -m rpx_benchmark.dataset_hub.cli preview --auto  --staging /scratch/rpx_stage   # TODO 4

# 5. Dry-run the upload — checks file count + bytes + ignores
python -m rpx_benchmark.dataset_hub.cli upload \
    --staging /scratch/rpx_stage \
    --repo-id IRVLUTD/RPX \
    --dry-run
# → e.g. "would push 2,103 files (889.2 GB) to IRVLUTD/RPX"

# 6. Real upload (resumable; hours)
python -m rpx_benchmark.dataset_hub.cli upload \
    --staging /scratch/rpx_stage \
    --repo-id IRVLUTD/RPX

# 7. Smoke-test a pull from a clean cache
HF_HOME=/tmp/hf_cache_check python -m rpx_benchmark.dataset_hub.cli download \
    --task segmentation --split easy --cache-dir /tmp/hf_cache_check
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `scan` reports `skipped: [mos/scence1]` (typo) | scene dir name doesn't match `scene*` or `object*` | rename on disk and re-scan |
| `pack` raises `refusing to overwrite existing shard` | staging dir already populated | pass `--overwrite` or remove the staging dir |
| `upload --dry-run` shows a `.fls`, `.aux`, etc. | LaTeX build artefacts leaked into staging | they are covered by `DEFAULT_IGNORE_PATTERNS`; confirm you're on a staging dir built by `pack`, not a hand-copied tree |
| `download` raises `No scenes matched task=... split=...` | manifest has `split = null` everywhere | pass `--splits scene_splits.json` to `manifest` when building the staging dir |
| Upload stalls or dies mid-transfer | network blip / HF rate limit | just re-run the same `upload` command — `upload_large_folder` checkpoints in `<staging>/.huggingface/` |

---

## 7. Tests

```bash
cd benchmark
pytest -q tests/test_dataset_hub_*.py
# → 80 tests across 7 files; runs in < 15 s.
```

Each source module has its own test file. New TODOs should ship with
tests — pattern-match off existing ones
(`test_dataset_hub_packer.py`, `test_dataset_hub_downloader.py`,
`test_dataset_hub_objects_meta.py`).

---

## Ownership

| Area | Primary owner | Backup |
|---|---|---|
| Source layout / wireframe conformance | **team (whoever arranges the captures on disk)** | — |
| Pipeline code + tests | jishnu | open to PRs |
| HF repo permissions | jishnu | — |
| Label re-releases (masks v2, vqa v1) | TBD (when data lands) | — |

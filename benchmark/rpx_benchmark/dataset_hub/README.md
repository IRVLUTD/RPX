# RPX Dataset Hub — team guide

> Uploads the RPX captures to
> [`IRVLUTD/RPX`](https://huggingface.co/datasets/IRVLUTD/RPX) on
> HuggingFace, and lets users download only the slice they need
> (by task + split) instead of pulling the full ~890 GB.

> **Status**: end-to-end pipeline works on synthetic data; tests green.
> Real upload pending team arranging captures into the wireframe layout
> (see [§3](#3-how-the-captures-must-be-arranged-on-disk)) on the
> target system.

## Contents

- [1. The 5-minute version](#1-the-5-minute-version)
- [2. What each command does](#2-what-each-command-does)
- [3. How the captures must be arranged on disk](#3-how-the-captures-must-be-arranged-on-disk)
- [4. What ends up on the HuggingFace repo](#4-what-ends-up-on-the-huggingface-repo)
- [5. Updating the dataset later](#5-updating-the-dataset-later)
- [6. What is Croissant?](#6-what-is-croissant)
- [7. Running the full pipeline against the real data](#7-running-the-full-pipeline-against-the-real-data)
- [8. Troubleshooting](#8-troubleshooting)
- [9. Where the code lives](#9-where-the-code-lives)
- [10. Tests](#10-tests)
- [Owners](#owners)

---

## 1. The 5-minute version

```bash
# one-time
pip install -e 'benchmark[hub]'
hf auth login

# upload (run on the system that has the ~890 GB captures)
python -m rpx_benchmark.dataset_hub.cli pack            --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli manifest        --src DATA --staging STAGE \
                                                         --splits benchmark/data/splits/scene_splits.json
# `manifest` writes:
#   1. STAGE/manifest/frames_v1.parquet   (all-frames index)
#   2. STAGE/manifest/current.json        (schema version + label versions)
#   3. STAGE/manifests/<recipe>/<split>.json — per-task per-split manifests
#      for the 7 task recipes the loader supports (monocular_depth,
#      segmentation, rgbd_segmentation, stereo_depth, relative_pose,
#      rgbd_relative_pose, object_tracking) × {easy, medium, hard} = 21 JSONs.
#      The vqa recipe is wired but emits 0 entries until the team's VQA
#      label-generation pipeline lands (logs a clear warning).
#  Without --splits the manifest step now FAILS LOUDLY (used to silently
#  produce 0 per-task JSONs which made the published HF tree unusable).
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli dataset-card    --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli upload          --staging STAGE  --repo-id IRVLUTD/RPX

# download (any machine, any time) — fetches the requested split's
# manifest + tar shards, then extracts the tars into <snapshot>/extracted/
# automatically (so RPXDataset.from_manifest can consume the JSON
# without an extra extraction step on the user side).
python -m rpx_benchmark.dataset_hub.cli download --task segmentation --split easy
```

`DATA` is your captures root (the dir holding `mos/` and `sos/`).
`STAGE` is any local scratch dir with enough free space.

---

## 2. What each command does

| Command | One-line job |
|---|---|
| `scan` | Walk the captures, report counts and bytes per modality. Read-only. |
| `pack` | Bundle each `(scene, phase, modality)` into a tar shard ready for upload. |
| `manifest` | Build the per-frame Parquet that tells downloaders which scenes belong to which split. |
| `stage-splits` | Copy `splits/{easy,medium,hard}.txt` and `scene_splits.json` into the staging dir. |
| `dataset-card` | Generate the `README.md` HuggingFace shows on the dataset page. |
| `stage-croissant` | Copy the Croissant metadata JSON (see §6 for what that is). |
| `upload` | Push the staging dir to the HF repo. Resumable. |
| `download` | (Users.) Pull just the files a `(task, split)` needs. |

If you forget any of `stage-splits` / `dataset-card` / `stage-croissant`,
the upload still works, but the HF dataset page will be sparse and
downloaders will not find a `split` column to filter on.

---

## 3. How the captures must be arranged on disk

The pipeline expects this directory shape under your data root. There's
a fully-populated wireframe at `benchmark/templates/rpx_capture_wireframe/`
that the team can copy as a starting template.

```
DATA/
├── mos/                              # 100 multi-object scenes
│   ├── scene1/
│   │   ├── 0/    (clutter phase)
│   │   ├── 1/    (interaction phase)
│   │   └── 2/    (clean phase)
│   │       ├── rgb/         00000.png …  (5-digit, 250 frames)
│   │       ├── depth/       00000.png …  (16-bit grayscale)
│   │       ├── fisheye/
│   │       │   ├── left/    00000.png …  (T265 left)
│   │       │   └── right/   00000.png …  (T265 right, same filenames)
│   │       ├── cam_pose/    00000.json … (T265 SLAM pose)
│   │       └── sam2/
│   │           ├── masks/                  00000.png … (instance IDs)
│   │           ├── bbox_overlay/ … rgb_and_mask/   (six viz dirs, opt-in)
│   │           ├── mask_to_object.json     {"1": "tape_and_holder", "2": …}
│   │           ├── verified_masks.txt
│   │           └── iter1_faulty.txt … iter4_faulty.txt
│   └── … scene100/
└── sos/                              # 220 single-object scenes
    ├── tape_and_holder/              # ← directory name = the canonical object_id
    │   ├── questionnaire.txt         # FewSOL Q&A about this object
    │   └── 0/                        # only one phase (360° turntable)
    │       └── … same modality dirs as MOS …
    └── … 219 more objects …
```

**Three rules to honour:**

1. **MOS dirs**: `scene1`, `scene2`, …, `scene100`.
2. **SOS dirs**: bare object names (`tape_and_holder`, `coffee_mug`, …),
   no prefix. **The directory name is the object's canonical ID** — use
   the same string in MOS `mask_to_object.json` values.
3. **Frame names**: `00000.png` (5-digit zero-padded), aligned across
   modalities, 250 per phase.

If anything looks wrong, run `scan` — it lists "skipped" entries that
didn't fit the pattern (typos, stray files).

---

## 4. What ends up on the HuggingFace repo

Your `pack`/`upload` produces this on `IRVLUTD/RPX`:

```
IRVLUTD/RPX/
├── README.md                         # the dataset card (from `dataset-card`)
├── rpx_croissant.json                # metadata for ML platforms (from `stage-croissant`)
├── manifest/
│   ├── frames_v1.parquet             # one row per frame; users pull this first
│   └── current.json                  # which label version is the default
├── splits/
│   ├── scene_splits.json
│   ├── easy.txt  medium.txt  hard.txt
├── scenes/<scene_id>/<phase>/                       # MOS captures
│   ├── rgb.tar  depth.tar  fisheye.tar
│   └── labels/{cam_pose,masks,masks_aux,sam2_meta,vqa}/v1.tar
├── objects/<object_id>/0/                           # SOS captures
│   └── (same modality tars)
└── objects_meta/                                    # questionnaire dedup
    ├── _index.json
    └── <object_id>/questionnaire.json
```

Two things to notice:

- **One tar per modality, not per frame.** A user fetching segmentation
  pulls a few hundred tars (rgb + masks for the matched scenes) — never
  the whole repo, never a million loose files.
- **Labels are versioned.** Bumping a label version is a one-line config
  change. Users get the new version automatically via `current.json`,
  or pin to the old one with `--label-version <name>=v<N>`.

---

## 5. Updating the dataset later

The most common updates and how to ship each.

### 5a. Adding new label data (e.g. VQA arrives)

```bash
# After the new vqa/ subdirs are written under each phase on disk:
python -m rpx_benchmark.dataset_hub.cli pack --src DATA --staging STAGE --overwrite
# Bump current.json so default downloaders pick up the new label
python -c "import json, pathlib; p = pathlib.Path('STAGE/manifest/current.json'); \
           c = json.loads(p.read_text()); c['label_versions']['vqa'] = 'v1'; \
           p.write_text(json.dumps(c, indent=2))"
python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE --repo-id IRVLUTD/RPX \
    --message "v1.1.0: add VQA labels"
```

Users get VQA on the next `download --task vqa --split <X>` call.
Other tasks (`segmentation`, `relative_pose`, …) are untouched
because their tar paths are unchanged — HF cache hits for everyone.

### 5b. Re-releasing an existing label (e.g. cam_pose refinement)

```bash
# After the refined cam_pose/ files replace the old ones on disk:
python -m rpx_benchmark.dataset_hub.cli pack --src DATA --staging STAGE \
    --label-version v2 --overwrite
# This adds  scenes/<scene>/<phase>/labels/cam_pose/v2.tar
# alongside the existing v1 files (which stay reachable for reproducibility).

# Bump current.json so default downloaders pick up v2
python -c "import json, pathlib; p = pathlib.Path('STAGE/manifest/current.json'); \
           c = json.loads(p.read_text()); c['label_versions']['cam_pose'] = 'v2'; \
           p.write_text(json.dumps(c, indent=2))"
python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE --repo-id IRVLUTD/RPX \
    --message "v1.2.0: refined cam_pose (v2)"
```

A user pinning to v1 just adds `--label-version cam_pose=v1` and gets
the original poses forever.

### 5c. Adding new scenes after v1 ships

Drop the new scenes into `DATA/mos/` (or `DATA/sos/`) and re-run
`pack` + `manifest` + `upload --message "v1.x.x: add N scenes"`. The
unchanged scenes' tars are byte-identical (the packer is deterministic)
so HF skips them; only the new tars transfer.

### 5d. Removing a scene

`huggingface_hub` doesn't expose a clean delete-folder API. Use the
HF web UI: Files & versions → navigate to the scene dir → Delete.
Then re-build the manifest locally and re-upload `manifest/frames_v1.parquet`
so the row count matches. Rare; usually fixing a bad capture rather
than removing.

### 5e. Bumping the dataset version (semantic releases)

Tag the HF repo with a Git revision (`v1.0.0`, `v1.1.0`, ...) so users
can pin to a specific release for reproducibility:

```python
# in code:
download_for_task(task=..., split=..., revision="v1.0.0")
```

The HF UI's "Settings → Git tags" creates these.

---

## 6. What is Croissant?

[Croissant](https://mlcommons.org/working-groups/croissant/) is a
metadata standard for ML datasets — a JSON file that describes the
dataset's structure (fields, types, splits, license, citation) in a
machine-readable form. HuggingFace, Kaggle, OpenML, and Papers-with-Code
all parse Croissant to render rich previews and search filters.

We already authored ours at
`paper-submission/croissant/rpx_croissant.json`. The
`stage-croissant` command copies it into the upload tree and patches a
few fields (URL, version, optional bibtex) to point at the live HF repo.

You don't need to write or read Croissant manually — `stage-croissant`
handles it. The only reason to touch the JSON is if you want to
register a new field schema for a new modality (e.g. once VQA labels
land, adding a `vqa_qa` recordSet). That's a one-time edit per major
schema change.

---

## 7. Running the full pipeline against the real data

On the system with the ~890 GB captures, in order:

```bash
# 0. install
pip install -e 'benchmark[hub]'
hf auth login

# 1. sanity-check the layout
python -m rpx_benchmark.dataset_hub.cli scan /data/test_dataset_aggregated
# expect: ~100 MOS scenes, ~220 SOS scenes, ~890 GB.
# any "skipped" entries are typos or strays — fix on disk and re-scan.

# 2. pack (≈2 TB scratch needed for the staging dir)
python -m rpx_benchmark.dataset_hub.cli pack \
    --src /data/test_dataset_aggregated --staging /scratch/rpx_stage

# 3. build the manifest with split assignments
python -m rpx_benchmark.dataset_hub.cli manifest \
    --src /data/test_dataset_aggregated --staging /scratch/rpx_stage \
    --splits benchmark/data/splits/scene_splits.json

# 4. small companion files
python -m rpx_benchmark.dataset_hub.cli stage-splits     --staging /scratch/rpx_stage
python -m rpx_benchmark.dataset_hub.cli dataset-card \
    --src /data/test_dataset_aggregated --staging /scratch/rpx_stage \
    --repo-id IRVLUTD/RPX
python -m rpx_benchmark.dataset_hub.cli stage-croissant \
    --staging /scratch/rpx_stage --repo-id IRVLUTD/RPX

# 5. dry-run the upload (file count + bytes, no network)
python -m rpx_benchmark.dataset_hub.cli upload \
    --staging /scratch/rpx_stage --repo-id IRVLUTD/RPX --dry-run

# 6. real upload (resumable; takes hours)
python -m rpx_benchmark.dataset_hub.cli upload \
    --staging /scratch/rpx_stage --repo-id IRVLUTD/RPX

# 7. smoke-test from a clean cache
HF_HOME=/tmp/hf_check python -m rpx_benchmark.dataset_hub.cli download \
    --task segmentation --split easy --cache-dir /tmp/hf_check
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `scan` reports `skipped: [mos/scence1]` | scene dir name typo | rename on disk and re-scan |
| `pack` raises `refusing to overwrite existing shard` | staging dir already populated | pass `--overwrite` or remove the staging dir |
| `upload --dry-run` shows `.aux`, `.fls`, etc. | LaTeX build artefacts leaked in | `DEFAULT_IGNORE_PATTERNS` covers these; check you're staging from `pack` output |
| `download` raises `No scenes matched task=... split=...` | manifest has `split = null` everywhere | run `manifest` with `--splits scene_splits.json` |
| Upload stalls or dies mid-transfer | network blip / HF rate limit | re-run the same `upload` command — checkpoint is in `<staging>/.huggingface/` |

---

## 9. Where the code lives

| Module | Job |
|---|---|
| `scanner.py` | Walks captures, reports inventory |
| `packer.py` | Captures → per-modality tars + `objects_meta/` |
| `manifest.py` | Builds per-frame Parquet + `current.json` |
| `staging.py` | Copies `splits/` files |
| `dataset_card.py` | Renders the HF `README.md` |
| `croissant.py` | Patches and copies the Croissant JSON |
| `uploader.py` | Pushes staging to HF (resumable) |
| `downloader.py` | Pulls just the (task, split) slice users need |
| `recipes.py` | Task → modality lookup table |
| `mock.py` | Generates a tiny synthetic capture tree for testing |
| `cli.py` | All `python -m rpx_benchmark.dataset_hub.cli <cmd>` entry points |

---

## 10. Tests

```bash
cd benchmark
pytest -q tests/test_dataset_hub_*.py
# 100+ tests across ~10 files; runs in ~20 s.
```

If you change layout or the recipe table, re-run; the suite catches the
common breakages (wrong tar paths, missing modalities, stale tests).

---

## Owners

| Area | Owner |
|---|---|
| Source layout / wireframe conformance | the team member who arranges captures on disk |
| Pipeline code + tests | jishnu |
| HF repo permissions | jishnu |
| Label re-releases (cam_pose v2, vqa v1) | TBD when the data lands |

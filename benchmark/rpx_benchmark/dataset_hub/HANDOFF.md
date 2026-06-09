# RPX Dataset Hub — Handoff Snapshot

**Purpose**: self-contained snapshot of the dataset-hub work so a new
session (human, LLM, or new teammate) can pick up without re-reading
the full conversation. Updated 2026-04-24.

Last commit on `jishnu/benchmark-code-skeleton`: **6405145**.
Tests: **435 passing**.

---

## 1. What we're building and why

The RPX benchmark ships a real-world RGB-D dataset (~890 GB total, 100
multi-object + 220 single-object scenes, ~250 frames per phase) for
evaluating robot perception under deployment conditions. This
subpackage (`rpx_benchmark.dataset_hub`) uploads that dataset to the
HuggingFace Hub at [`IRVLUTD/RPX`](https://huggingface.co/datasets/IRVLUTD/RPX)
and lets users download **just the slice they need** for a particular
task+split, never the whole thing.

The four levers that make selective download work:

| Lever | Job |
|---|---|
| **Recipe table** (`recipes.py`) | "task X needs these modalities" |
| **Manifest** (`manifest/frames_v1.parquet` on HF) | "split Y contains these scene_ids" |
| **Pattern builder** (`downloader.py`) | "(scenes × modalities) → HF glob list" |
| **`snapshot_download(allow_patterns=…)`** | "HF only sends files matching these globs" |

Delta downloads are free: HF's cache is content-addressed, so switching
tasks on the same scenes reuses the cached modalities and only pulls
the new ones.

---

## 2. The design decisions already locked in

These were debated and settled; **do not revisit without cause**:

1. **Hosting platform**: HuggingFace Hub (not hf-mount, not self-host).
   Repo `IRVLUTD/RPX`, type `dataset`, private until NeurIPS submission.

2. **Sharding granularity**: one tar per `(scene, phase, modality)`
   triple. At 100 scenes × 3 phases × ~7 modalities ≈ 2,100 MOS tars +
   ~1,540 SOS tars = ~3,600 shards total. Not loose files (too many),
   not one-big-blob (kills selective download).

3. **Source-side layout** (team arranges captures this way):
   ```
   DATA/
   ├── mos/scene<N>/<phase>/<modality>/...     # 100 scenes × 3 phases
   └── sos/<object_name>/0/<modality>/...      # 220 objects, 1 phase each
   ```
   MOS dirs are bare `scene1, scene2, …, scene100`. SOS dirs are bare
   object names (`tape_and_holder`, `coffee_mug`, …) — the directory
   name **is** the canonical `object_id`.

4. **HF-side layout**:
   ```
   IRVLUTD/RPX/
   ├── README.md                              # dataset card
   ├── rpx_croissant.json                     # ML metadata
   ├── manifest/frames_v1.parquet + current.json
   ├── splits/{easy,medium,hard}.txt + scene_splits.json
   ├── scenes/<scene_id>/<phase>/
   │   ├── rgb.tar  depth.tar  fisheye.tar
   │   └── labels/{cam_pose,masks,masks_aux,sam2_meta,vqa}/v1.tar
   ├── objects/<object_id>/0/
   │   └── (same modality tars)
   └── objects_meta/<object_id>/questionnaire.json   (shared across MOS/SOS)
   ```

5. **Raw vs label split**:
   - **Raw** (immutable, no version in path): rgb, depth, fisheye
   - **Labels** (versioned at `labels/<name>/v<N>.tar`): cam_pose,
     masks, masks_aux, sam2_meta, vqa
   - Put cam_pose in labels (not raw) because SLAM refinement will
     produce a v2 later; users should be able to pin to v1.

6. **Questionnaire dedup**: 220 SOS objects have their own
   `questionnaire.txt` (FewSOL Q&A format). 70 of those objects also
   appear in MOS scenes, discovered via each MOS phase's
   `sam2/mask_to_object.json` file that maps `{mask_id → object_id}`.
   Packer parses each `questionnaire.txt` once into
   `objects_meta/<object_id>/questionnaire.json` and users look up via
   the shared ID — no duplication, no per-scene copies.

7. **Release train**:
   - **v1.0.0** (upload now): rgb, depth, fisheye + cam_pose v1, masks
     v1, masks_aux v1, sam2_meta v1 + questionnaires. **VQA absent**
     (recipe slot reserved).
   - **v1.1.0** (later): add `labels/vqa/v1.tar` per scene + bump
     `current.json`. Everything else unchanged; HF cache hits for all
     other tasks.
   - **v1.2.0** (later): `labels/cam_pose/v2.tar` lands alongside v1;
     bump `current.json`. Users pinning to v1 pass
     `--label-version cam_pose=v1`.

---

## 3. Repo location and navigation

```
/home/jishnu/Projects/phd/code/RPX/               ← repo root
├── benchmark/
│   ├── rpx_benchmark/
│   │   ├── hub.py                                ← older HF helper (still used)
│   │   └── dataset_hub/                          ← THIS WORK
│   │       ├── __init__.py
│   │       ├── README.md                         ← team guide (start here)
│   │       ├── HANDOFF.md                        ← this file
│   │       ├── recipes.py
│   │       ├── mock.py
│   │       ├── scanner.py
│   │       ├── packer.py
│   │       ├── manifest.py
│   │       ├── staging.py
│   │       ├── dataset_card.py
│   │       ├── croissant.py
│   │       ├── uploader.py
│   │       ├── downloader.py
│   │       └── cli.py
│   ├── templates/
│   │   └── rpx_capture_wireframe/                ← empty dir template for the team
│   ├── tests/
│   │   └── test_dataset_hub_*.py                 ← ~120 tests across 10 files
│   └── pyproject.toml                            ← [hub] extras = huggingface_hub + pyarrow
└── paper-submission/
    └── croissant/rpx_croissant.json              ← the pre-authored Croissant (gets copied+patched)
```

---

## 4. Full pipeline in seven commands

```bash
# prereq (once)
pip install -e 'benchmark[hub]'
hf auth login

# DATA = captures root (holds mos/ and sos/), STAGE = scratch dir
#
# Optional pre-step: re-encode rgb/fisheye/ego PNGs as lossless WebP,
# re-compress depth/mask PNGs at level 9 (still lossless), and
# consolidate per-frame cam_pose .npz into per-frame .npy. ~28% denser
# end-to-end with zero loss; depth/mask file format stays PNG so the
# benchmarking loaders are untouched. Each modality has its own
# --skip-* flag to opt out if needed.
python -m rpx_benchmark.dataset_hub.cli lossless-convert --src DATA --out DATA_v2
# DATA=DATA_v2 from here on if you used the pre-step.

python -m rpx_benchmark.dataset_hub.cli pack            --src DATA --staging STAGE
python -m rpx_benchmark.dataset_hub.cli manifest        --src DATA --staging STAGE \
                                                         --splits benchmark/data/splits/scene_splits.json
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging STAGE
python -m rpx_benchmark.dataset_hub.cli dataset-card    --src DATA --staging STAGE
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging STAGE
python -m rpx_benchmark.dataset_hub.cli upload          --staging STAGE --repo-id IRVLUTD/RPX

# user side (any machine)
python -m rpx_benchmark.dataset_hub.cli download --task segmentation --split easy
```

### lossless-convert — per-modality contract

Every re-encoded artefact is round-trip verified at conversion time:
the worker decodes the freshly-written file, asserts
`numpy.array_equal` (or full pose-vector equality for cam_pose), and
the run aborts on the first mismatch.

| Modality dir | On-disk before | On-disk after | Loader sees | Decoder-side change? |
|---|---|---|---|---|
| `rgb/`, `fisheye/`, `ego/rgb/` | `*.png` (8-bit) | `*.webp` (lossless) | identical uint8 array | none — `PIL.Image.open` / `cv2.imread` sniff headers |
| `depth/` | `*.png` (16-bit I;16) | `*.png` re-encoded at level 9 | identical uint16/float32 array | none — same file format |
| `sam2/masks/`, `sam2/masks_verified/` | `*.png` (palette / I;16) | `*.png` re-encoded at level 9 | identical int32 instance IDs, mode preserved | none — same file format |
| `cam_pose/` | `*.npz` ({position, orientation}) | `*.npy` ((7,) float64) | identical 4×4 SE(3) matrix via backward-compatible `_load_pose` dispatch | one-line suffix dispatch already in `loader.py:_load_pose` |
| anything else | unchanged | hard-linked | unchanged | none |

The per-modality opt-outs are `--skip-rgb-webp`,
`--skip-png-recompress`, `--skip-cam-pose`.

`mock` is the synthetic-dataset generator for local testing; `scan` is
the read-only inventory report. Both have flags in `cli.py`.

---

## 5. What's still TODO

Two items remain. Both are documented in
`dataset_hub/README.md` §9 and have explicit acceptance criteria.

### TODO 4 — Preview subset for HF Dataset Viewer

Add `preview.py`:
- CLI: `rpx … preview --staging STAGE [--auto | --config preview.yaml]`
- `--auto`: pick 1 frame per tertile from 5 MOS scenes + 10 SOS scenes
- For each curated frame: decode from tar, overlay mask, resize to 512×512
- Write JPEGs to `<staging>/preview/<nnn>_<scene>_<phase>_<frame>.jpg`
- Also write `<staging>/preview/thumbnails.parquet` with embedded JPEG
  bytes (HF Dataset Viewer auto-detects Parquet with image columns)
- Target total size < 50 MB
- Effort: ~2 h

### TODO 5 — Python loader / iterator

Add `loader.py`:
- `DatasetHubLoader(download_result, order="sequential"|"shuffle(seed=N)")`
- Opens each relevant tar with `tarfile.open(…, "r|")` for streaming
- Yields `Sample(rgb=PIL, depth=np.ndarray, masks=…, cam_pose=…)`
- Honours manifest's `has_<modality>` columns
- Works for both MOS (with split) and SOS recipes
- Torch/jax-agnostic — wrappable into a DataLoader in 3 lines
- Effort: ~3 h

---

## 6. Status quick reference

| | |
|---|---|
| Branch | `jishnu/benchmark-code-skeleton` |
| Last commit | `6405145` ("dataset hub: rewrite README for the team") |
| Tests | 435 passing (`cd benchmark && pytest -q`) |
| Dataset-hub tests only | `pytest -q tests/test_dataset_hub_*.py` (~120 tests, ~20 s) |
| HF repo | `IRVLUTD/RPX` (private, nothing uploaded yet) |
| Python | 3.10+ |
| Extras | `'rpx-benchmark[hub]'` → `huggingface_hub[hf_xet]>=0.26`, `pyarrow>=14` |

---

## 7. Commit history on the branch (dataset-hub slices only)

```
6405145  dataset hub: rewrite README for the team
0522bfd  dataset hub: TODO 3 — Croissant staging
1556f2b  dataset hub: TODO 2 — HF dataset card generator
46128a9  dataset hub: TODO 1 — splits staging
ceadcce  dataset hub: cam_pose moved to versioned label; fisheye L/R modeled
3d832b1  dataset hub: team workflow + TODO checklist for v1 release
ae4c487  dataset hub: objects_meta/ layer for per-object questionnaire dedup
7941ce4  RPX dataset hub: scanner, packer, manifest, uploader, downloader + wireframe
```

Read any commit message with `git show <sha>` for the design notes that
went with each slice.

---

## 8. Open questions the team still needs to answer on the target system

1. **Actual size** (first `scan` report). Earlier estimates: ~89 GB
   (890 MB/scene × 100) vs. ~890 GB. Scan settles it.
2. **Exact fisheye layout inside `fisheye/`**. Confirmed by user as
   `fisheye/{left,right}/00000.png` with synced filenames; the packer
   tars whatever's there, so no code change needed.
3. **Filename padding**: 5-digit (`00000.png`) or 6-digit (`000000.png`).
   Packer is padding-agnostic — this only affects wireframe docs.
4. **How to regenerate `scene_splits.json` against the new `scene<N>`
   naming**. Existing file uses old naming (`scene11.ecss.4f.sofa`).
   Run the difficulty-splits builder against the re-organised source
   (the builder is paper-methodology code kept under the local-only
   workspace, not shipped with the toolkit).

---

## 9. How to resume

New conversation / new teammate:

```bash
git checkout jishnu/benchmark-code-skeleton && git pull
cd benchmark/rpx_benchmark/dataset_hub
less README.md        # team guide — start here
less HANDOFF.md       # this file — fuller context
cd ../../.. && pytest -q tests/test_dataset_hub_*.py    # confirm green
```

To pick up a TODO: look at §5 above, pick TODO 4 or 5, read the
acceptance criteria, write the module + a test file in the same style
as the existing ones (e.g., `packer.py` + `test_dataset_hub_packer.py`
are a good template). Wire into `__init__.py` exports and add a CLI
subcommand in `cli.py`.

Commit style: `dataset hub: <short summary>` with a body that lists
the new module(s), what the CLI hook does, and the test count.

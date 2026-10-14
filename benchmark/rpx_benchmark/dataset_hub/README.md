# RPX dataset hub

This package builds and publishes [IRVLUTD/RPX](https://huggingface.co/datasets/IRVLUTD/RPX) from the
raw captures (about 890 GB), and lets users download only the shards a task and split need.
Maintainers use every command below; users only need `download` and `verify`.

<p align="center">
  <a href="#commands">Commands</a> ·
  <a href="#publishing-a-release">Publishing a release</a> ·
  <a href="#hub-layout">Hub layout</a> ·
  <a href="#updating-the-dataset">Updating the dataset</a> ·
  <a href="#troubleshooting">Troubleshooting</a>
</p>

All commands run as `python -m rpx_benchmark.dataset_hub.cli <command>`. Install with
`pip install -e 'benchmark[hub]'` and log in with `hf auth login` before uploading.

## Commands

| Command | What it does |
|:--|:--|
| `scan ROOT` | inventory a capture tree (read-only) |
| `mock --out DIR` | generate a small synthetic capture tree |
| `prepare-ego` | arrange raw GoPro captures for `scan` and `pack` |
| `lossless-convert` | verified lossless re-encoding of a capture tree |
| `pack` | one tar per scene, phase and modality, plus `objects_meta/` |
| `manifest` | frame index, checksums and per-task manifests |
| `stage-splits` | copy `benchmark/data/splits/` into staging |
| `dataset-card` | generate a dataset card (new repos only) |
| `stage-croissant` | fill in the Croissant template (new repos only) |
| `upload` | push the staging tree, resumably |
| `download --task T --split S` | fetch only the shards a recipe needs |
| `verify SNAPSHOT` | check extracted files against their SHA-256 |

Run any command with `--help` for the full set of flags. A few details matter in practice:

- `scan` reports scenes, counts and bytes per modality, plus a `skipped` list of stray top-level
  entries and loose files.
- `prepare-ego` writes `<DATA>/ego/<scene_id>/0/`.
- `lossless-convert --src DATA --out DATA_v2` re-encodes RGB and fisheye PNGs as lossless WebP,
  recompresses depth and mask PNGs at level 9, and turns per-frame `cam_pose` `.npz` files into
  `.npy`. Every frame is decoded and compared with the source; the first mismatch aborts the run.
- `manifest --src DATA --staging STAGE --splits FILE` writes `current.json` and both tar and
  per-file checksums.
- `stage-croissant` also fills in the repo URL, version and citation.
- `upload --dry-run` reports files and bytes without touching the network. New repos are private
  unless you pass `--public`.
- `download` task names are defined in `recipes.py`; `verify` reads
  `manifest/file_checksums.json`.

## Publishing a release

`DATA` is the capture root (the directory holding `mos/`, `sos/` and `ego/`), and `STAGE` is a
scratch directory. Packing the full set needs about 2 TB of scratch space, and the lossless
conversion takes 6–9 hours on 8 workers.

```bash
CLI="python -m rpx_benchmark.dataset_hub.cli"

# Expect 100 MOS scenes; check the names and the skipped list.
$CLI scan /data/rpx
$CLI lossless-convert --src /data/rpx --out /data/rpx_v2 --workers 8
$CLI pack --src /data/rpx_v2 --staging STAGE --overwrite

# Updating the existing repo: start from its current.json so keys the toolkit
# doesn't write (manifests, sos, mos, metadata_versions) survive the merge.
hf download IRVLUTD/RPX --repo-type dataset \
    --include "manifest/current.json" --local-dir STAGE

$CLI manifest --src /data/rpx_v2 --staging STAGE \
    --splits benchmark/data/splits/scene_splits.json
$CLI stage-splits --staging STAGE --overwrite

$CLI upload --staging STAGE --repo-id IRVLUTD/RPX --dry-run
$CLI upload --staging STAGE --repo-id IRVLUTD/RPX --revision v2-webp
```

`manifest` refuses to finish without `--splits`: without split labels it would produce no per-task
manifests, and `download` could not match any scenes.

Do not run `dataset-card` or `stage-croissant` against IRVLUTD/RPX. The live card and Croissant file
carry hand-written `configs:` blocks for the Hub viewer that the generators don't reproduce. The card
is versioned separately in [`tools/dataset/hf_card/`](../../../tools/dataset/hf_card/README.md).

Uploading to a new revision such as `v2-webp` leaves earlier revisions intact for anyone who has
already benchmarked against them. The default branch moves only once the new revision has been
checked.

After uploading, check a download from a clean cache:

```bash
HF_HOME=/tmp/hf_check python -m rpx_benchmark.dataset_hub.cli download \
    --task segmentation --split easy --cache-dir /tmp/hf_check
python -m rpx_benchmark.dataset_hub.cli verify <snapshot-dir>
```

## Source layout

The capture tree must follow the layout in
[`benchmark/templates/rpx_capture_wireframe/`](../../templates/rpx_capture_wireframe/README.md). In
short: multi-object scenes are `mos/scene<N>/<phase>/`, single-object scans are
`sos/<object_name>/0/` where the directory name is the canonical object ID, and frames are named
`00000.png` and aligned across modalities.

## Hub layout

```
IRVLUTD/RPX/
├── README.md                            dataset card
├── rpx_croissant.json                   Croissant metadata
├── manifest/
│   ├── frames_v1.parquet                one row per frame
│   ├── current.json                     default label versions and file extensions
│   ├── checksums.json                   SHA-256 per tar
│   └── file_checksums.json              SHA-256 per extracted file
├── manifests/<recipe>/<split>.json      per-task, per-split manifests
├── splits/                              scene_splits.json and per-tier .txt
├── scenes/<scene_id>/
│   ├── <phase>/                         0 = Clutter, 1 = Interaction, 2 = Clean
│   │   ├── rgb.tar  depth.tar  fisheye.tar
│   │   └── labels/{cam_pose,masks,masks_aux,sam2_meta}/v1.tar
│   └── ego/                             egocentric capture of the same scene
├── objects/<object_id>/0/               single-object scans, same modality tars
└── objects_meta/
    ├── _index.json
    └── <object_id>/questionnaire.json   one per object, shared by MOS and SOS
```

There is one tar per modality rather than one file per frame, so a task downloads a few hundred tars
instead of the whole repo. Raw modalities are sealed once; labels are versioned under
`labels/<name>/v<N>.tar`, and `current.json` names the default version of each. The packer sorts tar
members, zeroes mtimes and normalizes ownership, so re-packing unchanged data gives byte-identical
tars that the Hub doesn't re-upload.

## Updating the dataset

### New or revised labels

Re-pack with a new label version, point `current.json` at it, and upload:

```bash
python -m rpx_benchmark.dataset_hub.cli pack --src DATA --staging STAGE \
    --label-version v2 --overwrite
python - <<'PY'
import json, pathlib
p = pathlib.Path("STAGE/manifest/current.json")
c = json.loads(p.read_text())
c["label_versions"]["cam_pose"] = "v2"
p.write_text(json.dumps(c, indent=2))
PY
python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE \
    --repo-id IRVLUTD/RPX --message "v1.2.0: refined cam_pose (v2)"
```

The `v1` tars stay in the repo. Users who need them pass `--label-version cam_pose=v1` to
`download`. Tasks that don't use the changed label keep their cached files.

### New scenes

Add them under `DATA/mos/` or `DATA/sos/` and re-run `pack`, `manifest` and `upload`. Existing tars
are byte-identical, so only the new ones transfer.

### Removing a scene

`huggingface_hub` has no folder-delete call, so delete the scene directory in the Hub's Files and
versions tab, then rebuild and re-upload `manifest/frames_v1.parquet` so the row counts match.

### Release tags

Tag releases (`v1.0.0`, `v1.1.0`, …) in the repo settings so results can be pinned with
`download_for_task(task=..., split=..., revision="v1.0.0")`.

## Croissant

[Croissant](https://mlcommons.org/working-groups/croissant/) is the MLCommons metadata format that
Hugging Face, Kaggle and OpenML read to describe a dataset's fields, splits, license and citation.
The template is `croissant_template.json` in this package. Edit it only when adding a new record
set, for example a new modality.

## Troubleshooting

| Symptom | Fix |
|:--|:--|
| `scan` lists `scence1`, or entries under `skipped` | rename or move them and re-scan |
| `pack`: `refusing to overwrite existing shard` | pass `--overwrite` or empty the staging directory |
| `upload --dry-run` lists `.aux`, `.fls` or similar | stage from `pack` output only |
| `download`: `No scenes matched task=... split=...` | re-run `manifest` with `--splits` |
| an upload stops partway | re-run the same command |

Files such as `.aux` and `.fls` are filtered by `DEFAULT_IGNORE_PATTERNS` when you stage from
`pack` output, and an interrupted upload resumes from the progress kept in `<staging>/.huggingface/`.

<details>
<summary>Code map</summary>

| Module | Role |
|:--|:--|
| `scanner.py` | inventory of a capture tree |
| `lossless_convert.py` | verified lossless re-encoding |
| `ego_layout.py` | arranges egocentric captures |
| `packer.py` | per-modality tars and `objects_meta/` |
| `manifest.py`, `split_manifests.py` | frame index, `current.json`, checksums, per-task manifests |
| `staging.py` | copies split files |
| `dataset_card.py`, `croissant.py` | dataset card and Croissant metadata |
| `uploader.py`, `downloader.py` | resumable upload; per-task download |
| `recipes.py` | which modalities each task needs |
| `mock.py` | synthetic capture trees for tests |
| `cli.py` | command-line entry point |

Tests: `cd benchmark && pytest -q tests/test_dataset_hub_*.py`.

</details>

---

<sub>Part of <a href="../../../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>

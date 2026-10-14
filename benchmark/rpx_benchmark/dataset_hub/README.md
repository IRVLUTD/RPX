# RPX dataset hub

This package builds and publishes [IRVLUTD/RPX](https://huggingface.co/datasets/IRVLUTD/RPX) from the raw
captures (about 890 GB), and lets users download only the shards a task and split need. Maintainers use every
command below; users only need `download` and `verify`.

All commands run as `python -m rpx_benchmark.dataset_hub.cli <command>`. Install with `pip install -e 'benchmark[hub]'`
and log in with `hf auth login` before uploading.

## Commands

| Command | What it does |
|---|---|
| `scan ROOT` | Inventory a capture tree: scenes, counts and bytes per modality, plus a `skipped` list of stray top-level entries and loose files. Read-only. |
| `mock --out DIR` | Generate a small synthetic capture tree for testing. |
| `prepare-ego` | Arrange raw GoPro captures into `<DATA>/ego/<scene_id>/0/` so `scan` and `pack` pick them up. |
| `lossless-convert --src DATA --out DATA_v2` | Re-encode RGB and fisheye PNGs as lossless WebP, recompress depth and mask PNGs at level 9, and turn per-frame `cam_pose` `.npz` files into `.npy`. Every frame is decoded and compared with the source; the first mismatch aborts the run. |
| `pack --src DATA --staging STAGE` | Write one tar per (scene, phase, modality), plus `objects_meta/`. |
| `manifest --src DATA --staging STAGE --splits FILE` | Write the frame index, `current.json`, tar and per-file checksums, and the per-task, per-split manifests. |
| `stage-splits --staging STAGE` | Copy `benchmark/data/splits/` into the staging tree. |
| `dataset-card` | Generate a dataset card `README.md`. Only for a new repo; see below. |
| `stage-croissant` | Copy the packaged Croissant template into the staging tree and fill in the repo URL, version and citation. Only for a new repo. |
| `upload --staging STAGE` | Push the staging tree. Resumable; `--dry-run` reports files and bytes without touching the network. New repos are private unless you pass `--public`. |
| `download --task T --split S` | Fetch only the shards a recipe needs (see `recipes.py` for task names). |
| `verify SNAPSHOT` | Check every extracted file's SHA-256 against `manifest/file_checksums.json`. |

Run any command with `--help` for the full set of flags.

## Publishing a release

`DATA` is the capture root (the directory holding `mos/`, `sos/` and `ego/`), and `STAGE` is a scratch directory.
Packing the full set needs about 2 TB of scratch space, and the lossless conversion takes 6–9 hours on 8 workers.

```bash
python -m rpx_benchmark.dataset_hub.cli scan /data/rpx                # expect 100 MOS scenes; check names and skipped
python -m rpx_benchmark.dataset_hub.cli lossless-convert --src /data/rpx --out /data/rpx_v2 --workers 8
python -m rpx_benchmark.dataset_hub.cli pack --src /data/rpx_v2 --staging STAGE --overwrite

# Updating the existing repo: start from its current.json so keys the toolkit doesn't
# write (manifests, sos, mos, metadata_versions) are preserved by the merge.
hf download IRVLUTD/RPX --repo-type dataset --include "manifest/current.json" --local-dir STAGE

python -m rpx_benchmark.dataset_hub.cli manifest --src /data/rpx_v2 --staging STAGE \
    --splits benchmark/data/splits/scene_splits.json
python -m rpx_benchmark.dataset_hub.cli stage-splits --staging STAGE --overwrite

python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE --repo-id IRVLUTD/RPX --dry-run
python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE --repo-id IRVLUTD/RPX --revision v2-webp
```

`manifest` refuses to finish without `--splits`: without split labels it would produce no per-task manifests, and
`download` could not match any scenes.

Do not run `dataset-card` or `stage-croissant` against IRVLUTD/RPX. The live card and Croissant file carry
hand-written `configs:` blocks for the Hub viewer that the generators don't reproduce. The card is versioned
separately in [`tools/dataset/hf_card/`](../../../tools/dataset/hf_card/README.md).

Uploading to a new revision such as `v2-webp` leaves earlier revisions intact for anyone who has already benchmarked
against them. The default branch moves only once the new revision has been checked.

After uploading, check a download from a clean cache:

```bash
HF_HOME=/tmp/hf_check python -m rpx_benchmark.dataset_hub.cli download \
    --task segmentation --split easy --cache-dir /tmp/hf_check
python -m rpx_benchmark.dataset_hub.cli verify <snapshot-dir>
```

## Source layout

The capture tree must follow the layout in
[`benchmark/templates/rpx_capture_wireframe/`](../../templates/rpx_capture_wireframe/README.md). In short:
multi-object scenes are `mos/scene<N>/<phase>/`, single-object scans are `sos/<object_name>/0/` where the directory
name is the canonical object ID, and frames are named `00000.png` and aligned across modalities.

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
├── splits/                              scene_splits.json, easy.txt, medium.txt, hard.txt
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

There is one tar per modality rather than one file per frame, so a task downloads a few hundred tars instead of the
whole repo. Raw modalities are sealed once; labels are versioned under `labels/<name>/v<N>.tar`, and `current.json`
names the default version of each. The packer sorts tar members, zeroes mtimes and normalizes ownership, so
re-packing unchanged data gives byte-identical tars that the Hub doesn't re-upload.

## Updating the dataset

**New or revised labels.** Re-pack with a new label version, point `current.json` at it, and upload:

```bash
python -m rpx_benchmark.dataset_hub.cli pack --src DATA --staging STAGE --label-version v2 --overwrite
python -c "import json, pathlib; p = pathlib.Path('STAGE/manifest/current.json'); \
           c = json.loads(p.read_text()); c['label_versions']['cam_pose'] = 'v2'; \
           p.write_text(json.dumps(c, indent=2))"
python -m rpx_benchmark.dataset_hub.cli upload --staging STAGE --repo-id IRVLUTD/RPX \
    --message "v1.2.0: refined cam_pose (v2)"
```

The `v1` tars stay in the repo. Users who need them pass `--label-version cam_pose=v1` to `download`. Tasks that don't
use the changed label keep their cached files.

**New scenes.** Add them under `DATA/mos/` or `DATA/sos/` and re-run `pack`, `manifest` and `upload`. Existing tars are
byte-identical, so only the new ones transfer.

**Removing a scene.** `huggingface_hub` has no folder-delete call, so delete the scene directory in the Hub's
Files and versions tab, then rebuild and re-upload `manifest/frames_v1.parquet` so the row counts match.

**Release tags.** Tag releases (`v1.0.0`, `v1.1.0`, …) in the repo settings so results can be pinned with
`download_for_task(task=..., split=..., revision="v1.0.0")`.

## Croissant

[Croissant](https://mlcommons.org/working-groups/croissant/) is the MLCommons metadata format that Hugging Face,
Kaggle and OpenML read to describe a dataset's fields, splits, license and citation. The template is
`croissant_template.json` in this package. Edit it only when adding a new record set, for example a new modality.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `scan` lists a scene called `scence1`, or entries under `skipped` | Rename or move them and re-scan. |
| `pack` says `refusing to overwrite existing shard` | Pass `--overwrite` or start from an empty staging directory. |
| `upload --dry-run` lists `.aux`, `.fls` or similar files | Stage from `pack` output only; `DEFAULT_IGNORE_PATTERNS` filters these. |
| `download` says `No scenes matched task=... split=...` | The manifest has no split labels. Re-run `manifest` with `--splits`. |
| An upload stops partway | Re-run the same command. Progress is kept in `<staging>/.huggingface/`. |

## Code

| Module | Role |
|---|---|
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

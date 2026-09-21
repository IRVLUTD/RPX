# RPX Rerun Visual Checks

This folder contains visual and structural QA helpers for the RPX release.

## Quick RGB-D Segmentation Check

`make_rerun_check.py` samples one representative middle frame from every scene phase in `preview/data_studio_preview.csv`.
By default that is all 300 MOS entries: 100 scenes x 3 phases.

It verifies:

- RGB, depth, and mask shards exist.
- RGB/depth/mask frame counts match.
- RGB/depth/mask image sizes match.
- Depth has nonzero values.
- Masks contain foreground instance IDs.

It writes:

- `out/rpx_visual_check.rrd`: Rerun recording for interactive visual review.
- `out/rpx_visual_contact_sheet.jpg`: contact sheet for a quick scan.
- `out/visual_check_report.json`: machine-readable QA report.

This is useful for a fast Data Studio / segmentation preview sanity check, but it is not the full release check.

## Full All-Modality Release Check

`check_all_modalities.py` checks the dataset contract across MOS and SOS:

- 300 MOS scene-phases and 70 SOS selected objects.
- RGB, depth, masks, fisheye left/right, camera pose, masks_aux, and SAM2 metadata shards.
- Frame-count alignment and numbered filename sequences.
- Decoded image samples from each visual stream.
- Camera pose NPZ fields, finite values, and quaternion norms.
- MOS mask-object joins against `manifest/mos_mask_object_map_v1.csv`.
- SOS object metadata and questionnaires.
- ESD split consistency, Data Studio preview images, release assets, and parquet row counts.

By default it decodes first/middle/last frames for every aligned stream. Use `--sample-policy all` only when you want a much slower exhaustive decode pass.

It writes:

- `out_all/rpx_all_modalities_visual_check.rrd`: Rerun recording for scanning each scene/object unit.
- `out_all/all_modalities_contact_sheet.jpg`: RGB/depth/mask/fisheye overview for all checked units.
- `out_all/all_modalities_report.json`: machine-readable QA report.
- `out_all/all_modalities_summary.md`: short human-readable summary.

## Install

From the repo root:

```bash
python -m venv /tmp/rpx-rerun-venv
/tmp/rpx-rerun-venv/bin/pip install -r rerun_check/requirements.txt
```

If starting from a fresh machine, first use the published RPX Quick Start API:

```bash
pip install "rpx-benchmark[hub]"
hf auth login
python rerun_check/download_rgbd_segmentation.py --repo-id anonymous/RPX
```

## Run

From the repo root:

```bash
/tmp/rpx-rerun-venv/bin/python rerun_check/make_rerun_check.py
```

Run the full all-modality release check:

```bash
/tmp/rpx-rerun-venv/bin/python rerun_check/check_all_modalities.py
```

Open the Rerun recording:

```bash
/tmp/rpx-rerun-venv/bin/rerun rerun_check/out/rpx_visual_check.rrd
```

Open the full all-modality Rerun recording:

```bash
/tmp/rpx-rerun-venv/bin/rerun rerun_check/out_all/rpx_all_modalities_visual_check.rrd
```

For a smaller smoke test:

```bash
/tmp/rpx-rerun-venv/bin/python rerun_check/make_rerun_check.py --limit 10
/tmp/rpx-rerun-venv/bin/python rerun_check/check_all_modalities.py --limit 10
```

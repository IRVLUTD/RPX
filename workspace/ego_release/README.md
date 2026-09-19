# RPX Ego Release Prep

This folder contains the staging script for the upcoming egocentric MOS view.
It does not recompute ESD. Ego is treated as an auxiliary view under each MOS
scene:

```text
scenes/<scene_id>/ego/
```

## Expected Source Shape

The script searches for capture directories with this shape:

```text
<ego_source>/<scene_number>/<capture>/
├── rgb/*.png
├── depth/*.png
├── fisheye/left/*.png
├── fisheye/right/*.png
├── cam_pose/*.npz
└── sam2/
    ├── masks/*.png
    ├── mask_to_object.json
    └── ...
```

The first numeric ancestor is mapped through
`manifest/scene_name_mapping_v1.csv`, so `ego_example/2/2` becomes
`scenes/scene002/ego`.

## Generate Upload-Ready Files

From the repo root:

```bash
python -m venv /tmp/rpx-ego-venv
/tmp/rpx-ego-venv/bin/pip install -r rerun_check/requirements.txt huggingface_hub
/tmp/rpx-ego-venv/bin/python ego_release/prepare_ego_release.py \
    --ego-source-root ego_example \
    --apply \
    --overwrite
```

The generated HF-facing files are:

- `scenes/<scene_id>/ego/*.tar`
- `manifest/ego_frames_v1.csv`
- `manifest/ego_frames_v1.parquet`
- `manifest/frames_v2.parquet`
- `manifest/mos_ego_mask_object_map_v1.csv`
- `manifest/mos_ego_mask_object_map_v1.parquet`
- `preview/ego_preview.csv`
- `preview/ego_preview.parquet`
- `preview/mos_phase_preview.parquet`

## Verify

```bash
/tmp/rpx-ego-venv/bin/python rerun_check/check_all_modalities.py \
    --kinds mos,sos,ego
```

Open the generated Rerun recording:

```bash
/tmp/rpx-ego-venv/bin/rerun rerun_check/out_all/rpx_all_modalities_visual_check.rrd
```

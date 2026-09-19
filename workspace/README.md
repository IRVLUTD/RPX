---
license: cc-by-4.0
pretty_name: "RPX: Robot Perception X"
task_categories:
- image-segmentation
- depth-estimation
- object-detection
- visual-question-answering
language:
- en
tags:
- robotics
- embodied-ai
- rgb-d
- benchmark
- perception
- manipulation
- stereo
- tabular
- image
- video
size_categories:
- 100B<n<1T
configs:
- config_name: multi_object
  description: "MOS phase records, tertile-cut by Effort-Stratified Difficulty."
  default: true
  data_files:
  - split: "easy"
    path: "splits/easy.parquet"
  - split: "medium"
    path: "splits/medium.parquet"
  - split: "hard"
    path: "splits/hard.parquet"
- config_name: mos_phase_preview
  description: "Data Studio media table with one video-layout RGB/depth/mask image preview per MOS phase."
  data_files:
  - split: "preview"
    path: "preview/mos_phase_preview.parquet"
- config_name: ego_frames
  description: "Frame-level manifest rows for optional egocentric MOS views. Ego does not contribute to ESD."
  data_files:
  - split: "ego"
    path: "manifest/ego_frames_v1.parquet"
- config_name: ego_preview
  description: "Data Studio media table for egocentric MOS RGB/depth/mask/fisheye previews."
  data_files:
  - split: "preview"
    path: "preview/ego_preview.parquet"
- config_name: single_object
  description: "SOS selected-object catalog; one 360 collection per object, no difficulty split."
  data_files:
  - split: "objects"
    path: "manifest/selected_sos_objects_v1.parquet"
---

# RPX: Robot Perception X

A real-world RGB-D benchmark for evaluating robot perception under
embodied deployment conditions.

* Code: https://github.com/IRVLUTD/RPX

<video controls muted loop playsinline width="100%" src="https://huggingface.co/datasets/IRVLUTD/RPX/resolve/main/assets/rpx-jumbotron.mp4?v=23e835e"></video>

## Dataset at a glance

| | |
|---|---|
| Multi-object scenes (MOS) | **100** (`scene001` to `scene100`, 3 ESD phases each: clutter / interaction / clean) |
| Egocentric MOS views | Schema-ready optional `scenes/<scene_id>/ego/` captures tracked separately from ESD |
| Single-object scenes (SOS) | **70 selected objects** (one 360 collection per object) |
| Frame manifest rows | **110,000** in `frames_v1` (75,000 MOS + 35,000 selected SOS); `frames_v2` adds optional ego rows |
| MOS mask-object rows | **2,100** local mask IDs mapped to global object IDs |

Scene renames are documented in `manifest/scene_name_mapping_v1.csv`,
mapping original scene names to `scene001` through `scene100`.

The Hugging Face Dataset Viewer exposes `multi_object` as structured split
tables and `mos_phase_preview` as a Data Studio image table for normal MOS
phases. The `ego_frames` and `ego_preview` configs are kept separate for the
upcoming egocentric MOS view and do not affect ESD splits.

## Effort-Stratified Difficulty (ESD)

ESD is the difficulty protocol used to split multi-object scenes. Each
`sceneXXX.phaseY` receives an RPX Difficulty Score (`rpx_ds`); higher scores
indicate harder perception conditions. The score is produced by
`effort_stratified_v1` using `primary_method: mean_pn`, a weighted aggregate of
the normalized feature set recorded in `splits/scene_splits.json`.

The released split files expose two levels:

| level | where | construction |
|---|---|---|
| phase-level splits | `splits/easy.txt`, `medium.txt`, `hard.txt` and CSV copies | all 300 scene phases are sorted by `rpx_ds` and tertile-cut into 100 easy, 100 medium, 100 hard entries |
| scene-level tiers | `splits/scene_splits.json` | each scene score is the mean of its three phase scores, then 100 scenes are tertile-cut into 33 easy, 33 medium, 34 hard scenes |

Because scene tiers are aggregated, a hard scene can still contain an easy
phase. Use the phase-level split when downloading or evaluating MOS tasks.
Egocentric `ego` views are auxiliary MOS captures and do **not** contribute to
ESD scores, scene tiers, or easy/medium/hard split membership.

The ESD feature weights are fully accounted for in
`splits/scene_splits.json`: 27 feature names, 27 weights, no missing weights,
and total weight `1.0`. `iter_mean` and `iter_max` each carry weight `0.125`;
the other 25 features each carry weight `0.03`.

| feature group | features |
|---|---|
| refinement effort | `iter_mean`, `iter_max` |
| object complexity | `obj_mean`, `obj_std`, `obj_consist` |
| occlusion | `occ_mean`, `occ_p90`, `occ_heavy` |
| depth quality | `depth_invalid`, `depth_invalid_mask`, `depth_std`, `depth_std_mask` |
| image appearance | `specular`, `dark` |
| mask/visibility stability | `area_cv`, `area_drop`, `vis_instability` |
| motion and trajectory | `trans_mean`, `trans_p90`, `rot_mean`, `rot_p90`, `jerk` |
| fisheye quality | `fisheye_dark`, `fisheye_bright`, `fisheye_sharpness`, `fisheye_corr`, `fisheye_texture` |

The split tables include the final `rpx_ds`, `difficulty`, `scene_tier`, and
`scene_score` fields for Dataset Viewer filtering. Raw per-feature values are
not duplicated in those tables; `splits/scene_splits.json` is the source of
truth for the score/tier assignments and the feature/weight provenance.

## Modality inventory

The table below describes this cleaned release. `cam_pose_icp` is not included
in this cleaned release. Use `manifest/frames_v1.parquet` and the identity
manifests under `manifest/` as the source of truth for the v1 MOS/SOS release;
use `manifest/frames_v2.parquet` when optional ego rows are present.

| modality | files | bytes |
|---|---:|---:|
| `cam_pose` | 110,000 | 163.3 MB |
| `depth` | 110,000 | 16.9 GB |
| `fisheye` | 220,000 | 63.4 GB |
| `rgb` | 110,000 | 41.0 GB |
| `masks` | 110,000 | 341.7 MB |

## Quick start

```bash
pip install "rpx-benchmark[hub]"
hf auth login
```

```python
from rpx_benchmark.dataset_hub import download_for_task

# Pull just RGB + masks for the Easy difficulty tier — never the whole repo.
res = download_for_task(task="segmentation", split="easy",
                          repo_id="IRVLUTD/RPX")
print(res.local_dir, res.matched_scenes)
```

```bash
# Or from the CLI:
python -m rpx_benchmark.dataset_hub.cli download \
    --task segmentation --split easy \
    --repo-id IRVLUTD/RPX
```

A subsequent call for a different task on the same split (e.g.
`relative_pose`) reuses the cached RGB tars and only fetches the new
modality (`cam_pose`) as the delta.

Egocentric views are not difficulty-split. They are represented as a separate
view under `scenes/<scene_id>/ego/` when released, with their own frame
manifest and local mask-to-object map. The intended benchmark API contract is a
view-aware call such as `download_for_task(task="ego_segmentation",
split="ego", ...)` or `download_for_task(task="segmentation", view="ego",
split="ego", ...)`.

## Repo layout

```
IRVLUTD/RPX/
├── manifest/
│   ├── frames_v1.parquet                 # v1 per-frame metadata for MOS phases + SOS
│   ├── frames_v2.parquet                 # v2 metadata including optional ego rows
│   ├── ego_frames_v1.csv/parquet         # frame metadata for scenes/<scene_id>/ego
│   ├── scene_name_mapping_v1.csv         # original scene names to scene001..scene100
│   ├── selected_sos_objects_v1.csv       # selected 70-object SOS catalog
│   ├── selected_sos_objects_v1.parquet   # Dataset Viewer SOS catalog table
│   ├── object_catalog_v1.json            # SOS object/global-ID catalog
│   ├── mos_raw_mask_object_map_v1.csv    # MOS local mask IDs from sam2 metadata
│   ├── mos_mask_object_map_v1.csv        # MOS local mask IDs joined to SOS global IDs
│   ├── mos_mask_object_map_v1.parquet    # parquet copy of the MOS phase map
│   ├── mos_ego_mask_object_map_v1.csv    # ego local mask IDs joined to global IDs
│   ├── mos_ego_mask_object_map_v1.parquet
│   └── current.json                      # default version per label modality
├── splits/
│   ├── scene_splits.json
│   ├── easy.txt  medium.txt  hard.txt      # phase-level split IDs
│   ├── easy.csv  medium.csv  hard.csv      # human-readable split tables
│   └── easy.parquet  medium.parquet  hard.parquet  # Dataset Viewer split tables
├── preview/
│   ├── mos_phase_preview.parquet           # Dataset Viewer image media table for MOS phases
│   ├── mos_phase_preview.csv               # source index for MOS phase previews
│   ├── ego_preview.parquet                 # Dataset Viewer image media table for ego views
│   ├── ego_preview.csv                     # source index for ego previews
│   ├── data_studio_preview.csv             # legacy name for the MOS phase preview index
│   └── image_examples/preview/             # source JPEGs for the media preview
├── scenes/<scene_id>/<phase>/                     # MOS ESD phases: 0, 1, 2
│   ├── rgb.tar  depth.tar  fisheye.tar
│   └── labels/{cam_pose,masks,masks_aux,sam2_meta}/v1.tar
├── scenes/<scene_id>/ego/                         # optional egocentric MOS view
│   ├── rgb.tar  depth.tar  fisheye.tar
│   └── labels/{cam_pose,masks,masks_aux,sam2_meta}/v1.tar
├── objects/<object_id>/0/                         # SOS
│   └── (same modality structure)
├── objects_meta/                                  # questionnaire dedup
│   ├── _index.json
│   └── <object_id>/questionnaire.json
└── README.md   ←  this file
```

## Object identity manifests

SOS objects use dataset-wide IDs from `manifest/selected_sos_objects_v1.csv`
and `manifest/object_catalog_v1.json`.

| field | meaning |
|---|---|
| `global_object_id` | New integer ID, 1 to 70 |
| `source_catalog_id` | Original catalog/PDF ID, kept as a string such as `88.2` |
| `object_id` | Actual folder name, such as `mug.2` |
| `questionnaire_path` | Linked questionnaire under `objects_meta/<object_id>/questionnaire.json` |

MOS masks use local IDs. A `local_mask_id` is only meaningful within one
`scene_id + phase`; it is not a global object ID. The raw source is
`scenes/<scene_id>/<phase_index>/labels/sam2_meta/v1.tar:sam2/mask_to_object.json`.

Use `manifest/mos_mask_object_map_v1.csv` or
`manifest/mos_mask_object_map_v1.parquet` to join MOS masks to SOS objects:

```csv
scene_id,phase,local_mask_id,object_id,global_object_id,source_catalog_id,object_name
scene001,phase0,2,boot.2,11,18.2,boot
```

That row means `scene001/0` mask ID `2` is `boot.2`, whose global object ID
is `11`, with questionnaire
`objects_meta/boot.2/questionnaire.json` and SOS template `objects/boot.2/`.

Ego masks follow the same local-ID rule. Use
`manifest/mos_ego_mask_object_map_v1.csv` or parquet to join
`scenes/<scene_id>/ego/labels/masks/v1.tar` local mask IDs to global object
IDs. Ego local IDs are scoped to `scene_id + ego`.

## Tasks

### Multi-object (use a difficulty split)

| recipe | inputs → labels |
|---|---|
| `monocular_depth` | ['rgb'] → ['depth'] |
| `rgbd_segmentation` | ['depth', 'rgb'] → ['masks'] |
| `segmentation` | ['rgb'] → ['masks'] |
| `relative_pose` | ['rgb'] → ['cam_pose'] |
| `rgbd_relative_pose` | ['depth', 'rgb'] → ['cam_pose'] |
| `stereo_depth` | ['fisheye'] → ['depth'] |
| `object_tracking` | ['rgb'] → ['masks'] |
| `vqa` | ['rgb'] → ['questionnaire', 'vqa'] |

### Single-object (no split — these are object templates)

| recipe | inputs → labels |
|---|---|
| `object_templates` | ['rgb'] → ['masks'] |
| `object_templates_rgbd` | ['depth', 'rgb'] → ['masks'] |
| `object_pose_library` | ['depth', 'rgb'] → ['cam_pose', 'masks'] |

### Egocentric MOS views (no ESD split)

| recipe | inputs → labels |
|---|---|
| `ego_segmentation` | ['rgb'] → ['masks'] |
| `ego_rgbd_segmentation` | ['depth', 'rgb'] → ['masks'] |
| `ego_relative_pose` | ['rgb'] → ['cam_pose'] |
| `ego_stereo_depth` | ['fisheye'] → ['depth'] |

## Label versioning

Labels live at `labels/<name>/v<N>.tar`. Newer versions land at new
paths; old versions stay reachable for reproducibility.

| modality | current version |
|---|---|
| `masks` | `v1` |
| `masks_aux` | `v1` |
| `sam2_meta` | `v1` |
| `cam_pose` | `v1` |

To pin to a specific version:

```python
download_for_task(
    task="relative_pose", split="easy", repo_id="IRVLUTD/RPX",
    label_versions={"cam_pose": "v1"},   # don't auto-upgrade to v2
)
```

## Citation

```bibtex
@misc{rpx2026,
    title  = {RPX: Robot Perception X — A real-world RGB-D benchmark for
              embodied perception},
    author = {IRVL UT Dallas},
    year   = 2026,
    url    = {https://huggingface.co/datasets/IRVLUTD/RPX},
}
```

## License

Released under the **cc-by-4.0** license.

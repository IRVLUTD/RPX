---
license: cc-by-4.0
pretty_name: "RPX: Robot Perception X"
task_categories:
- depth-estimation
- image-segmentation
- object-detection
- visual-question-answering
- image-to-image
language:
- en
tags:
- robotics
- embodied-ai
- rgb-d
- benchmark
- perception
- manipulation
size_categories:
- 10K<n<100K
paperswithcode_id: rpx-robot-perception-x
configs:
- config_name: default
  description: Per-task manifests under manifests/<task>/<difficulty>.json
---

# RPX: Robot Perception X

RPX is a unified real-world RGB-D benchmark for evaluating robot perception under embodied deployment conditions. 75,000 frames across 100 indoor scenes, captured with an Intel RealSense D435 + T265 rig under a three-phase reconfiguration protocol (clutter → human interaction → clean). Ground-truth annotations include instance masks, tracklets, metric depth, 6-DoF camera pose, and language attributes — enabling evaluation of 10 perception tasks on identical scenes.

Difficulty splits (Easy / Medium / Hard) are derived per `(scene, phase)` from the Effort-Stratified Difficulty (ESD) score and are *logical views*, not physical copies: a frame referenced by Hard is the same file on disk as that frame in Easy.

## Quick start

```bash
pip install "rpx-benchmark[hub]"
```

```python
import rpx_benchmark as rpx

# Fetches only RGB + depth for scenes in the Hard split.
ds = rpx.load("monocular_depth", "hard")
for batch in ds:
    sample = batch[0]
    rgb = sample.rgb                      # uint8 H×W×3
    depth = sample.ground_truth.depth_map # float32 H×W metres
```

Task-aware, incremental downloads share the HuggingFace content-addressed cache — switching to `visual_grounding` on the same split only pulls the QA JSON, reusing the already-cached RGB/depth.

```python
# No redundant download — reuses cached RGB from the call above.
qa = rpx.load("visual_grounding", "hard")
```

CLI:

```bash
rpx ls
rpx info     --task monocular_depth --split hard
rpx download --task monocular_depth --split hard
```

## Supported tasks

| Task                      | Modalities fetched                                  |
|---------------------------|-----------------------------------------------------|
| `monocular_depth`         | rgb, depth                                          |
| `sparse_depth`            | rgb, depth, sparse samples                          |
| `object_segmentation`     | rgb, mask                                           |
| `object_detection`        | rgb, mask, tracklets                                |
| `open_vocab_detection`    | rgb, mask, tracklets, questionnaires                |
| `object_tracking`         | rgb, mask, tracklets                                |
| `relative_camera_pose`    | rgb (pair), pose                                    |
| `novel_view_synthesis`    | rgb (source + target), depth, pose                  |
| `visual_grounding`        | rgb, questionnaires, spatial_qa                     |
| `keypoint_matching`       | rgb (pair), pose                                    |

All ten tasks are evaluated on identical scenes; differences in scores therefore isolate task- and model-specific failure modes from scene-level confounds.

## Dataset structure

```
rpx-benchmark/
├── README.md                            # this file
├── metadata/
│   ├── scenes.parquet                   # scene_id, environment, categories
│   └── esd_scores.parquet               # (scene, phase) → ESD features + difficulty
├── manifests/                           # logical views (not duplicated data)
│   └── <task>/{easy,medium,hard}.json
└── scenes/scene_000/                    # 100 scenes total
    ├── 0/   (clutter)
    │   ├── rgb/00000.png ... .png       # 640×480 uint8
    │   ├── depth/00000.png              # 16-bit mm (D435 raw)
    │   ├── mask/00000.png               # int32 instance IDs
    │   ├── pose/00000.npz               # position + orientation_xyzw
    │   ├── tracklets.json               # per-phase track IDs
    │   ├── questionnaires.json          # per-object FewSOL attributes
    │   ├── spatial_qa.json              # spatial awareness Q&A
    │   └── general_qa.json              # general scene Q&A
    ├── 1/   (interaction)
    └── 2/   (clean)
```

### File formats

- **RGB**: PNG, 640×480 uint8.
- **Depth**: 16-bit PNG in **millimetres**. `0` = invalid (sensor miss). The toolkit converts to float32 metres on load.
- **Mask**: PNG; pixel values are instance IDs consistent with `tracklets.json` within a `(scene, phase)`.
- **Pose**: `.npz` with keys `position` (`[x, y, z]` metres) and `orientation` (`[x, y, z, w]` quaternion, T265 convention). The loader converts to 4×4 SE(3) float64 camera-to-world.
- **Labels** (`tracklets.json`, `*_qa.json`, `questionnaires.json`): one file per `(scene, phase)` describing all frames in that phase.

### Splits

- **Scene splits** (`metadata/splits/{train,val,test}.json`): 70/15/15 at the scene level; no scene crosses splits.
- **ESD difficulty splits** (`manifests/<task>/{easy,medium,hard}.json`): computed per `(scene, phase)` from annotation effort, occlusion, depth invalidity, visibility instability, and jerk. See the paper §4 for the full RPX-DS formulation.

## Capture protocol

Each scene is captured in three phases with the same sensor trajectory style:

1. **Clutter** (`0`): dense object arrangement, significant inter-object occlusion — baseline operating state.
2. **Interaction** (`1`): human operator manipulates objects; introduces hand–object contact and transient occlusion.
3. **Clean** (`2`): same objects re-organised sparsely — within-scene control baseline. Any metric difference vs. Clutter is attributable to arrangement, not scene identity.

Full capture takes ~10 minutes per scene, ensuring lighting consistency across phases.

## Data collection & responsible AI

- **Sensor rig**: hand-held Intel RealSense D435 (RGB-D) + T265 (6-DoF VIO).
- **Annotation**: SAM2 auto-masks with iterative human refinement; refinement iteration count feeds the ESD score.
- **Consent**: all human hands appearing in interaction phases belong to lab personnel who provided informed consent for release. No faces, biometrics, or PII are captured.
- **Bias / scope**: indoor tabletop and room-scale only; 0.3–3 m depth range; household and laboratory object categories.

## License

Dataset released under **CC BY 4.0**. Benchmark toolkit released under **MIT** at <https://github.com/IRVLUTD/RPX>.

## Citation

```bibtex
@inproceedings{rpx2026,
  title     = {RPX: A Unified Benchmark for Robot Perception under Embodied Deployment Conditions},
  author    = {Anonymous},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2026}
}
```

A machine-readable [Croissant](https://mlcommons.org/croissant) description is also included with the paper submission.

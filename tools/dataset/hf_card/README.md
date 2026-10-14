---
license: cc-by-4.0
pretty_name: "RPX: Same Scene, Different Story"
task_categories:
- image-segmentation
- depth-estimation
- object-detection
- robotics
- visual-question-answering
language:
- en
tags:
- robotics
- embodied-ai
- rgb-d
- benchmark
- perception
- stereo
- video
- parquet
size_categories:
- 100K<n<1M
configs:
- config_name: multi_object
  description: "MOS scene-phases grouped by the paper's difficulty tiers (33 Easy, 33 Medium, 34 Hard scenes, all three phases each)."
  default: true
  data_files:
  - split: easy
    path: splits/easy.parquet
  - split: medium
    path: splits/medium.parquet
  - split: hard
    path: splits/hard.parquet
- config_name: single_object
  description: "The canonical 70-object SOS catalog."
  data_files:
  - split: objects
    path: manifest/selected_sos_objects_v1.parquet
- config_name: mos_preview
  description: "One RGB/depth/mask preview per MOS phase."
  data_files:
  - split: preview
    path: preview/mos_phase_preview.parquet
- config_name: ego_preview
  description: "One RGB/mask preview per egocentric scene."
  data_files:
  - split: preview
    path: preview/ego_preview_v2.parquet
- config_name: vqa_attribute
  data_files:
  - split: train
    path: vqa/attribute.parquet
- config_name: vqa_spatial_bbox
  data_files:
  - split: train
    path: vqa/spatial_bbox.parquet
- config_name: vqa_spatial_binary
  data_files:
  - split: train
    path: vqa/spatial_binary.parquet
- config_name: vqa_incontext_mos_attribute
  data_files:
  - split: train
    path: vqa/incontext_mos_attribute_bbox.parquet
- config_name: vqa_incontext_ego_attribute
  data_files:
  - split: train
    path: vqa/incontext_ego_attribute_bbox.parquet
- config_name: vqa_incontext_mos_spatial
  data_files:
  - split: train
    path: vqa/incontext_mos_spatial_bbox.parquet
- config_name: tracking_object_vocab
  data_files:
  - split: train
    path: tracking/metadata/text_initialization_v1/object_catalog_vocab.parquet
- config_name: tracking_scene_vocab
  data_files:
  - split: train
    path: tracking/metadata/text_initialization_v1/scene_condition_vocab.parquet
---

<div align="center">

<img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/logo.png" alt="RPX" height="72">

## Same Scene, Different Story

### Evaluating Robot Perception Across Scene Phases in the Wild

**Jishnu Jaykumar P<sup>1\*</sup>, Itay Kadosh<sup>2\*</sup>, Narendhiran Vijayakumar<sup>3\*</sup>, Srinanditha Kamath<sup>4</sup>, Sai Haneesh Allu<sup>1</sup>,<br>
Govind Tyagi Rangappa<sup>1</sup>, Animesh Maheshwari<sup>1</sup>, Jikai Wang<sup>1</sup>, Yu Xiang<sup>1</sup>**<br>
<sub><sup>1</sup>UT Dallas &nbsp; <sup>2</sup>UF &nbsp; <sup>3</sup>IIIT Hyderabad &nbsp; <sup>4</sup>UT Austin &nbsp; \*Equal contribution</sub>

[Toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/) |
[Code](https://github.com/IRVLUTD/RPX) |
[PyPI](https://pypi.org/project/rpx-benchmark/) |
[Docker images](https://hub.docker.com/r/irvlutd/rpx) |
[Citation](#citation)

</div>

![RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires](https://huggingface.co/datasets/IRVLUTD/RPX/resolve/main/assets/rpx-overview.webp)

A robot's workspace changes as it works: a table starts cluttered, a hand rearranges it, and it ends clean.
RPX records the same real scenes before, during and after manipulation with robot sensors, and scores six
perception tasks on them with one pipeline. Two numbers put every model on a common scale: phase robustness Φ,
which measures whether quality holds as the scene changes, and worst-phase quality 𝒥<sub>min</sub>, which
measures how good the model is in its weakest phase.

| **100** real scenes | **3 + 1** views | **70** objects |
|:---:|:---:|:---:|
| <sub>50 indoor, 50 outdoor, 20 sites</sub> | <sub>Clutter, Interaction, Clean, plus Ego</sub> | <sub>each also scanned alone in 360°</sub> |
| **133,121** frames | **6** tasks | **65** evaluations |
| <sub>RGB-D, stereo, pose, masks</sub> | <sub>from depth to visual grounding</sub> | <sub>zero-shot, across 49 models</sub> |

In the paper, depth and tracking quality change significantly across phases (p < 0.001), model rankings shift
across tasks, prompting modes and input modalities on the same scenes, and no evaluation reaches both
𝒥<sub>min</sub> ≥ 0.75 and Φ ≥ 0.86.

![What RPX reveals: the same scene in Clutter, Interaction, Clean and egocentric views, with mean quality per view for video depth, tracking and VQA dropping when hands enter; a map of all 65 evaluations by worst-phase quality and phase robustness with none in the golden zone](https://huggingface.co/datasets/IRVLUTD/RPX/resolve/main/assets/rpx-findings.webp)

## Tasks

| Task | Scored on | Metrics | Image |
|:--|:--|:--|:--|
| T1 Image depth | 3 phases | AbsRel, RMSE, SILog, δ<sub>1</sub> | `depth-zipdepth-latest` |
| T2 Video depth | 3 phases | T1's, plus TGM, TGSE | `depth-zipdepth-latest` |
| T3 Object tracking | 3 phases + Ego | HOTA, MOTA, IDSW | `tracking-dam4sam-rpx-latest` |
| T4 Relative camera pose | Clutter, Clean | AUC at 5°, 10°, 20° | `rcpe-monst3r-rpx-latest` |
| T5 VQA | 3 phases + Ego | Acc@0.5, GIoU, center-in-GT | `vqa-vllm` |
| T6 In-context VQA | 3 phases + Ego | Acc@0.5, GIoU, center-in-GT | `vqa-vllm` |

Each task keeps its field's usual metrics. Results are mapped onto a common scene–phase scale, and Φ and
𝒥<sub>min</sub> are computed from that. The [Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md#which-image-to-pull)
lists which model is in which image; every tag is under `irvlutd/rpx`.

## Benchmark your model

The `rpx-benchmark` package downloads the data, runs the paper's protocol and metrics, and computes Φ and
𝒥<sub>min</sub>. You write one function that runs your model, and a per-task adapter connects it to RPX.

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

```python
import numpy as np
import rpx_benchmark as rpx

# 1. Download the task's data
manifest = rpx.download_split("monocular_depth", "easy", max_samples=6)

# 2. Your model: RGB image (H, W, 3) in, depth (H, W) in metres out
def predict(rgb):
    return np.ones(rgb.shape[:2], np.float32)

# 3. Wrap it in the task's adapter and run the benchmark
model = rpx.make_numpy_depth_model(predict, name="my-model")
result, report, paths = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(
    model=model, manifest_path=str(manifest), output_dir="rpx_results/my-model",
))
print(result.aggregated["absrel"], result.aggregated["delta1"])
```

Drop `max_samples` for the full tier, then compute Φ and 𝒥<sub>min</sub> with the toolkit's calculator. The
other tasks work the same way; the [toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/getting-started/)
list each task's download name, adapter and runner. The toolkit pins a dataset commit by default
(`rpx.DEFAULT_REVISION`), so two runs read identical data. There is no need to download the whole 237 GB
repository for one task.

To browse the data without the toolkit:

```python
from datasets import load_dataset

# The paper's Easy tier: 33 scenes × 3 phases
easy = load_dataset("IRVLUTD/RPX", "multi_object", split="easy")
objects = load_dataset("IRVLUTD/RPX", "single_object", split="objects")
```

## Difficulty tiers

Scenes are split into Easy (33), Medium (33) and Hard (34) tiers by annotation effort, measured as the number
of correction passes a scene's masks needed, which serves as a proxy for perceptual ambiguity. A tier always
contains all three phases of its scenes, as in the paper and in every benchmark manifest.

| File | Contents |
|---|---|
| `splits/{easy,medium,hard}.parquet` (`.csv`, `.txt`) | one row per scene–phase of that tier: 99 / 99 / 102 rows |
| `splits/scene_splits.json` | scene lists per tier, scores, per-phase scores, and scoring provenance |
| `manifests/<task>/<tier>.json` | the exact samples each benchmark task evaluates |

Each split row also keeps the phase's own score (`rpx_ds`) and the tier that score alone would give
(`phase_difficulty`), for analyses at phase level. Ego captures are not scored for difficulty; they follow
their scene's tier.

## Contents

| Collection | Coverage | Modalities |
|---|---:|---|
| Multi-object scenes (MOS) | 100 scenes × 3 phases × 250 frames = 75,000 | RGB, metric depth, fisheye stereo, 6-DoF pose, instance masks, tracklets |
| Egocentric view (Ego) | 100 Interaction clips, 23,121 frames | RGB (1920 × 1080), instance masks, tracklets |
| Single-object scans (SOS) | 70 objects × 500 frames = 35,000 | RGB, metric depth, fisheye stereo, pose, masks, questionnaire |
| VQA question bank | 4,529,322 single-image and 749,083 two-image records | attribute, spatial-box and spatial-binary questions |
| Benchmark VQA subset | 31,500 runnable questions | fixed, seeded evaluation manifests |
| Tracking text prompts | 70 objects, 2,739 scene–condition entries | stable IDs, canonical names, fixed prompts |

Capture: Intel RealSense D435 (RGB-D, 640 × 480) with a T265 (fisheye stereo and visual–inertial pose) on a
hand-held rig, plus a head-mounted GoPro during Interaction. Every object keeps one global identity across
phases, views and the single-object catalogue.

<details>
<summary>Repository layout and file formats</summary>

```text
assets/                         dataset-card media
manifest/                       canonical inventories, object catalog and checksums
manifests/<task>/<tier>.json    executable benchmark sample manifests
objects/                        70 single-object scan shard trees
objects_meta/                   per-object metadata and questionnaires
preview/                        lightweight MOS and Ego previews
scenes/                         100 MOS phase trees and 100 Ego view trees
splits/                         difficulty tiers and scoring provenance
tracking/metadata/              text-initialization vocabularies and checksums
vqa/                            full VQA Parquets and benchmark JSONL manifests
```

Large modalities are packed in tar shards. RGB and fisheye images are lossless WebP; depth and masks are PNG;
camera poses are `(7,) float64` NPY vectors `[x, y, z, qx, qy, qz, qw]`. Labels are versioned under
`labels/<name>/v<N>.tar`. `manifest/frames_v1.parquet` is the canonical frame inventory (133,121 records,
with absent Ego depth, fisheye and pose marked explicitly); `frames_v2.parquet` is historical only.

</details>

<details>
<summary>Object and mask identifiers</summary>

| Field | Scope and rule |
|---|---|
| `global_object_id` | canonical integer identity, unique and contiguous from 1 to 70 |
| `source_catalog_id` | original catalog identity, stored as a string; dotted values such as `"18.2"` must stay strings |
| `object_id` | canonical SOS folder key, for example `boot.2` |
| `local_mask_id` / `mask_index` | local to one scene condition; join with scene, view and phase before mapping to a global ID |
| `sample_id` | stable identifier of a VQA record; never derive object identity from it |

The source of truth is `manifest/object_catalog_v1.json`. Join MOS masks through
`manifest/mos_mask_object_map_v1.parquet` and tracking prompts through
`tracking/metadata/text_initialization_v1/scene_condition_vocab.parquet`.

</details>

<details>
<summary>VQA files</summary>

| File | Rows | Purpose |
|---|---:|---|
| `vqa/attribute.parquet` | 3,241,564 | colour, material, function, composition, count and related attribute questions |
| `vqa/spatial_bbox.parquet` | 543,017 | spatial questions with box answers |
| `vqa/spatial_binary.parquet` | 744,741 | left/right and above/below questions |
| `vqa/incontext_mos_attribute_bbox.parquet` | 317,873 | MOS two-image attribute transfer |
| `vqa/incontext_ego_attribute_bbox.parquet` | 113,193 | Ego two-image attribute transfer |
| `vqa/incontext_mos_spatial_bbox.parquet` | 318,017 | MOS two-image spatial reasoning |

The runnable manifest is `vqa/metadata/benchmark_subset_v1/benchmark_available_31500.jsonl`.
`benchmark_pending_4500.jsonl` holds reserved descriptors, not runnable questions: Ego has no depth, so those
spatial questions are not fabricated. A model answers with a target-image box; the evaluator never passes
ground truth to the model.

</details>

<details>
<summary>Reproducibility</summary>

- Pin the dataset commit you used and report it with your results.
- Verify downloads with the SHA-256 manifests under `manifest/`, `vqa/metadata/` and `tracking/metadata/`.
- `manifest/current.json` is the active metadata index.
- Read `source_catalog_id` as text from CSV; prefer Parquet or JSON when types matter.

</details>

## Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across
            Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and
            Kamath, Srinanditha and Allu, Sai Haneesh and Rangappa, Govind Tyagi and
            Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

## License

The RPX dataset is released under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
you may share and adapt it with attribution. The benchmark code is MIT-licensed in the
[GitHub repository](https://github.com/IRVLUTD/RPX).

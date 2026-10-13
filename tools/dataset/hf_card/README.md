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

# RPX: Same Scene, Different Story

### Evaluating Robot Perception Across Scene Phases in the Wild

**Jishnu Jaykumar P\*, Itay Kadosh\*, Narendhiran Vijayakumar\*, Srinanditha Kamath, Sai Haneesh Allu,<br>
Govind Tyagi Rangappa, Animesh Maheshwari, Jikai Wang, Yu Xiang**<br>
<sub>UT Dallas · UF Gainesville · IIIT Hyderabad · UT Austin   \*equal contribution</sub>

[**Toolkit docs**](https://irvlutd.github.io/RPX/toolkit-docs/) ·
[**Code**](https://github.com/IRVLUTD/RPX) ·
[**`pip install rpx-benchmark`**](https://pypi.org/project/rpx-benchmark/) ·
[**Docker images**](https://hub.docker.com/r/irvlutd/rpx) ·
[**Cite**](#citation)

</div>

![RPX overview: the same real scenes captured before (Clutter), during (Interaction) and after (Clean) manipulation plus an egocentric view; six perception tasks evaluated through one pipeline; 100 multi-object scenes in three difficulty tiers; 70 single objects; RGB-D, stereo, pose, masks and questionnaires](https://huggingface.co/datasets/IRVLUTD/RPX/resolve/main/assets/rpx-overview.webp)

**A robot's scene does not hold still.** A tabletop starts cluttered, a hand rearranges it, and it ends clean.
RPX captures **the same real scenes before, during and after manipulation** on robot-grade sensors, and scores
six perception tasks on them through one pipeline. Two numbers summarize every model on a common scale:
**phase robustness Φ** (does quality hold as the scene changes?) and **worst-phase quality 𝒥<sub>min</sub>**
(how good is it at its weakest?).

| 100 | 3 + 1 | 70 | 133,121 | 6 | 65 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| real scenes<br><sub>50 indoor · 50 outdoor · 20 sites</sub> | phases + ego view<br><sub>Clutter · Interaction · Clean · Ego</sub> | everyday objects<br><sub>each scanned alone in 360°</sub> | annotated frames<br><sub>RGB-D · stereo · pose · masks</sub> | tasks<br><sub>depth to grounding</sub> | zero-shot evaluations |

**What it shows.** Depth and tracking change significantly across phases (p < 0.001); model rankings shift
across tasks, prompting modes and input modalities on the very same scenes; and no evaluation reaches both
𝒥<sub>min</sub> ≥ 0.75 and Φ ≥ 0.86.

![What RPX reveals: the same scene in Clutter, Interaction, Clean and egocentric views, with mean quality per view for video depth, tracking and VQA dropping when hands enter; a map of all 65 evaluations by worst-phase quality and phase robustness with none in the golden zone](https://huggingface.co/datasets/IRVLUTD/RPX/resolve/main/assets/rpx-findings.webp)

## Six tasks, one pipeline

| | Task | Scored on | Native metrics | Reference runtime |
|---|---|---|---|---|
| T1 | Image depth | 3 phases | AbsRel · RMSE · SILog · δ<sub>1</sub> | `irvlutd/rpx:depth-zipdepth-latest` |
| T2 | Video depth | 3 phases | T1 metrics + TGM · TGSE | `irvlutd/rpx:depth-zipdepth-latest` |
| T3 | Multi-object tracking | 3 phases + Ego | HOTA · MOTA · IDSW | `irvlutd/rpx:tracking-dam4sam-rpx-latest` |
| T4 | Relative camera pose | Clutter · Clean | AUC@5° · @10° · @20° | `irvlutd/rpx:rcpe-monst3r-rpx-latest` |
| T5 | VQA localization | 3 phases + Ego | Acc@0.5 · GIoU · Center-in-GT | `irvlutd/rpx:vqa-vllm` |
| T6 | In-context VQA | 3 phases + Ego | Acc@0.5 · GIoU · Center-in-GT | `irvlutd/rpx:vqa-vllm` |

Every task keeps its field's native metrics; each result is mapped onto the same scene–phase scale, from which Φ
and 𝒥<sub>min</sub> are computed. The [Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md#which-image-to-pull)
lists which image holds which model.

## Get started in two minutes

```bash
python -m pip install 'rpx-benchmark[hub,schemas]'

# Check the install on all six tasks with synthetic data (CPU, no downloads):
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke --output results/six-task-smoke
```

**Score your own model.** Wrap any callable, local model or API, and the toolkit fetches only the shards that
task needs, runs it, and writes the metrics:

```python
import rpx_benchmark as rpx
from my_model import predict_depth          # RGB array -> depth in metres

model = rpx.make_numpy_depth_model(predict_depth, name="my-depth")
result, report, paths = rpx.run_monocular_depth(
    rpx.MonocularDepthRunConfig(
        model=model, split="easy", repo_id="IRVLUTD/RPX",
        output_dir="rpx_results/my-depth/easy",
    )
)
print(result.aggregated)
```

The same pattern covers video depth, tracking, relative pose and VQA; see the
[toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/). The toolkit pins a dataset commit by default
(`rpx.DEFAULT_REVISION`), so two runs always read identical data. **Do not download the whole 237 GB
repository** for one task.

**Just browse the data:**

```python
from datasets import load_dataset

easy = load_dataset("IRVLUTD/RPX", "multi_object", split="easy")   # the paper's Easy tier: 33 scenes × 3 phases
objects = load_dataset("IRVLUTD/RPX", "single_object", split="objects")
```

## Difficulty tiers

Scenes are stratified into **Easy (33), Medium (33) and Hard (34)** by annotation-refinement effort, the number
of correction passes a scene's masks needed, as a proxy for perceptual ambiguity. A tier always contains **all
three phases of its scenes**, exactly as in the paper and in every benchmark manifest.

| File | Contents |
|---|---|
| `splits/{easy,medium,hard}.parquet` (`.csv`, `.txt`) | one row per scene–phase of that tier: 99 / 99 / 102 rows |
| `splits/scene_splits.json` | scene lists per tier, scores, per-phase scores, and scoring provenance |
| `manifests/<task>/<tier>.json` | the exact samples each benchmark task evaluates |

Each split row also keeps the phase's own score (`rpx_ds`) and the tier that score alone would give
(`phase_difficulty`), for analyses at phase level. Ego captures are not scored for difficulty; they follow
their scene's tier.

## What is included

| Collection | Coverage | Modalities |
|---|---:|---|
| Multi-object scenes (MOS) | 100 scenes × 3 phases × 250 frames = **75,000** | RGB, metric depth, fisheye stereo, 6-DoF pose, instance masks, tracklets |
| Egocentric view (Ego) | 100 Interaction clips, **23,121** frames | RGB (1920 × 1080), instance masks, tracklets |
| Single-object scans (SOS) | 70 objects × 500 frames = **35,000** | RGB, metric depth, fisheye stereo, pose, masks, questionnaire |
| VQA question bank | **4,529,322** single-image + **749,083** two-image records | attribute, spatial-box and spatial-binary questions |
| Benchmark VQA subset | 31,500 runnable questions | fixed, seeded evaluation manifests |
| Tracking text prompts | 70 objects, 2,739 scene–condition entries | stable IDs, canonical names, fixed prompts |

Capture: Intel RealSense D435 (RGB-D, 640 × 480) with a T265 (fisheye stereo and visual–inertial pose) on a
hand-held rig, plus a head-mounted GoPro during Interaction. Every object keeps one global identity across
phases, views and the single-object catalogue.

<details>
<summary><b>Repository layout and file formats</b></summary>

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
<summary><b>Object and mask identifiers</b></summary>

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
<summary><b>VQA files</b></summary>

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
<summary><b>Reproducibility</b></summary>

- Pin the dataset commit you used and report it with your results.
- Verify downloads with the SHA-256 manifests under `manifest/`, `vqa/metadata/` and `tracking/metadata/`.
- `manifest/current.json` is the active metadata index.
- Read `source_catalog_id` as text from CSV; prefer Parquet or JSON when types matter.

</details>

## Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and Kamath, Srinanditha and
            Allu, Sai Haneesh and Rangappa, Govind Tyagi and Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

## License

The RPX dataset is released under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
share and adapt it with attribution. The benchmark code is MIT-licensed in the
[GitHub repository](https://github.com/IRVLUTD/RPX).

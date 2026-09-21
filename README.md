<div align="center">

# RPX

**A real-world RGB-D benchmark for robot perception.**

[![tests](https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=naren%2Fall&label=tests)](https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-toolkit-2563eb)](https://irvlutd.github.io/RPX/toolkit-docs/)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](benchmark/pyproject.toml)
[![dataset](https://img.shields.io/badge/Dataset-IRVLUTD%2FRPX-yellow)](https://huggingface.co/datasets/IRVLUTD/RPX)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

```bash
git clone --branch naren/all https://github.com/IRVLUTD/RPX.git
cd RPX
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e './benchmark[hub,schemas]'
python benchmark/examples/run_depth.py
```

The example checks the installation using a synthetic depth model and local
data. It is CPU-only and downloads nothing. To evaluate actual checkpoints, use
the task-specific environments and commands in
[the benchmark README](benchmark/README.md#reference-models).

| Dataset | Benchmark surface | Reference coverage | Custom models |
|---|---|---|---|
| 100 scenes · 3 states · 75k+ RGB-D frames | 10 task APIs · 3 evaluation axes | Depth · video depth · pose · tracking · VQA | Local checkpoints or inference APIs |

---

## Why RPX

Perception models are usually reported on a single accuracy number measured on
a static scene. That number does not say how the model behaves when the same
room is rearranged, when a person reaches into the frame, or what the
prediction costs to compute.

RPX evaluates perception across **100 indoor scenes** captured in three
states — clutter, human interaction, and clean — and reports every model on
**three independent axes**, never collapsed into a single composite:

1. **Task performance** — the per-task primary metric (AbsRel, mIoU, rotation
   error, …), stratified by difficulty.
2. **Scene-change robustness** — how far the output moves when the same scene
   transitions between states: state-transition robustness, cross-phase
   deltas, and temporal stability.
3. **Compute cost** — parameters, FLOPs and measured wall time, recorded with
   the hardware and precision that produced them.

Rankings routinely disagree across these axes. RPX reports them separately so
that disagreement stays visible.

Relative-pose evaluation uses the clutter and clean phases; video-depth
evaluation operates on whole phase clips. The VQA benchmark adds egocentric
and two-image questions.

Images, depth, instance masks, poses and question data live in the
[RPX dataset](https://huggingface.co/datasets/IRVLUTD/RPX) — roughly 75,000
frames over 100 scenes and ~70 object categories, with instance masks,
tracklets, metric depth, 6-DoF camera pose and language attributes. Model
weights and large experiment outputs are downloaded or generated separately.

---

## Get started

| Goal | Start here |
|---|---|
| Browse the Python API | [Toolkit documentation](https://irvlutd.github.io/RPX/toolkit-docs/) |
| Run the supplied models | [Reference models and environments](benchmark/README.md#reference-models) |
| Evaluate your own model or API | [Bring your own model](benchmark/README.md#bring-your-own-model) |
| Score a custom VLM on VQA | [Custom VQA model or API](benchmark/README.md#custom-vqa-model-or-api) |
| Inspect the dataset and splits | [Dataset metadata](benchmark/data/README.md) |
| Publish or re-package the dataset | [Dataset hub tools](benchmark/rpx_benchmark/dataset_hub/README.md) |
| Develop or verify the toolkit | [Testing](benchmark/README.md#testing) |

Most users only need `benchmark/`. The capture and annotation trees exist to
*produce* the dataset, not to consume it.

### Toolkit showcase

Run a reference-model gate before a full sweep:

```bash
# The runtime reports the complete, pinned VQA roster without loading weights.
docker run --rm narendhiranv04/rpx-vqa-smoke:vllm list-models

# Model-specific depth, tracking and pose commands live in their Docker READMEs.
python benchmark/scripts/run_depth_smoke_matrix.py --list-models
```

Bring a Python model or an inference API through the same evaluator:

```python
import rpx_benchmark as rpx

# predict_depth receives one RGB ndarray and returns an (H, W) depth map.
model = rpx.make_numpy_depth_model(predict_depth, name="my-depth-service")
result, report, paths = rpx.run_monocular_depth(
    rpx.MonocularDepthRunConfig(
        model=model,
        split="easy",
        output_dir="rpx_results/my-depth-service/easy",
    )
)
print(result.aggregated)
```

The default dataset revision is the immutable, consolidated release. Override
`RPX_HF_REPO` or `RPX_HF_REVISION` only when evaluating another dataset build.
For VQA, a custom callable receives ordered image paths plus the public prompt;
see the [API example](benchmark/README.md#custom-vqa-model-or-api).

---

## How it works

**Capture.** An Intel RealSense D435 records synchronised RGB and depth while
a T265 logs 6-DoF pose from fisheye stereo and IMU. Each scene is recorded
three times — a cluttered initial state, a pass with a human interacting with
the objects, and a clean organised state. Same scene, three states.
See [capture tools](data/capture/README.md).

**Annotate.** The mask pipeline propagates reviewed object annotations through
every frame of a phase, producing per-frame instance masks tied to a globally
consistent object-ID mapping. Grounding and VQA tools derive the corresponding
question data from those masks. See [annotation tools](data/mask_annotation/README.md).

**Benchmark.** Task runners load a pinned dataset revision or a local
manifest, execute a model, compute task metrics, and write reports with
per-sample and scene/phase detail. Reference model adapters and custom
callables go through the same metrics. See [benchmark usage](benchmark/README.md).

```text
   CAPTURE                    ANNOTATE                   BENCHMARK
   ───────                    ────────                   ─────────
   D435 RGB-D + T265 pose  →  reviewed per-frame     →   load → run → score
   three phases per scene     instance masks and         on three independent
                              derived question data      axes

   data/capture/              data/mask_annotation/      benchmark/  ← start here
```

---

<details>
<summary><b>Concepts and acronyms</b></summary>

| Term | Meaning |
|---|---|
| **RPX** | Robot Perception X — the dataset and the benchmark toolkit in this repository. |
| **RGB-D** | A colour image plus a per-pixel depth map. The D435 captures both at once. |
| **Metric depth** | Per-pixel distance in metres. Relative depth is defined only up to an unknown scale; the toolkit keeps the two kinds distinct and will not report relative output as metric. |
| **6-DoF pose** | The camera's 3D position and 3D orientation — six numbers per frame. |
| **VIO** | Visual-Inertial Odometry. The T265 fuses fisheye stereo and IMU to estimate 6-DoF pose. |
| **Phase** | One of the three capture passes of a scene: `clutter` (objects scattered), `interaction` (a person reaches in), `clean` (organised). Pair tasks such as relative camera pose use clutter and clean only. |
| **Split** | A named subset of the scenes: `easy`, `medium`, `hard`, assigned 33 / 33 / 34. All three phases of a scene land in the same split, so cross-phase behaviour stays measurable. |
| **ESD** | Effort-Stratified Difficulty. Difficulty derived from how visually hard a scene is and how much annotation effort its masks required. Used both to assign splits and to weight scores. |
| **Weighted phase score** | The ESD-weighted combination of a per-phase metric: `S_p = 0.25·M(p, easy) + 0.35·M(p, medium) + 0.40·M(p, hard)`. |
| **STR** | State-Transition Robustness. Performance change across a phase boundary — the drop from clutter to interaction and the recovery from interaction to clean. |
| **TS** | Temporal Stability. Pose-compensated frame-to-frame consistency of a model's output. |
| **SGC** | Stack-Level Geometric Coherence. Agreement between predicted mask boundaries and predicted depth discontinuities. |
| **RCPE** | Relative Camera Pose Estimation. Given two RGB frames, estimate the rotation and translation between them. |
| **Adapter** | A small wrapper putting a model behind the toolkit's uniform interface. |
| **Manifest** | A JSON file naming a task, a dataset root and the samples to evaluate, with modality paths, scene, phase and difficulty. The loader reads it; you rarely write one by hand. |
| **Revision** | The immutable dataset commit a run was evaluated against. Pin it; results from moving revisions are not comparable. |
| **MOS / Ego** | The two tracking protocols — the main multi-object scene protocol and the egocentric one. |
| **HF** | Hugging Face, where the dataset and most reference checkpoints are hosted. |

</details>

<details>
<summary><b>Tasks, metrics and outputs</b></summary>

The core toolkit registers ten tasks. Each declares a primary metric that
drives the ESD-weighted phase score:

| Task | Primary metric | Direction | Public runner |
|---|---|---|---|
| Monocular depth | `absrel` | lower is better | `run_monocular_depth` |
| Video depth | `absrel` | lower is better | `run_video_depth` |
| Object segmentation | `miou` | higher is better | `run_segmentation` |
| Object detection | `f1` | higher is better | `run_object_detection` |
| Open-vocabulary detection | `f1` | higher is better | `run_open_vocab_detection` |
| Object tracking | `mota` | higher is better | `run_object_tracking` |
| Visual grounding | `grounding_acc` | higher is better | `run_visual_grounding` |
| Relative camera pose | `rotation_error_deg` | lower is better | `run_relative_pose` |
| Sparse depth | `sparse_rmse` | lower is better | `run_sparse_depth` |
| Keypoint matching | `keypoint_acc` | higher is better | `run_keypoint_matching` |

VQA is evaluated separately, through JSONL question manifests and a
bbox-scoring workflow rather than the core task manifests — see
[custom VQA models](benchmark/README.md#custom-vqa-model-or-api).

Bundled reference sweeps exist for image depth, video depth, relative pose,
tracking and VQA. For segmentation, detection, open-vocabulary detection,
generic grounding, sparse depth and keypoint matching the toolkit supplies the
loaders, callable APIs and evaluators, but no bundled pretrained sweep.

Generic frame runs write `result.json`, `summary.md`, `cells.parquet` and
`per_sample_metrics.parquet`; video runs write clip and cell results.
Production runners add prediction caches, protocol-specific reports and
provenance. Store outputs outside the source tree, or under the ignored
`rpx_results/` directory.

</details>

<details>
<summary><b>Repository map</b></summary>

```text
benchmark/                 Installable Python package, examples and tests
  rpx_benchmark/           Loaders, adapters, metrics and public APIs
  scripts/                 Reference model runners and analysis utilities
  data/                    Dataset splits and release metadata
tracking/metadata/         Versioned tracking vocabulary and checksums
data/capture/              Sensor capture tools
data/mask_annotation/      Mask, grounding and VQA annotation tools
docker/                    Task-specific reproducible model environments
tools/dataset/             Dataset release preparation and QA
.github/workflows/         Tests, lint, types and package release
```

Each directory carries its own README with the details for that piece.

</details>

---

## Contributing and license

Run the [checks](benchmark/README.md#testing) before submitting changes: the
offline test suite, `ruff`, `mypy` and a package build. Install the
toolkit editable (`-e`) while developing — a frozen wheel install will show
stale behaviour.

The offline suite never downloads checkpoints. Validating real weights,
available memory and dataset access is a separate step, run on the target
machine through the per-environment gates documented under
[`docker/`](docker/). Capture and annotation-UI changes need the corresponding
hardware.

Keep documentation in README files, dataset metadata in its designated
folders, and generated outputs out of Git.

**Citation.** If you use RPX — the dataset, the toolkit, or any part of this
repository — please cite the accompanying paper. The BibTeX entry will be
added here once the paper is released.

**License.** Code in this repository is **MIT**; see [LICENSE](LICENSE). The
RPX dataset is released under **CC BY 4.0**; its terms are recorded in the
[dataset card](https://huggingface.co/datasets/IRVLUTD/RPX).

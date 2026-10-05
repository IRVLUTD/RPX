# API reference

Browse the toolkit by workflow below. Each module page documents its classes,
functions, signatures, type contracts, and source. The reference is generated
from the same source as the published Python package.

## Start with a workflow

| Goal | Entry points | Guide |
| --- | --- | --- |
| Load and download RPX | [`hub.load`, `fetch_manifest`, `download_split`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/hub.html), [`RPXDataset`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/loader.html) | [Get started](../getting-started/README.md) |
| Connect a model | [`BenchmarkableModel`, adapters and NumPy factories](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters.html) | [Bring your own model](../models/README.md) |
| Run a task | [Task runners and configurations](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks.html), [benchmark runner](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/runner.html) | [Add tasks](../tasks/README.md) |
| Compute or extend metrics | [Metric implementations](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics.html), [metric registry](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/registry.html) | [Add metrics](../metrics/README.md) |
| Use VQA or grounding contracts | [VQA tools](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa.html), [robot VQA](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/robot_vqa_v2.html) | [Model integration](../models/README.md) |
| Analyze robustness and efficiency | [Deployment](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/deployment.html), [Phi](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/phi.html), [JEDI](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/jedi.html), [profiler](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/profiler.html) | [Concepts](../README.md#concepts-and-acronyms) |

## Public Python interface

[`rpx_benchmark`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark.html) is the convenient top-level interface.
[`api`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/api.html) defines samples, task identities, ground truths,
and predictions. Module pages below expose the more specialized interfaces.

```python
import numpy as np
import rpx_benchmark as rpx

# Small integration fixture, not a pretrained model.
model = rpx.make_numpy_depth_model(
    lambda rgb: np.full(rgb.shape[:2], 2.0, dtype=np.float32)
)
# Point manifest_path at your task manifest before running inference.
config = rpx.MonocularDepthRunConfig(
    model=model,
    manifest_path="/data/manifests/monocular_depth/easy.json",
    device="cpu",
    output_dir="rpx_results/my-depth/easy",
)
```

## Installation and optional dependencies

```bash
python -m pip install 'rpx-benchmark[hub,schemas]'
```

Core evaluators use NumPy and SciPy. Dataset downloads and Parquet manifests use
`[hub]`; schema validation uses `[schemas]`; the Hugging Face Datasets bridge
uses `[hf-datasets]`; statistical analysis uses `[analysis]`; marker audits use
`[ar-pose-audit]`. Model runtimes remain isolated from the toolkit. Optional
modules may require these extras when called, even though their documentation
is available without loading model weights.

## Core harness and analysis

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.api`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/api.html) | Core task enums, data contracts, and the base model interface. |
| [`rpx_benchmark.ar_pose_audit`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/ar_pose_audit.html) | AR-board consistency audit for RPX single-object camera poses. |
| [`rpx_benchmark.box_upload`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/box_upload.html) | Box upload primitives — used by every task runner and the post-hoc |
| [`rpx_benchmark.cell_log`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/cell_log.html) | Per-(scene, phase) metric log — the canonical experiment artifact. |
| [`rpx_benchmark.cleanup`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/cleanup.html) | Graceful-shutdown plumbing for long-running CLI scripts. |
| [`rpx_benchmark.cli_ux`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/cli_ux.html) | Industry-grade colored logging + progress UI for the RPX CLI scripts. |
| [`rpx_benchmark.decode_contracts`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/decode_contracts.html) | Canonical decode helpers with runtime contracts. |
| [`rpx_benchmark.deployment`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/deployment.html) | Deployment-readiness metrics for RPX: TS, STR, SGC, ESD, weighted scoring. |
| [`rpx_benchmark.determinism`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/determinism.html) | Deterministic seeding for reproducible benchmark runs. |
| [`rpx_benchmark.evaluators`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/evaluators.html) | Legacy metric module (compat shim). |
| [`rpx_benchmark.exceptions`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/exceptions.html) | Exception hierarchy for the RPX benchmark toolkit. |
| [`rpx_benchmark.hub`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/hub.html) | HuggingFace Hub integration for RPX benchmark. |
| [`rpx_benchmark.jedi`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/jedi.html) | JEDI: Joint Empirical Desirability Index. |
| [`rpx_benchmark.loader`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/loader.html) | Data loading utilities for RPX benchmark. |
| [`rpx_benchmark.logging_utils`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/logging_utils.html) | Structured logging for the RPX benchmark toolkit. |
| [`rpx_benchmark.model_profiler`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/model_profiler.html) | Unified model profiler for all RPX tasks. |
| [`rpx_benchmark.paper_depth_analysis`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/paper_depth_analysis.html) | Paper-level statistical analysis for the six-metric RPX D1-F cell log. |
| [`rpx_benchmark.paper_tracking_analysis`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/paper_tracking_analysis.html) | Paper-level statistical analysis for RPX D3 tracking cells. |
| [`rpx_benchmark.phi`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/phi.html) | State-Transition Robustness Φ via repeated-measures MANOVA. |
| [`rpx_benchmark.phi_jedi_summary`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/phi_jedi_summary.html) | Glue between the runner's per-sample metrics and Φ / JEDI. |
| [`rpx_benchmark.pose_conventions`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_conventions.html) | Coordinate conventions shared by RPX camera-pose benchmarks. |
| [`rpx_benchmark.pose_metrics`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_metrics.html) | RPX-RCPE evaluation metrics — novel + standard. |
| [`rpx_benchmark.pose_pairs`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/pose_pairs.html) | On-the-fly deterministic pose-pair generation for RPX-RCPE. |
| [`rpx_benchmark.profiler`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/profiler.html) | Hardware-agnostic model efficiency profiling for RPX. |
| [`rpx_benchmark.reports`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/reports.html) | Result serialisation: JSON + markdown summary for benchmark runs. |
| [`rpx_benchmark.runner`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/runner.html) | Benchmark runner orchestrating model, data, and metrics. |
| [`rpx_benchmark.schemas`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/schemas.html) | Strict, typed manifest schemas for the RPX benchmark. |
| [`rpx_benchmark.temporal_budget_sweep`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/temporal_budget_sweep.html) | Frame-budget sweep runner and degradation analysis for video tasks. |
| [`rpx_benchmark.video_loader`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/video_loader.html) | Sequence-shaped loader for video tasks (Video Depth today; future video tasks). |

## Model adapters

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.adapters`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters.html) | Formal adapter framework for bringing models into the RPX benchmark. |
| [`rpx_benchmark.adapters.base`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/base.html) | Core types for the RPX adapter framework. |
| [`rpx_benchmark.adapters.batched_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/batched_depth.html) | True-batched depth model wrapper. |
| [`rpx_benchmark.adapters.batched_multimodal`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/batched_multimodal.html) | Generic batched-dispatch BenchmarkModel base for multi-modal tasks. |
| [`rpx_benchmark.adapters.depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/depth.html) | Depth-model adapters (paper task Image Depth: MONOCULAR_DEPTH). |
| [`rpx_benchmark.adapters.depth.skeletons`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/depth/skeletons.html) | Skeleton adapter classes for every Image Depth (monocular depth) model. |
| [`rpx_benchmark.adapters.depth_scaffold`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/depth_scaffold.html) | Shared scaffold for depth-model adapters (Image Depth and Video Depth). |
| [`rpx_benchmark.adapters.video_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/video_depth.html) | Video-depth adapters (paper task Video Depth: VIDEO_DEPTH). |
| [`rpx_benchmark.adapters.video_depth.skeletons`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/adapters/video_depth/skeletons.html) | Skeleton adapter classes for every Video Depth model. |

## Data loading and feature extraction

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.data`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data.html) | HuggingFace datasets integration for RPX. |
| [`rpx_benchmark.data.clip_dataset`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/clip_dataset.html) | Per-(scene, phase) clip iteration for Video Depth and other |
| [`rpx_benchmark.data.esd`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/esd.html) | Effort-Stratified Difficulty (ESD) feature extraction. |
| [`rpx_benchmark.data.esd_scoring`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/esd_scoring.html) | ESD scoring methods: turn the 27-feature table into a single difficulty |
| [`rpx_benchmark.data.features`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/features.html) | Canonical datasets.Features definitions for every RPX task. |
| [`rpx_benchmark.data.hf_bridge`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/hf_bridge.html) | Row-level bridge between datasets.Dataset and :class:Sample. |
| [`rpx_benchmark.data.load`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/load.html) | One-line datasets.load_dataset entry point for RPX. |
| [`rpx_benchmark.data.torch_dataloader`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/torch_dataloader.html) | PyTorch DataLoader adapter for RPX datasets. |

## Dataset publishing and storage

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.dataset_hub`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub.html) | RPX dataset hub — selective HuggingFace download/upload pipeline. |
| [`rpx_benchmark.dataset_hub.cli`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/cli.html) | python -m rpx_benchmark.dataset_hub.cli — sub-commands for the dataset hub. |
| [`rpx_benchmark.dataset_hub.croissant`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/croissant.html) | Stage the Croissant metadata JSON into the upload tree. |
| [`rpx_benchmark.dataset_hub.dataset_card`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/dataset_card.html) | Generate the HuggingFace dataset card (README.md) at the repo root. |
| [`rpx_benchmark.dataset_hub.downloader`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/downloader.html) | Selective download from the RPX HuggingFace dataset repo. |
| [`rpx_benchmark.dataset_hub.ego_layout`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/ego_layout.html) | Arrange raw ego (GoPro) captures into the layout scan_capture_root expects. |
| [`rpx_benchmark.dataset_hub.lossless_convert`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/lossless_convert.html) | Re-encode a capture tree into denser lossless formats for HF upload. |
| [`rpx_benchmark.dataset_hub.manifest`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/manifest.html) | Build the per-frame Parquet manifest the dataset hub ships. |
| [`rpx_benchmark.dataset_hub.mock`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/mock.html) | Synthetic mock generator that mimics the on-disk capture layout. |
| [`rpx_benchmark.dataset_hub.packer`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/packer.html) | Pack an on-disk capture tree into per-modality tar shards for HF upload. |
| [`rpx_benchmark.dataset_hub.recipes`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/recipes.html) | Task → modality recipes for selective dataset downloads. |
| [`rpx_benchmark.dataset_hub.scanner`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/scanner.html) | Walk an on-disk RPX capture tree and report structured inventory. |
| [`rpx_benchmark.dataset_hub.split_manifests`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/split_manifests.html) | Per-task, per-split manifest writer for the dataset hub. |
| [`rpx_benchmark.dataset_hub.staging`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/staging.html) | File-staging helpers — non-tar artefacts that ship to the HF repo root. |
| [`rpx_benchmark.dataset_hub.uploader`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/uploader.html) | Upload a packed staging directory to a HuggingFace dataset repo. |

## Metric implementations

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.metrics`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics.html) | Pluggable metric registry for RPX benchmark tasks. |
| [`rpx_benchmark.metrics.depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/depth.html) | Monocular absolute depth metric calculators. |
| [`rpx_benchmark.metrics.depth_alignment`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/depth_alignment.html) | Per-frame alignment of a predicted depth map to GT scale. |
| [`rpx_benchmark.metrics.depth_paper`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/depth_paper.html) | Canonical RPX D1-F paper metrics and fixed camera calibration. |
| [`rpx_benchmark.metrics.depth_robotics`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/depth_robotics.html) | Robotics-deployment depth metric calculators. |
| [`rpx_benchmark.metrics.depth_temporal`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/depth_temporal.html) | Clip-level temporal depth metrics for Video Depth. |
| [`rpx_benchmark.metrics.detection`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/detection.html) | Object detection + open-vocab detection metric calculators. |
| [`rpx_benchmark.metrics.grounding`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/grounding.html) | Visual grounding metric calculators. |
| [`rpx_benchmark.metrics.keypoints`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/keypoints.html) | Keypoint correspondence metric calculators. |
| [`rpx_benchmark.metrics.pose`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/pose.html) | Relative camera pose metric calculators. |
| [`rpx_benchmark.metrics.registry`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/registry.html) | Core of the metric plugin system. |
| [`rpx_benchmark.metrics.segmentation`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/segmentation.html) | Instance/semantic segmentation metric calculators. |
| [`rpx_benchmark.metrics.sparse_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/sparse_depth.html) | Sparse depth metric calculators (RGB + sparse GT depth points). |
| [`rpx_benchmark.metrics.specs`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/specs.html) | Per-metric specifications used by JEDI and Φ. |
| [`rpx_benchmark.metrics.tracking`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/tracking.html) | Multi-object tracking metric calculators. |
| [`rpx_benchmark.metrics.tracking_paper`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/tracking_paper.html) | Paper-protocol metrics for RPX D3 multi-object tracking. |
| [`rpx_benchmark.metrics.video_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/video_depth.html) | Metric calculators bound to TaskType.VIDEO_DEPTH (paper task Video Depth). |

## Task runners

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.tasks`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks.html) | Task runners + registry. |
| [`rpx_benchmark.tasks._pipeline`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/_pipeline.html) | Shared runner implementation; use the task-specific RunConfig and run_* interfaces. |
| [`rpx_benchmark.tasks._video_pipeline`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/_video_pipeline.html) | Shared runner implementation; use the task-specific RunConfig and run_* interfaces. |
| [`rpx_benchmark.tasks.detection`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/detection.html) | Object detection (closed vocab + open vocab). |
| [`rpx_benchmark.tasks.keypoint_matching`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/keypoint_matching.html) | Keypoint matching between a pair of RGB frames. |
| [`rpx_benchmark.tasks.monocular_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/monocular_depth.html) | Monocular absolute depth. |
| [`rpx_benchmark.tasks.registry`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/registry.html) | Task plugin registry. |
| [`rpx_benchmark.tasks.relative_pose`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/relative_pose.html) | Relative camera pose between two frames. |
| [`rpx_benchmark.tasks.robot_vqa`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/robot_vqa.html) | RPX Robot-Grounded VQA — tasks a robot actually needs. |
| [`rpx_benchmark.tasks.robot_vqa_v2`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/robot_vqa_v2.html) | RPX Robot VQA v2 — three context modes, all GT from recorded data. |
| [`rpx_benchmark.tasks.segmentation`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/segmentation.html) | Object (instance) segmentation. |
| [`rpx_benchmark.tasks.sparse_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/sparse_depth.html) | Sparse depth: predict depth at a set of pixel locations. |
| [`rpx_benchmark.tasks.tracking`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/tracking.html) | Object tracking: per-frame MOT with stable track IDs. |
| [`rpx_benchmark.tasks.video_depth`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/video_depth.html) | Video absolute depth (paper task Video Depth). |
| [`rpx_benchmark.tasks.visual_grounding`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/visual_grounding.html) | Visual grounding: referring expression -> box. |
| [`rpx_benchmark.tasks.vqa_generation`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/tasks/vqa_generation.html) | RPX VQA ground-truth generation. |

## VQA contracts and evaluation

| Module | Purpose |
| --- | --- |
| [`rpx_benchmark.vqa`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa.html) | Benchmark contracts for RPX's single-image VQA tasks. |
| [`rpx_benchmark.vqa.canonical`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/canonical.html) | Canonicalization for detecting semantic (not just literal-string) |
| [`rpx_benchmark.vqa.contract`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/contract.html) | Canonical, model-independent contract for RPX VQA benchmark rows. |
| [`rpx_benchmark.vqa.evaluate`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/evaluate.html) | Run any callable VQA model with the published RPX prompts and scoring. |
| [`rpx_benchmark.vqa.hub_rgb`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/hub_rgb.html) | Fetch one or two members from RPX's uncompressed RGB tar shards via HTTP |
| [`rpx_benchmark.vqa.metrics`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/metrics.html) | Benchmark metrics and smoke-gate aggregation for RPX VQA. |
| [`rpx_benchmark.vqa.outputs`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/outputs.html) | Strict parsing and normalization of VQA model outputs. |
| [`rpx_benchmark.vqa.prompts`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/prompts.html) | Task prompts before each model's official chat/image template is applied. |
| [`rpx_benchmark.vqa.roster`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/roster.html) | Frozen single-image model roster supplied for the RPX VQA benchmark. |
| [`rpx_benchmark.vqa.sampling`](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/vqa/sampling.html) | Deterministic, diversity-preferring row selection shared by the |

Reference pages include search across the generated modules. Use the workflow guides for runnable integration examples and the module pages for the detailed contract.

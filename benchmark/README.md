# RPX Benchmark Toolkit

**Benchmark robot-perception models on real-world RGB-D data — across clutter, human interaction, and clean phases of the same 99 indoor + outdoor scenes.**

`rpx-benchmark` is the reference toolkit for [RPX — Robot Perception
X](https://github.com/IRVLUTD/RPX). The toolkit core ships the **dataset
loader, task-specific metrics, hardware-agnostic profiling, and a
deployment-readiness scoring framework**. The team's reference
implementations of nine monocular-depth adapters and the per-task run
scripts live alongside it under `scripts/` for the paper sweep.

External users with their own models: [skip to the BYO-model
quickstart](#bring-your-own-model). Team members running the canonical
sweep: [start here](#for-the-team-canonical-sweep).

---

## For the team — canonical sweep

This is the day-the-data-lands launch sequence. **Single-page version:
[`TEAM_LAUNCH.md`](TEAM_LAUNCH.md)**.

### 0. One-time setup

```bash
pip install -e 'benchmark[hub,viz]'
hf auth login                                    # for HF dataset push
export BOX_DEVELOPER_TOKEN='<60-min-token>'      # https://app.box.com/developers/console (refresh hourly)
```

### 1. Push the dataset to HuggingFace (Itay's path)

```bash
# Run once per dataset version, on the system that holds the ~890 GB captures.
python -m rpx_benchmark.dataset_hub.cli pack            --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli manifest        --src DATA --staging STAGE \
                                                          --splits benchmark/data/splits/scene_splits.json
python -m rpx_benchmark.dataset_hub.cli stage-splits    --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli dataset-card    --src DATA --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli stage-croissant --staging STAGE --overwrite
python -m rpx_benchmark.dataset_hub.cli upload          --staging STAGE --repo-id IRVLUTD/RPX
```

Idempotent — re-run any time. Without `--splits` the `manifest` step
**fails loudly** rather than silently producing an unusable HF tree.
After upload, any consumer calling `rpx.load("monocular_depth", "easy")`
gets: JSON fetched → tar shards downloaded → tars **auto-extracted**
into `<snapshot>/extracted/scenes/...` (no manual extraction step).

Full details: [`rpx_benchmark/dataset_hub/README.md`](rpx_benchmark/dataset_hub/README.md).

### 2. Run the depth benchmark (Naren / depth lead)

```bash
# Per model, per split. Add --upload-to-box to mirror the result dir to UTD Box.
PYTHONPATH=. python scripts/run_depth.py --model <KEY> --split <easy|medium|hard> \
    --save-predictions --comprehensive-metrics --upload-to-box
```

The 9 live model keys (all reproducible from a single HF / pip URL):

| `--model <key>`           | Type                    | Native precision |
|---------------------------|-------------------------|------------------|
| `zoedepth`                | metric                  | fp32 |
| `depth_pro`               | metric (D435 FOV-corrected) | fp32 |
| `da_v2_metric_indoor`     | metric (Hypersim)       | fp16 |
| `da_v2_metric_outdoor`    | metric (VKITTI)         | fp16 |
| `unidepth_v2`             | metric                  | fp16 |
| `da_v2_relative`          | up-to-scale (`ls_affine`) | fp16 |
| `da_v1`                   | up-to-scale (`ls_affine`) | fp32 |
| `midas_v31`               | up-to-scale (`ls_affine`) | fp32 |
| `distill_any_depth`       | up-to-scale (`ls_affine`) | fp32 |

Per-run output lands under `rpx_results/<DisplayName>/<split>/`:
- `result.json` — primary metric, full deployment-readiness report (Tier 1/2/3 + DRS OperatingPoint), per-stage timing with 95% CIs.
- `summary.md` — human-readable.
- `comprehensive_metrics.json` — full Feynman-spec basket (9 errors + 3 accuracy + alignment modes + depth-band stratification + per-object basket + hole stats + ORD).
- `predictions/<scene>/<phase>/<frame>.npz` — per-frame raw depth (when `--save-predictions`).

Box mirror (when `--upload-to-box`):
`<box_root>/monocular_depth/<DisplayName>/<split>/...` matches the local
tree exactly. Idempotent (size-matched skip on re-upload).

Pending Turn C–F adapters (11 more, ~30 LOC each via `MODEL_REGISTRY`):
Marigold, Marigold-LCM, Lotus-2, GeoWizard, MoGe-2, MoGe v1,
PatchFusion, HyDen-metric, HyDen-relative, Metric3D V2, MetricSolver.

### 3. Aggregate the sweep into the paper table

```bash
# After every model has run on a split:
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy
PYTHONPATH=. python scripts/run_drs_sweep.py --split easy --sensitivity   # paper-appendix Kendall's τ
```

Outputs: `rpx_results/_sweep/drs_<split>.{csv,json}` and
`sensitivity_<split>.json`. The DRS (Deployment Readiness Score) is the
multiplicative `TP × R × E` headline metric — see Feynman's
`paper-submission/latex/drs_section.tex` for the axiomatization.

### 4. Cross-session coordination

| Channel | Owner | When to consult |
|---|---|---|
| [`TEAM_LAUNCH.md`](TEAM_LAUNCH.md) | one-page handoff | running the sweep |
| [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md) | log between Feynman + Session B Claude sessions | "what changed today?" |
| [`scripts/README.md`](scripts/README.md) | per-script docs | extending the depth zoo or the metric basket |
| [`rpx_benchmark/dataset_hub/README.md`](rpx_benchmark/dataset_hub/README.md) | hub team guide | HF upload questions |
| `paper-submission/BRIEF_FOR_ADVISOR.tex` | advisor brief | scope discussions |

### Known limits

- **`DEPTH_MAX_M = 5.0`** in `rpx_benchmark/metrics/depth_alignment.py`. D435 is unreliable beyond ~6 m; pixels with GT > 5 m are silently excluded. RPX has 35 outdoor scenes with GT > 5 m → those models are evaluated only on their <5 m subset. Decision needed before paper sweep: raise the cap, stratify by domain, or document and move on.
- **Box dev tokens expire every 60 min** — every relaunch needs a fresh export. The pipeline raises `ConfigError("Box API 401 — token expired", hint="…regenerate at developer console")` cleanly when stale.
- **VQA recipe** is wired but emits zero entries (waits on the team's VQA label-generation pipeline). All 7 other tasks ship.

---

## Bring Your Own Model

External users with their own perception model want this section.

```bash
pip install 'rpx-benchmark[hub]'
```

```python
import numpy as np
import rpx_benchmark as rpx


def my_depth(rgb: np.ndarray) -> np.ndarray:
    """rgb: H x W x 3 uint8  ->  H x W float32 (metres)."""
    ...


model = rpx.make_numpy_depth_model(my_depth, name="my_depth")
cfg = rpx.MonocularDepthRunConfig(model=model, split="hard", device="cpu")
result, report, paths = rpx.run_monocular_depth(cfg)

print(result.aggregated)              # absrel, rmse, delta1..3
print(report.weighted_phase_score)    # ESD-weighted phase score
print(paths["json"])                  # result.json
```

Same shape for every task: `make_numpy_<task>_model(fn)` →
`<TaskName>RunConfig(model=...)` → `run_<task>(cfg)`.

### Three integration paths

1. **Plain numpy callable.** Ten factories, one per task. Shown above.
2. **Custom `InputAdapter` / `OutputAdapter`.** When you need full
   control over preprocessing, batching, or post-processing.
3. **API / cloud model.** Wrap a remote inference service as an
   adapter and mark it `EfficiencyMetadata(model_type="api")`.

See [`docs/getting-started/bring-your-own-model.md`](docs/getting-started/bring-your-own-model.md)
for complete templates of all three.

### True-batched dispatch (optional)

The stock `BenchmarkableModel.predict` iterates samples one at a time,
which is fine for CPU and one-shot debugging. For GPU pipelines that
batch internally (HuggingFace pipelines, DataLoader-driven inference),
wrap your callable as a `BatchedDepthBenchmarkModel` (or
`BatchedSegmentationBenchmarkModel` / `BatchedRelativePoseBenchmarkModel`)
to dispatch the entire batch through your adapter in one call. Model
weights and forward semantics are unchanged — only the dispatch shape.

```python
from rpx_benchmark import BatchedDepthBenchmarkModel
model = BatchedDepthBenchmarkModel(my_depth_callable, name="my_model",
                                    save_dir="rpx_results/my_model/easy/predictions")
```

The `scripts/depth_models/` directory shows nine reference adapters
following this pattern (HF pipelines, custom post-processing for
focal-length-dependent metric depth, fp16 declarations, etc.).

## What the toolkit ships

| Area | Module | What's in it |
|---|---|---|
| Data contracts | `rpx_benchmark.api` | `TaskType`, `Sample`, 10 × GroundTruth, 10 × Prediction, `BenchmarkModel` ABC |
| Dataset | `rpx_benchmark.loader`, `rpx_benchmark.hub`, `rpx_benchmark.data` | `RPXDataset`, modality-aware snapshot downloads (now with auto-extraction), `load_hf()` one-liner |
| Manifest validation | `rpx_benchmark.schemas` | Pydantic v2 schemas + JSON Schema export (opt-in extra) |
| Adapter framework | `rpx_benchmark.adapters` | `BenchmarkableModel`, **batched-dispatch base class** for true GPU batching, 9 numpy fast-path factories |
| Metrics | `rpx_benchmark.metrics.*` | Per-task `MetricCalculator` registry; `depth_alignment` shared between runner + comprehensive post-processor |
| Tasks | `rpx_benchmark.tasks.*` | One module per task, each ~30 lines; self-register `TaskSpec` |
| Runner | `rpx_benchmark.runner` | Orchestrates dataset → model → metrics → report; persists `OperatingPoint` for DRS |
| Profiler | `rpx_benchmark.profiler` | Tier 1 (params/FLOPs/MACs/memory traffic/AI), Tier 2 (roofline for A100/4090/Orin), Tier 3 (measured + system card) |
| Deployment readiness | `rpx_benchmark.deployment` | DRS = TP × R × E (multiplicative, hardware-agnostic), ESD-weighted Phase Score, STR, Temporal Stability, SGC |
| Sweep utilities | `scripts/run_drs_sweep.py`, `rpx_benchmark.drs_sensitivity` | Sweep aggregator + sensitivity Kendall's τ |
| Determinism | `rpx_benchmark.determinism` | `seed_all()`, `deterministic()` context manager, project-wide `RPX_SEED = 5_062_026` |
| Reports | `rpx_benchmark.reports` | JSON + Markdown writers |

## Tasks (10 contracts; loader-side support varies)

| Task | Primary metric | Numpy factory | Loader status |
|---|---|---|---|
| Monocular depth | AbsRel ↓ | `make_numpy_depth_model` | ✓ live |
| Object segmentation | mIoU ↑ | `make_numpy_mask_model` | ✓ live |
| Object detection | mAP ↑ | `make_numpy_detection_model` | ✓ live |
| Open-vocab detection | mAP ↑ | `make_numpy_detection_model` | ✓ live |
| Object tracking | MOTA ↑ | `make_numpy_tracking_model` | ✓ live (manifest done) |
| Visual grounding | grounding_acc ↑ | `make_numpy_grounding_model` | ⚠ awaits VQA labels |
| Relative camera pose | rotation_error_deg ↓ | `make_numpy_pose_model` | ✓ live (paired manifest) |
| Sparse depth | RMSE ↓ | `make_numpy_sparse_depth_model` | ✓ live |
| Novel view synthesis | PSNR ↑ | `make_numpy_nvs_model` | ✓ live |
| Keypoint matching | keypoint_acc ↑ | `make_numpy_keypoint_model` | ✓ live |

## Dataset access

```python
# 1. HuggingFace `datasets` (streaming, one-liner)
from rpx_benchmark.data import load_hf
ds = load_hf("monocular_depth", split="hard")

# 2. Modality-aware snapshot (offline-friendly bulk; auto-extracts tars)
from rpx_benchmark.hub import load
ds = load("monocular_depth", "hard")
```

Both yield `list[Sample]` batches the runner consumes interchangeably.

## Visualize a scene

One command opens an interactive [Rerun](https://rerun.io/) viewer with all six modalities (RGB, depth, fisheye stereo, instance masks, and the T265 6-DoF camera trajectory) on a shared timeline.

![RPX in Rerun: 3 phases × 6 panels (cam pose, RGB, cividis depth, royal-tone masks, fisheye L, fisheye R)](docs/assets/rpx_rerun_viewer.png)

```bash
make visualize            # stream into viewer; no disk artifact (HF cache only)
make visualize-phases     # all 3 phases (clutter / interaction / clean) side-by-side
make visualize-save       # also write a portable .rrd to benchmark/site/
make visualize-lite       # constrained devices (Pi 4 / Jetson Nano): stride=3, smaller
make visualize-headless   # build .rrd only, no viewer (CI / SSH)
```

### Single-phase layout

```
┌──────────────┬──────────────┬──────────────┐
│   Cam Pose   │  Fisheye L   │  Fisheye R   │
├──────────────┼──────────────┼──────────────┤
│     RGB      │ Depth (mm)   │    Masks     │
└──────────────┴──────────────┴──────────────┘
```

### Visual conventions

- **Depth**: cividis colormap, **per-frame histogram-equalized** so subtle structure remains visible. Invalid (`raw == 0`) pixels render black.
- **Masks**: hand-curated **royal jewel-tone palette** with deterministic `id → palette[id mod 16]` mapping. Same instance ID → same color across frames *and* phases.
- **Camera pose**: per-phase trajectory rendered as a colored polyline (magenta = clutter, orange = interaction, cyan = clean) with a Pinhole frustum sweeping along the path.
- **Fisheye stereo**: T265 monochrome, percentile-stretched per frame.

See `python scripts/visualize_rerun.py --help` for the full flag set.

## Install extras

| Extra | Pulls | Use when |
|---|---|---|
| *(core)* | `numpy`, `Pillow` | Always |
| `hub` | `huggingface_hub[hf_xet]` | Downloading from HF |
| `hf-datasets` | `datasets>=2.18` | `load_hf(...)` one-liner |
| `schemas` | `pydantic>=2.5` | Strict manifest validation |
| `viz` | `rerun-sdk`, `matplotlib`, `Pillow`, `tqdm` | `make visualize` (interactive Rerun) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`, `pre-commit` | Contributing |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs locally |

## Quality status

- **524 tests passing** (`PYTHONPATH=. pytest tests/`), 2 expected skips on paired-task mocks.
- **Single canonical seed** `RPX_SEED = 5_062_026` (MMDDYYYY 05/06/2026) shared across bootstrap CIs, ORD pair sampler, sparse-depth sampler, keypoint-pair sampler, and tests.
- **Single source of truth for alignment** in `rpx_benchmark.metrics.depth_alignment` — no duplicated math between runner and post-processor.
- **9 live mono-depth adapters**, all reproducible from a single URL each. 11 more pending.
- **Zero bare exceptions** in the team's hot-path code; everything raises typed `AdapterError` / `ConfigError` / `DatasetError` with hints.
- **2 contract tests** for Box upload (mock-based, no token needed in CI) verify the scene/phase tree shape and idempotency on re-upload.
- **7 integration tests** for the writer → loader round-trip catch silent contract bugs (entry-key drift, file-extension drift, extraction-gap regressions).

## Docs

Full site: <https://irvlutd.github.io/RPX/>

- [Quickstart](docs/getting-started/quickstart.md)
- [Load the Dataset](docs/getting-started/load-the-dataset.md)
- [Bring Your Own Model](docs/getting-started/bring-your-own-model.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Adding a Task](docs/guides/adding-a-task.md)
- [Adding a Metric](docs/guides/adding-a-metric.md)
- [Contributing](docs/guides/contributing.md)

## License

MIT — see [LICENSE](LICENSE).

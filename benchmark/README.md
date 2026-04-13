# RPX Benchmark Toolkit

**Benchmark your robot-perception model on real-world RGB-D data.**

`rpx-benchmark` is the reference toolkit for [RPX — Robot Perception
X](https://github.com/IRVLUTD/RPX). We ship **datasets, task-specific
dataloaders, metric calculators, and hardware-agnostic profiling**.
You ship the model.

There is no model registry, no CLI, no shipped model code.

---

## Quickstart

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

## Bring Your Own Model — three paths

1. **Plain numpy callable.** Ten factories, one per task. Shown above.
2. **Custom `InputAdapter` / `OutputAdapter`.** When you need full
   control over preprocessing, batching, or post-processing.
3. **API / cloud model.** Wrap a remote inference service as an
   adapter and mark it `EfficiencyMetadata(model_type="api")`.

See [`docs/getting-started/bring-your-own-model.md`](docs/getting-started/bring-your-own-model.md)
for complete templates of all three.

## What the toolkit ships

| Area | Module | What's in it |
|---|---|---|
| Data contracts | `rpx_benchmark.api` | `TaskType`, `Sample`, 10 × GroundTruth, 10 × Prediction, `BenchmarkModel` ABC |
| Dataset | `rpx_benchmark.loader`, `rpx_benchmark.hub`, `rpx_benchmark.data` | `RPXDataset`, modality-aware snapshot downloads, `load_hf()` one-liner |
| Manifest validation | `rpx_benchmark.schemas` | Pydantic v2 schemas + JSON Schema export (opt-in extra) |
| Adapter framework | `rpx_benchmark.adapters` | `InputAdapter` / `OutputAdapter` protocols, `BenchmarkableModel`, 9 numpy fast-path factories |
| Metrics | `rpx_benchmark.metrics.*` | Per-task `MetricCalculator` registry |
| Tasks | `rpx_benchmark.tasks.*` | One module per task, each ~30 lines; self-register `TaskSpec` |
| Runner | `rpx_benchmark.runner` | Orchestrates dataset → model → metrics → report |
| Profiler | `rpx_benchmark.profiler` | Params, FLOPs, latency p50/p95/p99, CPU/CUDA/MPS peak memory |
| Deployment readiness | `rpx_benchmark.deployment` | ESD-weighted Phase Score, STR, Temporal Stability, SGC |
| Determinism | `rpx_benchmark.determinism` | `seed_all()`, `deterministic()` context manager |
| Reports | `rpx_benchmark.reports` | JSON + Markdown writers |

## Tasks (all 10 runnable)

| Task | Primary metric | Numpy factory |
|---|---|---|
| Monocular depth | AbsRel ↓ | `make_numpy_depth_model` |
| Object segmentation | mIoU ↑ | `make_numpy_mask_model` |
| Object detection | mAP ↑ | `make_numpy_detection_model` |
| Open-vocab detection | mAP ↑ | `make_numpy_detection_model` |
| Object tracking | MOTA ↑ | `make_numpy_tracking_model` |
| Visual grounding | grounding_acc ↑ | `make_numpy_grounding_model` |
| Relative camera pose | rotation_error_deg ↓ | `make_numpy_pose_model` |
| Sparse depth | RMSE ↓ | `make_numpy_sparse_depth_model` |
| Novel view synthesis | PSNR ↑ | `make_numpy_nvs_model` |
| Keypoint matching | keypoint_acc ↑ | `make_numpy_keypoint_model` |

## Dataset access

Two paths — pick the one that fits:

```python
# 1. HuggingFace `datasets` (streaming, one-liner)
from rpx_benchmark.data import load_hf
ds = load_hf("monocular_depth", split="hard")

# 2. Modality-aware snapshot (offline-friendly bulk)
from rpx_benchmark.hub import load
ds = load("monocular_depth", "hard")
```

Both yield `list[Sample]` batches the runner consumes interchangeably.

## Install extras

| Extra | Pulls | Use when |
|---|---|---|
| *(core)* | `numpy`, `Pillow` | Always |
| `hub` | `huggingface_hub[hf_xet]` | Downloading from HF |
| `hf-datasets` | `datasets>=2.18` | `load_hf(...)` one-liner |
| `schemas` | `pydantic>=2.5` | Strict manifest validation |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`, `pre-commit` | Contributing |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs locally |

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

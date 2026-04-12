# Architecture Overview

The toolkit separates **what stays fixed** (the benchmark itself) from
**what varies** (the model under test). This page is the map of the
moving parts.

!!! info "Framework vs reference"
    The framework surface (loader, schemas, adapter protocols, metric
    and task registries, runner, profiler) lives in the top-level
    `rpx_benchmark` namespace. Concrete adapters and model factories
    for popular backbones live under `rpx_benchmark.reference.{adapters,
    models}` and depend on heavy optional extras (torch, transformers,
    UniDepth, Metric3D). The separation makes the "bring your model,
    we bring the harness" contract explicit.

## Layers

```text
┌─────────────────────────────────────────────────────────────┐
│  CLI  (rpx_benchmark.cli)                                   │
│  ├── auto-discovers tasks from the task registry            │
│  └── maps RPXError → exit codes                             │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│  Task pipelines  (rpx_benchmark.tasks.*)                    │
│  ├── monocular_depth.py                                     │
│  ├── segmentation.py                                        │
│  └── <your new task here>                                   │
│     each registers a TaskSpec with the task registry        │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│  Runner  (rpx_benchmark.runner.BenchmarkRunner)             │
│  ├── iterates the dataset                                   │
│  ├── wraps first batch in FlopCounterMode for FLOPs         │
│  ├── latency percentiles (p50/p95/p99) via LatencyProfiler  │
│  ├── peak memory (CPU/CUDA/MPS) via MemoryProfiler          │
│  ├── dispatches temporal-stability / geometric-coherence    │
│  │   via TaskSpec hooks (no task-branching in the runner)   │
│  ├── attaches per-sample metadata (id/phase/difficulty)     │
│  └── builds DeploymentReadinessReport                       │
└─────────────────────────────────────────────────────────────┘
           │                      │                │
┌──────────┴──────┐  ┌────────────┴────────┐  ┌────┴─────────────┐
│  Adapters       │  │  Metric registry    │  │  Loader / Hub    │
│  (Input/Output  │  │  (per-task plugin   │  │  (manifest parse │
│   framework)    │  │   calculators)      │  │   + HF download) │
└─────────────────┘  └─────────────────────┘  └──────────────────┘
```

## Manifest validation

Manifests are loaded through two tiers:

| Tier | Module | Speed | Errors |
|---|---|---|---|
| Default | `rpx_benchmark.loader` | Fastest; tolerant | Coarse: `ManifestError` with a single message |
| Strict (opt-in) | `rpx_benchmark.schemas` | ~5ms per 1k samples; requires `pydantic` | Field-level: `error.details["pydantic_errors"]` with `loc` tuples |

Opt into strict mode with `RPXDataset.from_manifest(..., validate=True)`
or by calling `schemas.validate_manifest(...)` directly. The schema
module also exports `schemas.dump_json_schema(task?)` for third-party
tooling. See [`api/schemas`](../api/schemas.md) for the full surface.

## Deployment-readiness hooks

The runner computes Temporal Stability (TS) and Stack Geometric
Coherence (SGC) by looking up two optional callables on the active
`TaskSpec`:

| Hook | Signature | Who ships a default |
|---|---|---|
| `TaskSpec.temporal_stability_fn` | `(predictions, samples, camera_poses) -> TemporalStabilityResult \| None` | `monocular_depth`, `object_segmentation` |
| `TaskSpec.geometric_coherence_fn` | `(predictions, samples) -> StackGeometricCoherenceResult \| None` | `object_segmentation` |

Tasks that don't register a hook skip the corresponding computation —
the runner never branches on task identity. To add a new
deployment-readiness metric for your task, set the callable on your
`TaskSpec` at construction time; no runner change needed.

## Profiling

Two hardware-agnostic profilers are instantiated per
`run_with_deployment_readiness` call:

- `LatencyProfiler(warmup=1)` — accumulates per-sample timings and
  reports `p50`, `p95`, `p99`, and `mean` in milliseconds, trimming
  warmup samples.
- `MemoryProfiler()` — samples CPU RSS (via `psutil` or POSIX
  `resource`), CUDA peak allocated memory (`torch.cuda`), and MPS
  current allocated memory (`torch.mps`). Each backend returns
  `None` when the corresponding runtime is unavailable; nothing is
  required to be installed.

Both profilers feed into `EfficiencyMetadata` and the
`DeploymentReadinessReport` alongside parameter count and FLOPs.

## Plugin registries

All extensibility flows through three registries:

| Registry | Module | Adds | Touchpoints on existing code |
|---|---|---|---|
| **Models** | `rpx_benchmark.models.registry` | Named factory → `BenchmarkableModel` | 0 |
| **Metrics** | `rpx_benchmark.metrics.registry` | `MetricCalculator` subclass per task | 0 |
| **Tasks** | `rpx_benchmark.tasks.registry` | `TaskSpec(task, primary_metric, run, ...)` | 0 |

Adding a new task, metric, or model to the slate is always a
**one-file change**. The CLI auto-discovers new tasks from the task
registry at parser-build time.

## Data flow for one `rpx bench <task>` call

```text
1. CLI parses flags → task's _build_config → TypedConfig
2. Pipeline: resolve device (CUDA fallback)
3.            ↓
    hub.download_split(task, split)
    ↓ writes resolved manifest to ~/.cache/rpx_benchmark/
4.            ↓
    RPXDataset.from_manifest(path)      ← raises ManifestError
5.            ↓
    Resolve model:  cfg.model         │
                    cfg.model_name    │   priority order
                    cfg.hf_checkpoint │
    ↓
    BenchmarkableModel instance
6.            ↓
    BenchmarkRunner.run_with_deployment_readiness(...)
    ├── First batch → FlopCounterMode → flops_g
    ├── Per-batch   → time.perf_counter → latency_ms (median)
    ├── Per-sample  → metric calc → result.per_sample with metadata
    └── After loop  → compute WPS / STR / TS
7.            ↓
    Reports: write_json + format_markdown_summary
8.            ↓
    Return (BenchmarkResult, DeploymentReadinessReport, paths)
```

## Exception hierarchy

```text
RPXError                       # base — `except rpx.RPXError` catches everything
├── ConfigError               # invalid user config
├── DatasetError
│   ├── ManifestError         # malformed / missing manifest
│   └── DownloadError         # HuggingFace / network failure
├── ModelError
│   └── AdapterError          # input / output adapter failure
└── MetricError               # evaluator failure
```

Every exception carries a `hint` string that tells the user what to
fix, plus an optional `details` dict for structured context.

## Logging

Every module creates its logger with
`log = get_logger(__name__)`. The CLI calls
[`configure_logging`][rpx_benchmark.logging_utils.configure_logging]
once at startup with a level driven by `--verbose` / `--quiet` /
`RPX_LOG_LEVEL`. When `rich` is installed the logs render through
`RichHandler`; otherwise a plain stream handler is used.

Hierarchy mirrors the package structure, so turning a single module
up or down is one call:

```python
import logging
logging.getLogger("rpx_benchmark.hub").setLevel(logging.DEBUG)
```

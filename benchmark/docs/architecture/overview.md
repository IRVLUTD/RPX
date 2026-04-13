# Architecture Overview

The toolkit separates **what stays fixed** (the benchmark itself) from
**what varies** (the model under test). This page is the map of the
moving parts.

## Layers

```text
┌─────────────────────────────────────────────────────────────┐
│  Task runners  (rpx_benchmark.tasks.*)                      │
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

| Tier | Module | Errors |
|---|---|---|
| Default | `rpx_benchmark.loader` | Coarse: `ManifestError` with a single message |
| Strict (opt-in) | `rpx_benchmark.schemas` | Field-level: `error.details["pydantic_errors"]` with `loc` tuples |

Opt into strict mode with `RPXDataset.from_manifest(..., validate=True)`
or `schemas.validate_manifest(...)`. The schema module also exports
`schemas.dump_json_schema(task?)` for third-party tooling.

## Deployment-readiness hooks

The runner computes Temporal Stability (TS) and Stack Geometric
Coherence (SGC) by calling two optional hooks on the active
`TaskSpec`:

| Hook | Signature | Default shipped by |
|---|---|---|
| `TaskSpec.temporal_stability_fn` | `(preds, samples, poses) → TemporalStabilityResult \| None` | `monocular_depth`, `object_segmentation` |
| `TaskSpec.geometric_coherence_fn` | `(preds, samples) → StackGeometricCoherenceResult \| None` | `object_segmentation` |

Tasks that don't register a hook skip the corresponding computation.
The runner never branches on task identity.

## Profiling

Every `run_with_deployment_readiness` call instantiates:

- `LatencyProfiler(warmup=1)` — `p50`, `p95`, `p99`, `mean` in ms
  (trimming warmup).
- `MemoryProfiler()` — CPU RSS (`psutil`/`resource`), CUDA peak
  (`torch.cuda`), MPS current (`torch.mps`). Each backend returns
  `None` when its runtime is unavailable.

Results feed into `EfficiencyMetadata` on the deployment report
alongside parameter count and FLOPs.

## Plugin registries

Two registries hold extensibility:

| Registry | Module | Adds |
|---|---|---|
| **Tasks** | `rpx_benchmark.tasks.registry` | `TaskSpec(task, primary_metric, run, ...)` |
| **Metrics** | `rpx_benchmark.metrics.registry` | `MetricCalculator` subclass per task |

The toolkit ships **no** model registry — users supply a
`BenchmarkableModel` directly via `cfg.model`.

## Exception hierarchy

```text
RPXError                       # base
├── ConfigError                # invalid user config
├── DatasetError
│   ├── ManifestError          # malformed / missing manifest
│   └── DownloadError          # HuggingFace / network failure
├── ModelError
│   └── AdapterError           # input / output adapter failure
└── MetricError                # evaluator failure
```

Every exception carries a `hint` string and an optional
`details` dict for structured context.

## Logging

Every module creates its logger with `log = get_logger(__name__)`.
The hierarchy mirrors the package, so turning a single module up or
down is one call:

```python
import logging
logging.getLogger("rpx_benchmark.hub").setLevel(logging.DEBUG)
```

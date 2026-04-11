# Architecture Overview

The toolkit separates **what stays fixed** (the benchmark itself) from
**what varies** (the model under test). This page is the map of the
moving parts.

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
│  ├── records per-sample latency (median, skip warmup)       │
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

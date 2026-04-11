# Registries

Three plugin registries carry everything that can vary: **models**,
**metrics**, and **tasks**. Adding a new entry to any of them is a
one-file change that touches **zero** pre-existing shared modules.

## Task registry

```python
from rpx_benchmark.tasks.registry import TaskSpec, register_task

TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_SEGMENTATION,
    display_name="Object Segmentation",
    description="Predict a (H, W) int instance mask from a single RGB frame.",
    primary_metric="miou",
    required_modalities=["rgb", "mask"],
    higher_is_better=True,
    build_config=_build_config,       # argparse.Namespace → Config dataclass
    run=run_segmentation,             # Config → (result, dr_report, paths)
    add_cli_arguments=_add_cli_args,  # argparse.ArgumentParser → None
)
register_task(TASK_SPEC)
```

The CLI auto-discovers every registered spec at parser-build time and
generates `rpx bench <task>` subcommands from them. See
[`rpx_benchmark.tasks.registry`][rpx_benchmark.tasks.registry] for
the full API.

## Metric registry

```python
from rpx_benchmark.metrics import MetricCalculator, register_metric
from rpx_benchmark.api import TaskType

@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthMedian(MetricCalculator):
    name = "depth_median"

    def compute(self, prediction, ground_truth):
        import numpy as np
        valid = ground_truth.depth_map > 0
        rel = (np.abs(prediction.depth_map - ground_truth.depth_map)
               / np.maximum(ground_truth.depth_map, 1e-6))
        return {"median_absrel": float(np.median(rel[valid]))}
```

Any number of calculators can be registered per task; the runner
merges their outputs into each per-sample metric dict. The aggregator
uses numeric-only means so metric keys play nicely with attached
metadata (`id`, `phase`, `difficulty`).

Module reference:
[`rpx_benchmark.metrics.registry`][rpx_benchmark.metrics.registry].

## Model registry

```python
from rpx_benchmark.models.registry import register

register(
    name="my_new_depth",
    module_suffix="my_pkg.my_module",   # importable dotted path
    factory_name="build_my_depth",      # callable inside that module
)
```

The factory is a `(device: str, **kwargs) → BenchmarkableModel`
callable. Lazy import means the top-level package stays importable
without torch / transformers / model-specific dependencies.

Module reference:
[`rpx_benchmark.models.registry`][rpx_benchmark.models.registry].

## Deferred entries

The model registry supports **deferred** entries that appear in the
listing but raise a clean `NotImplementedError` when resolved. This
is how we document models that are intentionally not yet wired
(Video Depth Anything waits for a temporal mode, Prompt Depth
Anything needs a sparse depth prompt, Depth Anything 3 isn't in
transformers yet). Users running `rpx models` see the full intended
slate and the concrete reason each deferred entry isn't runnable.

Add your own deferred entry by listing it in
`DEFERRED_MODELS` and providing a stub factory under
`rpx_benchmark/models/_deferred.py`.

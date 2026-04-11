# Adding a New Metric

Adding a new metric to an existing task is a **three-line change**:
write a `MetricCalculator` subclass, decorate it with
`@register_metric(TaskType.FOO)`, and import the module once so the
decorator fires.

Any number of calculators can be registered per task; the runner
merges their outputs into each per-sample metric dict.

## Minimal example

```python
# my_extra_metrics.py
from rpx_benchmark.api import DepthGroundTruth, DepthPrediction, TaskType
from rpx_benchmark.metrics import MetricCalculator, register_metric
import numpy as np


@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthMedian(MetricCalculator):
    """Median absolute relative error (complementary to mean AbsRel)."""
    name = "depth_median"

    def compute(self, prediction: DepthPrediction,
                ground_truth: DepthGroundTruth) -> dict[str, float]:
        valid = ground_truth.depth_map > 0
        rel = (np.abs(prediction.depth_map - ground_truth.depth_map)
               / np.maximum(ground_truth.depth_map, 1e-6))
        return {"median_absrel": float(np.median(rel[valid]))}
```

Import this module once at program start (e.g. from a project
`conftest.py`, a package `__init__.py`, or a script entrypoint) and
every subsequent monocular-depth run will include `median_absrel` in
its output — **without touching** the runner, CLI, reports, or any
existing metric code.

## Testing your calculator

```python
# tests/test_my_extra_metrics.py
import numpy as np
from rpx_benchmark.api import DepthGroundTruth, DepthPrediction, TaskType
from rpx_benchmark.metrics import compute_metrics

import my_extra_metrics  # registers the calculator


def test_depth_median_perfect_is_zero():
    gt = DepthGroundTruth(depth_map=np.full((4, 4), 2.0, dtype=np.float32))
    pred = DepthPrediction(depth_map=np.full((4, 4), 2.0, dtype=np.float32))
    out = compute_metrics(TaskType.MONOCULAR_DEPTH, pred, gt)
    assert out["median_absrel"] == 0.0
    # The built-in AbsRel / RMSE / delta calculators still run.
    assert out["absrel"] == 0.0
```

## Contract requirements

- Subclass [`MetricCalculator`][rpx_benchmark.metrics.registry.MetricCalculator]
  and implement `compute(prediction, ground_truth) → dict[str, float]`.
- Set `name` to a short unique identifier. Used for
  `unregister_metric` and in error messages.
- Output keys **must be numeric**. The aggregator silently skips
  non-numeric values (used for `id` / `phase` / `difficulty`
  metadata), so non-numeric metric outputs vanish.
- Raise [`MetricError`][rpx_benchmark.exceptions.MetricError] when
  inputs have the wrong type or shape. Every built-in calculator does
  this — it gives users a usable error instead of a bare
  `AssertionError`.

## Removing a built-in calculator

To disable a shipped calculator at runtime:

```python
from rpx_benchmark.api import TaskType
from rpx_benchmark.metrics import unregister_metric

unregister_metric(TaskType.MONOCULAR_DEPTH, "depth_error_metrics")
```

Note that the runner will then raise
[`MetricError`][rpx_benchmark.exceptions.MetricError] when it tries
to evaluate that task, unless you have another calculator registered.

## Full API

See the [`rpx_benchmark.metrics`](../api/metrics.md) reference page
for the complete public surface (`MetricCalculator`, `MetricSuite`,
`register_metric`, `unregister_metric`, `compute_metrics`,
`available_metrics`, `BenchmarkResult`).

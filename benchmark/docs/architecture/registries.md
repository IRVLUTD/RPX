# Plugin registries

The toolkit exposes two plugin registries. Each registry is a single
dict under a module; new entries self-register on import.

| Registry | Module | Scope |
|---|---|---|
| **Tasks** | `rpx_benchmark.tasks.registry` | Per-task `TaskSpec` (metrics, modalities, run callable, optional deployment-readiness hooks) |
| **Metrics** | `rpx_benchmark.metrics.registry` | Per-task `MetricCalculator` — any number registered per task |

## Adding a task

```python
from rpx_benchmark.api import TaskType
from rpx_benchmark.tasks.registry import TaskSpec, register_task
from rpx_benchmark.tasks._pipeline import run_pipeline, TaskRunConfig

def run_my_task(cfg):
    return run_pipeline(task=TaskType.MY_TASK, primary_metric="my_metric", cfg=cfg)

TASK_SPEC = TaskSpec(
    task=TaskType.MY_TASK,
    display_name="My Task",
    description="One-line description.",
    primary_metric="my_metric",
    required_modalities=["rgb"],
    higher_is_better=True,
    run=run_my_task,
)
register_task(TASK_SPEC)
```

Task plugins can also carry two optional deployment-readiness hooks:

- `temporal_stability_fn(preds, samples, poses) -> TemporalStabilityResult | None`
- `geometric_coherence_fn(preds, samples) -> StackGeometricCoherenceResult | None`

The runner dispatches through these uniformly; tasks without a hook
skip the corresponding metric. See
`rpx_benchmark/tasks/monocular_depth.py` and
`rpx_benchmark/tasks/segmentation.py` for working examples.

## Adding a metric

```python
from rpx_benchmark.api import TaskType
from rpx_benchmark.metrics.registry import MetricCalculator, register_metric

@register_metric(TaskType.MONOCULAR_DEPTH)
class MyDepthMetric(MetricCalculator):
    name = "my_depth_metric"

    def compute(self, prediction, ground_truth):
        return {"my_depth_metric": float(...)}
```

Multiple calculators can register against the same task; the runner
merges their per-sample outputs.

## Models: there is no registry

The toolkit does not ship a model registry. Users bring their own
`BenchmarkableModel` and pass it to the config's `model` field — see
[Bring Your Own Model](../getting-started/bring-your-own-model.md).

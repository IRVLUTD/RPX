# Adding a New Task

A task module is ~30 lines. Clone
[`rpx_benchmark/tasks/keypoint_matching.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/tasks/keypoint_matching.py),
swap the task-specific knobs, and register a `TaskSpec`.

## Minimum viable task module

```python
# rpx_benchmark/tasks/my_task.py
from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "my_metric"     # key your MetricCalculator emits


@dataclass
class MyTaskRunConfig(TaskRunConfig):
    pass   # add task-specific fields here if you need any


def run_my_task(cfg: MyTaskRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.MY_TASK,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=True,    # flip to False for pair tasks
        compute_sgc=False,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.MY_TASK,
    display_name="My Task",
    description="One-line description.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "..."],
    higher_is_better=True,
    run=run_my_task,
)

register_task(TASK_SPEC)
```

## Register in the tasks package

Add one import line to `rpx_benchmark/tasks/__init__.py` so the
module loads when the package loads:

```python
from .my_task import MyTaskRunConfig, TASK_SPEC as MY_TASK_SPEC, run_my_task
```

## Checklist

- [ ] `TaskType.MY_TASK` exists in `rpx_benchmark/api.py`
- [ ] Ground-truth + Prediction dataclasses for the new task exist in
      `api.py` and the loader knows how to read the manifest fields
- [ ] At least one `MetricCalculator` is registered for that task
      (see [Adding a Metric](adding-a-metric.md))
- [ ] The HF `Features` schema in `rpx_benchmark/data/features.py`
      covers the new task (lets users build Parquet shards)
- [ ] Tests under `tests/test_my_task.py` exercise the end-to-end
      path on a tiny synthetic fixture
- [ ] The parity guardrail
      (`tests/test_all_tasks_parity.py`) picks up the new task
      automatically — no change required there.

See
[`rpx_benchmark/tasks/monocular_depth.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/tasks/monocular_depth.py)
and
[`rpx_benchmark/tasks/segmentation.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/tasks/segmentation.py)
for two complete examples — both also register deployment-readiness
hooks (Temporal Stability, and for segmentation, SGC).

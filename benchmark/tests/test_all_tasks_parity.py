"""Guardrail: every :class:`TaskType` has a full set of registered
plugins.

Adding a new task to the ``TaskType`` enum must also land:

- A registered :class:`TaskSpec` (so the CLI sees it).
- At least one registered metric calculator (so the runner can score
  samples).
- A canonical ``datasets.Features`` schema (so HF shard generators
  know the column layout).
- A per-task pydantic ``SampleEntry`` schema (so strict validation
  works end-to-end).

This file keeps all four invariants enforced in one place. A missing
entry here surfaces as an obvious parametrized failure pointing at
exactly which registry is behind.
"""

from __future__ import annotations

import pytest

from rpx_benchmark.api import TaskType
from rpx_benchmark.metrics.registry import get_calculators
from rpx_benchmark.tasks.registry import available_tasks, get_task_spec


@pytest.mark.parametrize("task", list(TaskType))
def test_task_spec_registered(task: TaskType) -> None:
    spec = get_task_spec(task)
    assert spec.task is task
    assert spec.primary_metric, f"{task}: TaskSpec missing primary_metric"
    assert spec.required_modalities, f"{task}: TaskSpec missing required_modalities"


@pytest.mark.parametrize("task", list(TaskType))
def test_metric_calculator_registered(task: TaskType) -> None:
    calcs = get_calculators(task)
    assert calcs, f"{task}: no metric calculator registered"
    assert all(c.name for c in calcs), f"{task}: calculator is missing name"


@pytest.mark.parametrize("task", list(TaskType))
def test_hf_features_registered(task: TaskType) -> None:
    pytest.importorskip("datasets", minversion="2.18")
    from rpx_benchmark.data.features import RPX_FEATURES

    assert task in RPX_FEATURES, f"{task}: no Features schema"
    feats = RPX_FEATURES[task]
    for col in ("id", "rgb", "camera_pose"):
        assert col in feats, f"{task}: Features missing {col}"


@pytest.mark.parametrize("task", list(TaskType))
def test_pydantic_schema_registered(task: TaskType) -> None:
    pytest.importorskip("pydantic", minversion="2.5")
    from rpx_benchmark.schemas import SAMPLE_MODELS

    assert task in SAMPLE_MODELS, f"{task}: no pydantic SampleEntry"


def test_every_task_enum_is_in_the_registry() -> None:
    """The registry covers every TaskType — no silent drops."""
    missing = set(TaskType) - set(available_tasks())
    assert not missing, f"Tasks missing from registry: {missing}"

"""Tests for the task plugin registry."""

from __future__ import annotations

import pytest

from rpx_benchmark.api import TaskType
from rpx_benchmark.exceptions import ConfigError
from rpx_benchmark.tasks.registry import (
    available_tasks,
    get_task_spec,
    iter_task_specs,
    register_task,
    unregister_task,
)


def test_monocular_depth_is_registered():
    assert TaskType.MONOCULAR_DEPTH in available_tasks()


def test_get_task_spec_returns_complete_spec():
    spec = get_task_spec(TaskType.MONOCULAR_DEPTH)
    assert spec.task is TaskType.MONOCULAR_DEPTH
    assert spec.primary_metric == "absrel"
    assert "rgb" in spec.required_modalities
    assert "depth" in spec.required_modalities
    assert callable(spec.run)
    assert not spec.higher_is_better  # depth metrics are lower-is-better
    assert spec.display_name


def test_get_task_spec_unknown_raises_config_error():
    unregister_task(TaskType.MONOCULAR_DEPTH)
    try:
        with pytest.raises(ConfigError, match="No runner registered"):
            get_task_spec(TaskType.MONOCULAR_DEPTH)
    finally:
        from rpx_benchmark.tasks.monocular_depth import TASK_SPEC

        register_task(TASK_SPEC)


def test_duplicate_registration_raises():
    spec = get_task_spec(TaskType.MONOCULAR_DEPTH)
    with pytest.raises(ConfigError, match="already registered"):
        register_task(spec)


def test_iter_task_specs_returns_stable_order():
    first = [s.task for s in iter_task_specs()]
    second = [s.task for s in iter_task_specs()]
    assert first == second

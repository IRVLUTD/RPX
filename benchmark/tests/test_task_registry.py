"""Tests for the task plugin registry."""

from __future__ import annotations

import argparse

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
    tasks = available_tasks()
    assert TaskType.MONOCULAR_DEPTH in tasks


def test_get_task_spec_returns_complete_spec():
    spec = get_task_spec(TaskType.MONOCULAR_DEPTH)
    assert spec.task is TaskType.MONOCULAR_DEPTH
    assert spec.primary_metric == "absrel"
    assert "rgb" in spec.required_modalities
    assert "depth" in spec.required_modalities
    assert callable(spec.build_config)
    assert callable(spec.run)
    assert callable(spec.add_cli_arguments)
    assert not spec.higher_is_better  # depth metrics are lower-is-better
    assert spec.display_name


def test_get_task_spec_unknown_raises_config_error():
    # Temporarily remove MONOCULAR_DEPTH so the error path fires
    spec = unregister_task(TaskType.MONOCULAR_DEPTH)
    try:
        with pytest.raises(ConfigError, match="No runner registered"):
            get_task_spec(TaskType.MONOCULAR_DEPTH)
    finally:
        # Re-register so other tests keep working.
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


def test_add_cli_arguments_populates_subparser():
    spec = get_task_spec(TaskType.MONOCULAR_DEPTH)
    parser = argparse.ArgumentParser()
    spec.add_cli_arguments(parser)
    # Should have at least the split argument
    help_text = parser.format_help()
    assert "--split" in help_text
    assert "--model" in help_text or "--hf-checkpoint" in help_text

"""Pluggable task registry for RPX benchmark pipelines.

Each benchmark task (monocular depth, segmentation, tracking, …)
registers a :class:`TaskSpec` describing:

- **which modalities** must be downloaded from HuggingFace,
- **which metric** drives the ESD-weighted phase score,
- **how to parse CLI arguments** into a typed config,
- **how to run** the end-to-end pipeline (download → load → evaluate
  → report).

The CLI auto-discovers every registered task at startup and
generates one ``rpx bench <task>`` subcommand per entry, so adding a
new task requires zero changes to ``cli.py`` — you write a new
module under ``rpx_benchmark/tasks/`` and decorate the runner with
``@register_task(...)``.

Example
-------

::

    from rpx_benchmark.api import TaskType
    from rpx_benchmark.tasks.registry import TaskSpec, register_task

    @register_task(
        TaskSpec(
            task=TaskType.OBJECT_SEGMENTATION,
            display_name="Object Segmentation",
            primary_metric="miou",
            required_modalities=["rgb", "mask"],
            build_config=_build_seg_config,
            run=_run_segmentation,
            add_cli_arguments=_add_seg_cli_arguments,
        )
    )
    def _run_segmentation(cfg): ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..api import TaskType
from ..deployment import DeploymentReadinessReport
from ..exceptions import ConfigError
from ..logging_utils import get_logger
from ..metrics.registry import BenchmarkResult

log = get_logger(__name__)


# Result returned by every task's run() callable. Tasks that don't emit
# a deployment-readiness report can leave that slot None.
TaskRunResult = Tuple[BenchmarkResult, Optional[DeploymentReadinessReport], Dict[str, Any]]


# Function signatures used by the task registry.
BuildConfigFn = Callable[[argparse.Namespace], Any]
RunFn = Callable[[Any], TaskRunResult]
AddCliArgsFn = Callable[[argparse.ArgumentParser], None]

# Deployment-readiness hook signatures. Both accept the full list of
# predictions / samples collected by the runner and return either a
# result object or None (when the hook decides the computation does
# not apply to this slice, e.g. SGC when no depth is available).
#
# Kept generic (``Any``) because the return types live under
# :mod:`rpx_benchmark.deployment` and importing them here would create
# a circular dependency (``deployment → api`` is fine; ``registry →
# deployment → ... → registry`` must be avoided).
TemporalStabilityFn = Callable[[List[Any], List[Any], List[Any]], Any]
"""Signature: ``(predictions, samples, camera_poses) -> TemporalStabilityResult | None``."""
GeometricCoherenceFn = Callable[[List[Any], List[Any]], Any]
"""Signature: ``(predictions, samples) -> StackGeometricCoherenceResult | None``."""


@dataclass
class TaskSpec:
    """Everything the CLI + runner need to drive a task end-to-end.

    Parameters
    ----------
    task : TaskType
        Enum value identifying the task.
    display_name : str
        Human-readable name shown in CLI help and UI headers.
    primary_metric : str
        Metric key (as produced by the task's calculators) used to
        drive the ESD-weighted phase score. Lower is better unless
        :attr:`higher_is_better` is set.
    required_modalities : list[str]
        Hint describing which RPX modalities (``rgb``, ``depth``,
        ``mask``, …) this task needs. Used for docs and for the hub
        downloader; the per-task run function is responsible for the
        actual download.
    build_config : callable
        ``(argparse.Namespace) -> config``. Converts CLI args into a
        typed config dataclass the task's ``run`` function expects.
    run : callable
        ``(config) -> (BenchmarkResult, DeploymentReadinessReport | None, {paths})``.
        The end-to-end pipeline entrypoint.
    add_cli_arguments : callable
        ``(argparse.ArgumentParser) -> None``. Populates a subparser
        with the task's flags. The CLI calls this once when building
        its parser tree.
    higher_is_better : bool
        When True, the primary metric is interpreted higher-is-better
        in reports and deltas. Default False (error metrics).
    description : str
        One-line description used in subparser help.
    """

    task: TaskType
    display_name: str
    primary_metric: str
    required_modalities: List[str]
    build_config: BuildConfigFn
    run: RunFn
    add_cli_arguments: AddCliArgsFn
    higher_is_better: bool = False
    description: str = ""
    temporal_stability_fn: Optional[TemporalStabilityFn] = None
    """Optional deployment-readiness hook: compute Temporal Stability.

    When set, the benchmark runner invokes this callable with the full
    per-sample ``(predictions, samples, camera_poses)`` lists after the
    prediction loop. Returning ``None`` is equivalent to "not
    applicable for this run" (e.g. < 2 samples). Depth and
    segmentation task modules ship defaults; new tasks can add their
    own without touching :mod:`rpx_benchmark.runner`.
    """
    geometric_coherence_fn: Optional[GeometricCoherenceFn] = None
    """Optional deployment-readiness hook: compute Stack Geometric Coherence.

    Symmetric to :attr:`temporal_stability_fn` — the runner calls it
    with ``(predictions, samples)`` and expects a
    :class:`StackGeometricCoherenceResult` or ``None``.
    """


_TASK_REGISTRY: Dict[TaskType, TaskSpec] = {}


def register_task(spec: TaskSpec) -> TaskSpec:
    """Install a :class:`TaskSpec` into the global registry.

    Returns the spec unchanged, so it can be used as a decorator or a
    direct call.

    Raises
    ------
    ConfigError
        If another spec is already registered under the same task.
    """
    if spec.task in _TASK_REGISTRY:
        raise ConfigError(
            f"Task {spec.task.value!r} is already registered.",
            hint="Call unregister_task() first if you intended to replace it.",
        )
    _TASK_REGISTRY[spec.task] = spec
    log.debug("registered task %s (primary_metric=%s)",
              spec.task.value, spec.primary_metric)
    return spec


def unregister_task(task: TaskType) -> bool:
    """Remove a registered task. Returns True if something was removed."""
    return _TASK_REGISTRY.pop(task, None) is not None


def get_task_spec(task: TaskType) -> TaskSpec:
    """Look up a registered spec.

    Raises
    ------
    ConfigError
        If the task is not registered. The error lists every task
        that *is* registered so the user can quickly spot typos.
    """
    if task not in _TASK_REGISTRY:
        raise ConfigError(
            f"No runner registered for task {task.value!r}.",
            hint=(
                "Registered tasks: "
                + ", ".join(sorted(t.value for t in _TASK_REGISTRY))
            ),
        )
    return _TASK_REGISTRY[task]


def available_tasks() -> List[TaskType]:
    """Return the sorted list of registered task enum values."""
    return sorted(_TASK_REGISTRY.keys(), key=lambda t: t.value)


def iter_task_specs() -> List[TaskSpec]:
    """Return every registered spec in stable order."""
    return [_TASK_REGISTRY[t] for t in available_tasks()]


__all__ = [
    "TaskSpec",
    "TaskRunResult",
    "register_task",
    "unregister_task",
    "get_task_spec",
    "available_tasks",
    "iter_task_specs",
]

"""Task plugin registry.

Each task module registers a :class:`TaskSpec` describing the primary
metric, required modalities, and two optional deployment-readiness
hooks. The runner looks these up and dispatches through them — it
never branches on task identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..api import TaskType
from ..deployment import DeploymentReadinessReport
from ..exceptions import ConfigError
from ..logging_utils import get_logger
from ..metrics.registry import BenchmarkResult

log = get_logger(__name__)


TaskRunResult = Tuple[BenchmarkResult, Optional[DeploymentReadinessReport], Dict[str, Any]]

# Hook signatures. Kept as ``Callable[..., Any]`` so importing this
# module does not drag the ``deployment`` result types through — the
# return types are documented on the attribute docstrings.
TemporalStabilityFn = Callable[[List[Any], List[Any], List[Any]], Any]
"""Signature: ``(predictions, samples, camera_poses) -> TemporalStabilityResult | None``."""
GeometricCoherenceFn = Callable[[List[Any], List[Any]], Any]
"""Signature: ``(predictions, samples) -> StackGeometricCoherenceResult | None``."""


@dataclass
class TaskSpec:
    """Per-task plugin record.

    Attributes
    ----------
    task, display_name, description
        Identification.
    primary_metric
        Metric key driving the ESD-weighted phase score.
    required_modalities
        Hint for hub download scope (``rgb``, ``depth``, ``mask`` …).
    higher_is_better
        Direction of the primary metric.
    run
        End-to-end callable ``(cfg) -> (BenchmarkResult, report, paths)``.
    temporal_stability_fn, geometric_coherence_fn
        Optional deployment-readiness hooks. The runner calls them if
        set; tasks without a hook skip that metric silently.
    """

    task: TaskType
    display_name: str
    primary_metric: str
    required_modalities: List[str]
    run: Callable[[Any], TaskRunResult]
    higher_is_better: bool = False
    description: str = ""
    temporal_stability_fn: Optional[TemporalStabilityFn] = None
    geometric_coherence_fn: Optional[GeometricCoherenceFn] = None


_TASK_REGISTRY: Dict[TaskType, TaskSpec] = {}


def register_task(spec: TaskSpec) -> TaskSpec:
    if spec.task in _TASK_REGISTRY:
        raise ConfigError(
            f"Task {spec.task.value!r} is already registered.",
            hint="Call unregister_task() first to replace it.",
        )
    _TASK_REGISTRY[spec.task] = spec
    log.debug("registered task %s (primary=%s)", spec.task.value, spec.primary_metric)
    return spec


def unregister_task(task: TaskType) -> bool:
    return _TASK_REGISTRY.pop(task, None) is not None


def get_task_spec(task: TaskType) -> TaskSpec:
    if task not in _TASK_REGISTRY:
        raise ConfigError(
            f"No runner registered for task {task.value!r}.",
            hint="Registered: " + ", ".join(sorted(t.value for t in _TASK_REGISTRY)),
        )
    return _TASK_REGISTRY[task]


def available_tasks() -> List[TaskType]:
    return sorted(_TASK_REGISTRY.keys(), key=lambda t: t.value)


def iter_task_specs() -> List[TaskSpec]:
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

"""Core of the metric plugin system.

Contains the :class:`MetricCalculator` base class, the task→calculators
registry, the ``@register_metric`` decorator, and a thin
:class:`MetricSuite` facade that the runner uses.

Most library users should not touch this module directly — instead
import from :mod:`rpx_benchmark.metrics`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

import numpy as np

from ..api import TaskType
from ..exceptions import MetricError
from ..logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #


class MetricCalculator(ABC):
    """Abstract base for a family of metrics bound to a single task.

    Subclasses are lightweight, stateless objects. The ``compute``
    method accepts one prediction and its ground truth and returns a
    dict mapping metric name → scalar float.

    Subclasses must set ``name`` (human-readable identifier used for
    registration) and implement :meth:`compute`. Subclasses should
    take their own hyperparameters via ``__init__`` when needed.

    Examples
    --------
    >>> from rpx_benchmark.metrics import MetricCalculator, register_metric
    >>> from rpx_benchmark.api import TaskType
    >>> @register_metric(TaskType.MONOCULAR_DEPTH)
    ... class MeanPred(MetricCalculator):
    ...     name = "mean_pred"
    ...     def compute(self, prediction, ground_truth):
    ...         return {"mean_pred": float(prediction.depth_map.mean())}
    """

    name: str = ""

    @abstractmethod
    def compute(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Return a dict of metric name → scalar for one sample.

        Parameters
        ----------
        prediction : Any
            Task-specific Prediction dataclass
            (e.g. :class:`DepthPrediction`). The concrete type is
            whatever the calculator is designed for.
        ground_truth : Any
            Task-specific GroundTruth dataclass
            (e.g. :class:`DepthGroundTruth`).

        Returns
        -------
        dict[str, float]
            Metric values. Keys should be short snake_case names;
            values must be numeric so the runner can average them.

        Raises
        ------
        MetricError
            When inputs have the wrong shape/type/content.
        """


# --------------------------------------------------------------------------- #
# Registry state
# --------------------------------------------------------------------------- #

_CALCULATORS: Dict[TaskType, List[MetricCalculator]] = defaultdict(list)


def register_metric(
    task: TaskType,
) -> Callable[[Type[MetricCalculator]], Type[MetricCalculator]]:
    """Class decorator registering a calculator under a task.

    Parameters
    ----------
    task : TaskType
        Task the calculator belongs to. Multiple calculators can be
        registered against the same task and will be composed at
        evaluation time.

    Returns
    -------
    Callable
        The decorator; when applied to a class it instantiates it
        once with no arguments and appends to the registry.

    Raises
    ------
    MetricError
        If ``task`` is not a :class:`TaskType`, or if the decorated
        object is not a :class:`MetricCalculator` subclass.

    Examples
    --------
    >>> from rpx_benchmark.api import TaskType
    >>> from rpx_benchmark.metrics import MetricCalculator, register_metric
    >>> @register_metric(TaskType.MONOCULAR_DEPTH)
    ... class MyMetric(MetricCalculator):
    ...     name = "my_metric"
    ...     def compute(self, prediction, ground_truth):
    ...         return {"my_metric": 0.0}
    """
    if not isinstance(task, TaskType):
        raise MetricError(
            f"register_metric expected a TaskType, got {type(task).__name__}",
            hint="Use one of the members of rpx_benchmark.api.TaskType.",
        )

    def decorator(cls: Type[MetricCalculator]) -> Type[MetricCalculator]:
        if not (isinstance(cls, type) and issubclass(cls, MetricCalculator)):
            raise MetricError(
                f"@register_metric can only decorate MetricCalculator subclasses; got {cls!r}",
            )
        instance = cls()
        _CALCULATORS[task].append(instance)
        log.debug("registered metric %s for task %s", instance.name or cls.__name__, task.value)
        return cls

    return decorator


def unregister_metric(task: TaskType, name: str) -> bool:
    """Remove a previously registered calculator by its ``name`` field.

    Parameters
    ----------
    task : TaskType
    name : str
        Name attribute of the calculator to remove.

    Returns
    -------
    bool
        True if a calculator was removed; False if no match was found.
    """
    before = len(_CALCULATORS.get(task, []))
    _CALCULATORS[task] = [c for c in _CALCULATORS.get(task, []) if c.name != name]
    after = len(_CALCULATORS[task])
    return after < before


def clear_registry() -> None:
    """Remove every registered calculator. Primarily for tests."""
    _CALCULATORS.clear()


def get_calculators(task: TaskType) -> List[MetricCalculator]:
    """Return the calculators registered for ``task`` (empty list if none)."""
    return list(_CALCULATORS.get(task, []))


def available_metrics() -> Dict[TaskType, List[str]]:
    """List the registered metric names grouped by task.

    Returns
    -------
    dict
        ``{TaskType: [calculator_name, ...]}``. Only tasks with at
        least one registered calculator appear.
    """
    return {task: [c.name for c in calcs] for task, calcs in _CALCULATORS.items() if calcs}


def compute_metrics(
    task: TaskType,
    prediction: Any,
    ground_truth: Any,
) -> Dict[str, float]:
    """Run every registered calculator for ``task`` and merge outputs.

    Parameters
    ----------
    task : TaskType
    prediction : Any
    ground_truth : Any

    Returns
    -------
    dict[str, float]
        Union of all calculators' output dicts. Later calculators may
        overwrite earlier ones if they emit the same key.

    Raises
    ------
    MetricError
        If no calculators are registered for ``task``.
    """
    calcs = _CALCULATORS.get(task)
    if not calcs:
        raise MetricError(
            f"No metric calculator registered for task {task.value!r}.",
            hint=(
                "Either import the built-in calculators (e.g. "
                "`import rpx_benchmark.metrics`) or register a custom "
                "one via @register_metric."
            ),
        )
    merged: Dict[str, float] = {}
    for calc in calcs:
        merged.update(calc.compute(prediction, ground_truth))
    return merged


# --------------------------------------------------------------------------- #
# Facade: MetricSuite
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkResult:
    """Outcome of running a :class:`BenchmarkRunner` against a dataset.

    Attributes
    ----------
    task : TaskType
        Which task was evaluated.
    per_sample : list of dict
        One dict per sample. Each dict mixes metric keys (numeric) and
        metadata keys (``id``, ``phase``, ``difficulty``, ``scene``)
        that :class:`MetricSuite.aggregate` silently skips when
        computing means.
    aggregated : dict[str, float]
        Mean over the numeric metric keys in :attr:`per_sample`.
    num_samples : int
    """

    task: TaskType
    per_sample: List[Dict[str, Any]]
    aggregated: Dict[str, float]
    num_samples: int


class MetricSuite:
    """Thin wrapper around the metric registry used by the runner.

    Kept as a class rather than a function because historical API
    expects ``MetricSuite.for_task(...).evaluate(pred, gt)``. New code
    can call :func:`compute_metrics` directly.
    """

    def __init__(self, task: TaskType) -> None:
        self.task = task

    @classmethod
    def for_task(cls, task: TaskType) -> "MetricSuite":
        """Create a suite for the given task.

        Raises
        ------
        MetricError
            If no calculators are registered for ``task``.
        """
        if not get_calculators(task):
            raise MetricError(
                f"No metric calculator registered for task {task.value!r}.",
                hint="Import rpx_benchmark.metrics to trigger built-ins.",
            )
        return cls(task=task)

    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Run every registered calculator and return merged results.

        Raises
        ------
        MetricError
            Propagated from individual calculators when inputs are
            shape-mismatched or wrong-typed.
        """
        return compute_metrics(self.task, prediction, ground_truth)

    def aggregate(self, per_sample: List[Dict[str, Any]]) -> Dict[str, float]:
        """Mean over numeric metric keys; non-numeric metadata is skipped.

        Parameters
        ----------
        per_sample : list of dict
            Per-sample rows. May contain metric floats and metadata
            strings/enums in the same dict.

        Returns
        -------
        dict[str, float]
            One float per numeric key. Empty dict if ``per_sample`` is
            empty or contains no numeric values.
        """
        if not per_sample:
            return {}
        numeric_keys = [
            k
            for k, v in per_sample[0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        out: Dict[str, float] = {}
        for k in numeric_keys:
            vals = [
                m[k]
                for m in per_sample
                if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)
            ]
            if vals:
                out[k] = float(np.mean(vals))
        return out

    def build_result(self, per_sample: List[Dict[str, Any]]) -> "BenchmarkResult":
        """Convenience: wrap ``per_sample`` and its aggregate in a
        :class:`BenchmarkResult`.
        """
        return BenchmarkResult(
            task=self.task,
            per_sample=per_sample,
            aggregated=self.aggregate(per_sample),
            num_samples=len(per_sample),
        )


__all__ = [
    "MetricCalculator",
    "MetricSuite",
    "BenchmarkResult",
    "available_metrics",
    "clear_registry",
    "compute_metrics",
    "get_calculators",
    "register_metric",
    "unregister_metric",
]

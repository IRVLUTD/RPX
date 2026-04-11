"""Task-level entrypoints for RPX benchmark pipelines.

Importing this package triggers :mod:`monocular_depth` (and in the
future, other task modules) which self-register their
:class:`TaskSpec` with :mod:`rpx_benchmark.tasks.registry`. After
import, :func:`available_tasks` lists everything the CLI can run.
"""

from .monocular_depth import (
    MonocularDepthRunConfig,
    run_monocular_depth,
    TASK_SPEC as MONOCULAR_DEPTH_SPEC,
)
from .segmentation import (
    SegmentationRunConfig,
    run_segmentation,
    TASK_SPEC as SEGMENTATION_SPEC,
)
from .registry import (
    TaskRunResult,
    TaskSpec,
    available_tasks,
    get_task_spec,
    iter_task_specs,
    register_task,
    unregister_task,
)

__all__ = [
    # Task runners
    "MonocularDepthRunConfig",
    "run_monocular_depth",
    "MONOCULAR_DEPTH_SPEC",
    "SegmentationRunConfig",
    "run_segmentation",
    "SEGMENTATION_SPEC",
    # Registry
    "TaskSpec",
    "TaskRunResult",
    "register_task",
    "unregister_task",
    "get_task_spec",
    "available_tasks",
    "iter_task_specs",
]

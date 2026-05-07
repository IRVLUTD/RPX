"""Object tracking: per-frame MOT with stable track IDs."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "mota"


@dataclass
class ObjectTrackingRunConfig(TaskRunConfig):
    pass


def run_object_tracking(cfg: ObjectTrackingRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.OBJECT_TRACKING,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=False,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_TRACKING,
    display_name="Object Tracking",
    description="Per-frame multi-object tracking with persistent IDs.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "mask", "tracklets"],
    higher_is_better=True,
    run=run_object_tracking,
)

register_task(TASK_SPEC)

"""Relative camera pose between two frames."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "rotation_error_deg"


@dataclass
class RelativePoseRunConfig(TaskRunConfig):
    pass


def run_relative_pose(cfg: RelativePoseRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.RELATIVE_CAMERA_POSE,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=False,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.RELATIVE_CAMERA_POSE,
    display_name="Relative Camera Pose",
    description="6-DoF pose of frame B relative to frame A.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "pose"],
    higher_is_better=False,
    run=run_relative_pose,
)

register_task(TASK_SPEC)

"""Keypoint matching between a pair of RGB frames."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "keypoint_acc"


@dataclass
class KeypointMatchingRunConfig(TaskRunConfig):
    pass


def run_keypoint_matching(cfg: KeypointMatchingRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.KEYPOINT_MATCHING,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=False,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.KEYPOINT_MATCHING,
    display_name="Keypoint Matching",
    description="Corresponding points between two RGB frames.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "keypoints"],
    higher_is_better=True,
    run=run_keypoint_matching,
)

register_task(TASK_SPEC)

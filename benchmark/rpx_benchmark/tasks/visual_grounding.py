"""Visual grounding: referring expression -> box."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "grounding_acc"


@dataclass
class VisualGroundingRunConfig(TaskRunConfig):
    pass


def run_visual_grounding(cfg: VisualGroundingRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.VISUAL_GROUNDING,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.VISUAL_GROUNDING,
    display_name="Visual Grounding",
    description="Predict the box for a referring expression on an RGB frame.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "questionnaires"],
    higher_is_better=True,
    run=run_visual_grounding,
)

register_task(TASK_SPEC)

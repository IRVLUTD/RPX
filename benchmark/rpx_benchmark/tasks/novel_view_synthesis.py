"""Novel view synthesis: RGB at a held-out target pose."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "psnr"


@dataclass
class NovelViewSynthesisRunConfig(TaskRunConfig):
    pass


def run_novel_view_synthesis(cfg: NovelViewSynthesisRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.NOVEL_VIEW_SYNTHESIS,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=False,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.NOVEL_VIEW_SYNTHESIS,
    display_name="Novel View Synthesis",
    description="Synthesise RGB at a held-out target camera pose.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth", "pose"],
    higher_is_better=True,
    run=run_novel_view_synthesis,
)

register_task(TASK_SPEC)

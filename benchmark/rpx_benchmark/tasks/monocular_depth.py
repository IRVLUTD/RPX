"""Monocular absolute depth."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "absrel"


@dataclass
class MonocularDepthRunConfig(TaskRunConfig):
    pass


def run_monocular_depth(cfg: MonocularDepthRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.MONOCULAR_DEPTH,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=True,
    )


def _temporal_stability_hook(predictions, samples, camera_poses):
    from ..deployment import compute_temporal_stability_depth

    return compute_temporal_stability_depth([p.depth_map for p in predictions], camera_poses)


TASK_SPEC = TaskSpec(
    task=TaskType.MONOCULAR_DEPTH,
    display_name="Monocular Absolute Depth",
    description="Metric depth (metres) from a single RGB frame.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth"],
    higher_is_better=False,
    run=run_monocular_depth,
    temporal_stability_fn=_temporal_stability_hook,
)

register_task(TASK_SPEC)

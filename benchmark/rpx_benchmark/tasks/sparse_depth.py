"""Sparse depth: predict depth at a set of pixel locations."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "rmse"


@dataclass
class SparseDepthRunConfig(TaskRunConfig):
    pass


def run_sparse_depth(cfg: SparseDepthRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.SPARSE_DEPTH,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.SPARSE_DEPTH,
    display_name="Sparse Depth",
    description="Predict depth at a given set of pixel locations.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth", "sparse_depth"],
    higher_is_better=False,
    run=run_sparse_depth,
)

register_task(TASK_SPEC)

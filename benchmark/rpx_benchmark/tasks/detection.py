"""Object detection (closed vocab + open vocab)."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "f1"


@dataclass
class ObjectDetectionRunConfig(TaskRunConfig):
    pass


def run_object_detection(cfg: ObjectDetectionRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.OBJECT_DETECTION,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
    )


def run_open_vocab_detection(cfg: ObjectDetectionRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.OPEN_VOCAB_DETECTION,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_DETECTION,
    display_name="Object Detection",
    description="Detect and label boxes on a single RGB frame (closed vocab).",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "boxes"],
    higher_is_better=True,
    run=run_object_detection,
)

OPEN_VOCAB_TASK_SPEC = TaskSpec(
    task=TaskType.OPEN_VOCAB_DETECTION,
    display_name="Open-Vocabulary Detection",
    description="Detect objects given a free-text vocabulary.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "boxes"],
    higher_is_better=True,
    run=run_open_vocab_detection,
)

register_task(TASK_SPEC)
register_task(OPEN_VOCAB_TASK_SPEC)

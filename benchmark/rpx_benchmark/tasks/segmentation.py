"""Object (instance) segmentation."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "miou"


@dataclass
class SegmentationRunConfig(TaskRunConfig):
    pass


def run_segmentation(cfg: SegmentationRunConfig) -> PipelineResult:
    return run_pipeline(
        task=TaskType.OBJECT_SEGMENTATION,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        compute_ts=True,
    )


def _temporal_stability_hook(predictions, samples, camera_poses):
    from ..deployment import compute_temporal_stability_seg

    return compute_temporal_stability_seg([p.mask for p in predictions], camera_poses)


def _geometric_coherence_hook(predictions, samples):
    """SGC needs paired mask + depth. Depth arrives via sample.metadata."""
    import numpy as np

    from ..deployment import compute_sgc

    pairs = [
        (p.mask, s.metadata["depth_map"])
        for p, s in zip(predictions, samples, strict=False)
        if s.metadata and s.metadata.get("depth_map") is not None
    ]
    if not pairs:
        return None
    masks = [m for m, _ in pairs]
    depths = [np.asarray(d, dtype=np.float32) for _, d in pairs]
    return compute_sgc(masks, depths)


TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_SEGMENTATION,
    display_name="Object Segmentation",
    description="Instance mask from a single RGB frame; scored with mIoU.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "mask"],
    higher_is_better=True,
    run=run_segmentation,
    temporal_stability_fn=_temporal_stability_hook,
    geometric_coherence_fn=_geometric_coherence_hook,
)

register_task(TASK_SPEC)

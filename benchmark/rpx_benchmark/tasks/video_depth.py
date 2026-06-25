"""Video absolute depth (paper task D1-V).

Per-clip metric depth from a full ``(scene, phase)`` RGB sequence. The
D1-V roster (DepthCrafter, ChronoDepth, RollingDepth, MonST3R, VGGT-Ω,
DA3, etc.) consumes the whole clip in one inference call and emits a
per-frame depth sequence.

The end-to-end runner is :func:`run_video_pipeline`
(:mod:`rpx_benchmark.tasks._video_pipeline`); this module wraps it
with the D1-V task spec and registers it in the task registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult
from ._video_pipeline import VideoTaskRunConfig, run_video_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "absrel"


@dataclass
class VideoDepthRunConfig(VideoTaskRunConfig):
    """Knobs for a D1-V run.

    Inherits ``frame_budget`` and ``sampling`` from
    :class:`~rpx_benchmark.tasks._video_pipeline.VideoTaskRunConfig` for
    the temporal-resolution ablation (paper §5.2); leave the defaults
    (``frame_budget=None, sampling="all"``) for the headline run.
    """


def run_video_depth(cfg: VideoDepthRunConfig) -> PipelineResult:
    """End-to-end D1-V pipeline.

    Delegates to :func:`run_video_pipeline`, which handles download →
    :class:`~rpx_benchmark.video_loader.D1VDataset` build → per-clip
    predict → per-clip ``(s, t)`` alignment (for relative-depth
    models) →
    :class:`~rpx_benchmark.metrics.video_depth.VideoDepthErrorMetrics`
    + :class:`~rpx_benchmark.metrics.video_depth.VideoDepthTemporalMetrics`
    → cell-log row → JSON / markdown summary.
    """
    return run_video_pipeline(
        task=TaskType.VIDEO_DEPTH,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.VIDEO_DEPTH,
    display_name="Video Absolute Depth",
    description=(
        "Metric depth (metres) from the full ~250-frame phase RGB clip. "
        "Differs from MONOCULAR_DEPTH in iteration unit (per-clip) and "
        "metric set (adds temporal consistency metrics on top of D1-F's "
        "per-frame metrics)."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth"],
    higher_is_better=False,
    run=run_video_depth,
)

register_task(TASK_SPEC)

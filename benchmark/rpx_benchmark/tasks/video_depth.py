"""Video absolute depth (paper task D1-V).

Per-clip metric depth from a full ``(scene, phase)`` RGB sequence. The
D1-V roster (DepthCrafter, ChronoDepth, RollingDepth, MonST3R, VGGT-Ω,
DA3 etc.) consumes the whole clip in one inference call and emits a
per-frame depth sequence.

This module registers a ``TASK_SPEC`` so the runner registry knows
about D1-V — required for the "every TaskType is in the registry"
parity invariant.

The ``run`` callable currently raises ``ConfigError`` because the
full video-pipeline runner (the analogue of
:func:`rpx_benchmark.tasks._pipeline.run_pipeline` adapted for per-clip
iteration over :class:`~rpx_benchmark.video_loader.D1VDataset`) is
landing in a follow-up PR with the first model adapter (DA3). Every
other piece — the per-clip dataset, the metric calculators, the cell
log column, the manifest writer — is already on ``main``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ..exceptions import ConfigError
from ._pipeline import PipelineResult, TaskRunConfig
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "absrel"


@dataclass
class VideoDepthRunConfig(TaskRunConfig):
    """Knobs for a D1-V run. Mirrors :class:`MonocularDepthRunConfig`
    today; will grow ``frame_budget`` / ``sampling`` knobs once the
    runner consumes :class:`~rpx_benchmark.video_loader.D1VDataset`."""


def run_video_depth(cfg: VideoDepthRunConfig) -> PipelineResult:
    """End-to-end D1-V pipeline. Not yet wired.

    The infrastructure pieces are in place:

    * :class:`~rpx_benchmark.video_loader.D1VDataset` yields
      ``VideoSample`` per (scene, phase) clip.
    * :class:`~rpx_benchmark.metrics.video_depth.VideoDepthErrorMetrics`
      and ``VideoDepthTemporalMetrics`` are registered against
      :data:`~rpx_benchmark.api.TaskType.VIDEO_DEPTH`.
    * The cell-log writer accepts ``frame_budget``.

    Missing piece: the video-pipeline runner itself — the analogue of
    :func:`rpx_benchmark.tasks._pipeline.run_pipeline` adapted for
    per-clip iteration. Lands with the first D1-V model adapter (DA3).
    """
    raise ConfigError(
        "run_video_depth is registered but not yet implemented",
        hint=(
            "D1-V runner is pending the first model adapter integration. "
            "Track progress in benchmark/docs/depth_metric_decisions.md."
        ),
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

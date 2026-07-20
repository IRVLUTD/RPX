"""Video absolute depth (paper task Video Depth).

Per-clip metric depth from a full ``(scene, phase)`` RGB sequence. The
Video Depth roster (DepthCrafter, ChronoDepth, RollingDepth, MonST3R, VGGT-Ω,
DA3, etc.) consumes the whole clip in one inference call and emits a
per-frame depth sequence.

The end-to-end runner is :func:`run_video_pipeline`
(:mod:`rpx_benchmark.tasks._video_pipeline`); this module wraps it
with the Video Depth task spec and registers it in the task registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TaskType
from ._pipeline import PipelineResult
from ._video_pipeline import VideoTaskRunConfig, run_video_pipeline
from .registry import TaskSpec, register_task

PRIMARY_METRIC = "absrel"

# --------------------------------------------------------------------------- #
# MANOVA K-vectors (paper §3.2)
# --------------------------------------------------------------------------- #
#
# The K-vector is the set of metrics standardised across all 3N (scene, phase)
# cells before running the repeated-measures MANOVA that yields Φ. K must stay
# safely below N (scenes) and metrics inside must not be collinear or the
# within-phase SSCP E becomes singular. See :func:`rpx_benchmark.phi.compute_phi_oneway`.
#
# Locked K-vectors — RGB-D-only constraint (see SESSION_HANDOFF.md):
#
# Depth-estimation tasks in RPX use RGB-D only. Even though the dataset
# captures T265 poses, fisheye stereo, and other modalities, the depth
# tasks intentionally exclude them — both from model input and from
# metric inputs. This rules out any metric that needs poses (TAE, ATE)
# or external models (OPW/RAFT flow, MFC/Sintel flow).
#
#   * Image Depth K = 5: (absrel, rmse, delta1, silog, fscore_5cm).
#       Grasp F@5cm needs only GT depth + D435 intrinsics, so it stays.
#
#   * Video Depth K = 6: (absrel, rmse, delta1, silog, tgm, tgse).
#       Two temporal metrics — TGM (L1 of |Δd_pred|−|Δd_gt| in static
#       regions) and TGSE (L2 signed variant) — both pose-free and
#       flow-free, needing only pred + GT depth. fscore_5cm drops from
#       the video K-vector into diagnostics because the temporal-slot
#       tradeoff prioritises the differentiating axis for video models.
#
# TAE, OPW, fscore_5cm (for D1-V), TCC, TMC, δ₂/δ₃, and range-stratified
# variants remain registered as diagnostics — the calculators still emit
# them where possible, they just don't enter Φ.
FRAME_DEPTH_MANOVA_METRICS: tuple[str, ...] = (
    "absrel", "rmse", "delta1", "silog", "fscore_5cm",
)
VIDEO_DEPTH_MANOVA_METRICS: tuple[str, ...] = (
    "absrel", "rmse", "delta1", "silog", "tgm", "tgse",
)

# Paper-facing aliases (D1-F / D1-V naming used in tables and figures).
D1F_MANOVA_METRICS = FRAME_DEPTH_MANOVA_METRICS
D1V_MANOVA_METRICS = VIDEO_DEPTH_MANOVA_METRICS

D1F_TASK = "monocular_depth"
D1V_TASK = VIDEO_DEPTH_TASK = "video_depth"


@dataclass
class VideoDepthRunConfig(VideoTaskRunConfig):
    """Knobs for a Video Depth run.

    Inherits ``frame_budget`` and ``sampling`` from
    :class:`~rpx_benchmark.tasks._video_pipeline.VideoTaskRunConfig` for
    the temporal-resolution ablation (paper §5.2); leave the defaults
    (``frame_budget=None, sampling="all"``) for the headline run.
    """


def run_video_depth(cfg: VideoDepthRunConfig) -> PipelineResult:
    """End-to-end Video Depth pipeline.

    Delegates to :func:`run_video_pipeline`, which handles download →
    :class:`~rpx_benchmark.video_loader.VideoDepthDataset` build → per-clip
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
        "metric set (adds temporal consistency metrics on top of Image Depth's "
        "per-frame metrics)."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth"],
    higher_is_better=False,
    run=run_video_depth,
)

register_task(TASK_SPEC)

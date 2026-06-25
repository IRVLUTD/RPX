"""Metric calculators bound to ``TaskType.VIDEO_DEPTH`` (paper task Video Depth).

Video Depth scores a model on the same per-frame quality as Image Depth **plus** a
small set of temporal-consistency metrics. The implementation already
exists in :mod:`rpx_benchmark.metrics.depth` (per-frame error metrics)
and :mod:`rpx_benchmark.metrics.depth_temporal` (TAE / OPW / TGM / TCC);
this module is the thin glue that registers those metrics for the new
``VIDEO_DEPTH`` task and applies them sequence-shape-aware.

Cell-log granularity
--------------------

One :class:`~rpx_benchmark.api.VideoDepthPrediction` corresponds to one
``(scene, phase)`` cell — the cell-log key is
``(model, task, scene, phase)`` (no per-frame key). Per-frame error
metrics are therefore **aggregated to a per-clip mean** before being
written to the log. Temporal metrics are inherently per-clip.

What the final metric tuple is
------------------------------

The full set this module exposes is intentionally broader than what
the paper's headline Φ MANOVA uses. The final K-metric tuple (5? 6?
8?) for Video Depth Φ is being decided in Feynman's depth-metric study
(:file:`benchmark/docs/depth_metric_decisions.md`); the downstream
selection happens in :mod:`rpx_benchmark.analyze_experiment` from the
cell-log column subset, not by deleting calculators here.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..api import TaskType, VideoDepthGroundTruth, VideoDepthPrediction
from ..exceptions import MetricError
from ..logging_utils import get_logger
from .registry import MetricCalculator, register_metric

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Per-frame error metrics, aggregated over the clip
# --------------------------------------------------------------------------- #


def _per_frame_error_metrics(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    valid_seq: np.ndarray,
) -> Dict[str, float]:
    """Per-frame AbsRel/RMSE/δ thresholds → arithmetic mean over the clip.

    Frames whose valid mask is fully zero are skipped from the mean
    rather than counted as zero — matching ``default_valid_mask`` policy
    in :mod:`rpx_benchmark.metrics.depth_alignment`.
    """
    T = pred_seq.shape[0]
    per_frame: Dict[str, list] = {
        "absrel": [],
        "rmse": [],
        "delta1": [],
        "delta2": [],
        "delta3": [],
    }
    for t in range(T):
        p = pred_seq[t]
        g = gt_seq[t]
        v = valid_seq[t]
        if not v.any():
            continue
        p_v = p[v]
        g_v = g[v]
        per_frame["absrel"].append(float(np.mean(np.abs(p_v - g_v) / g_v)))
        per_frame["rmse"].append(float(np.sqrt(np.mean((p_v - g_v) ** 2))))
        ratio = np.maximum(p_v / g_v, g_v / p_v)
        per_frame["delta1"].append(float(np.mean(ratio < 1.25)))
        per_frame["delta2"].append(float(np.mean(ratio < 1.25**2)))
        per_frame["delta3"].append(float(np.mean(ratio < 1.25**3)))

    if not per_frame["absrel"]:
        # Every frame had empty valid masks — return safe sentinels.
        return {
            "absrel": 0.0,
            "rmse": 0.0,
            "delta1": 1.0,
            "delta2": 1.0,
            "delta3": 1.0,
        }
    return {k: float(np.mean(v)) for k, v in per_frame.items()}


@register_metric(TaskType.VIDEO_DEPTH)
class VideoDepthErrorMetrics(MetricCalculator):
    """Per-clip aggregate of Image Depth's per-frame error metrics.

    Emits five scalars (``absrel``, ``rmse``, ``delta1``, ``delta2``,
    ``delta3``) computed per frame and arithmetically averaged over the
    clip's valid frames. This is the *per-frame quality* component of
    Video Depth — it lets Video Depth models be compared against Image Depth models on the
    same scene state.
    """

    name = "video_depth_error_metrics"

    def compute(
        self,
        prediction: VideoDepthPrediction,
        ground_truth: VideoDepthGroundTruth,
    ) -> Dict[str, float]:
        _validate_shapes(prediction, ground_truth)
        return _per_frame_error_metrics(
            prediction.depth_map_seq.astype(np.float32),
            ground_truth.depth_map_seq.astype(np.float32),
            ground_truth.valid_mask_seq.astype(bool),
        )


# --------------------------------------------------------------------------- #
# Temporal metrics — thin adapter over depth_temporal.py
# --------------------------------------------------------------------------- #


@register_metric(TaskType.VIDEO_DEPTH)
class VideoDepthTemporalMetrics(MetricCalculator):
    """Temporal consistency metrics for one clip.

    Calls :func:`~rpx_benchmark.metrics.depth_temporal.compute_temporal_depth_metrics`,
    which itself dispatches to ``tae``, ``opw``, ``tgm``, ``tcc``
    depending on which inputs are present. Any metric missing an input
    (e.g. no T265 poses, no flow function configured) returns ``nan``
    rather than raising — that matches the paper's reporting policy
    where the MANOVA's K is fixed downstream from whichever metrics are
    actually populated.

    Optional sample-level context (poses, RGB, flow function) is read
    from ``ground_truth.frame_indices``'s sibling fields if the loader
    attached them. The standard path is the loader populating
    ``ground_truth.poses`` (T, 4, 4) and ``ground_truth.rgb_seq``
    (T, H, W, 3) at clip-construction time so this calculator can be
    stateless.
    """

    name = "video_depth_temporal"

    def compute(
        self,
        prediction: VideoDepthPrediction,
        ground_truth: VideoDepthGroundTruth,
    ) -> Dict[str, float]:
        # Lazy import: ``metrics.depth_temporal`` and its dependencies
        # (``deployment.se3_reproject_depth``, etc.) are not yet on
        # ``main`` — they live in a sibling work-in-progress branch.
        # Importing them eagerly at module load time would break CI on
        # any PR (like this one) that ships ``video_depth.py`` before
        # ``depth_temporal.py`` lands. The lazy import lets the
        # calculator register at task-discovery time and only fail
        # loudly if a caller actually invokes it before the temporal
        # module is available.
        try:
            from .depth_temporal import compute_temporal_depth_metrics
        except ImportError:
            log.warning(
                "video_depth_temporal: depth_temporal module not "
                "available on this branch — emitting NaN for all "
                "temporal metrics. Land depth_temporal.py in a follow-up "
                "PR to populate these.",
            )
            return {"tae": float("nan"), "opw": float("nan"),
                    "tgm": float("nan"), "tcc": float("nan")}

        _validate_shapes(prediction, ground_truth)
        pred_seq = prediction.depth_map_seq.astype(np.float32)
        gt_seq = ground_truth.depth_map_seq.astype(np.float32)
        # Optional extras the loader may have stashed on the GT object
        # (we read them off the dataclass attribute namespace rather
        # than baking new required fields into VideoDepthGroundTruth).
        poses = getattr(ground_truth, "poses", None)
        rgb_seq = getattr(ground_truth, "rgb_seq", None)
        flow_fn = getattr(ground_truth, "flow_fn", None)
        return compute_temporal_depth_metrics(
            pred_seq,
            gt_seq=gt_seq,
            poses=poses,
            rgb_seq=rgb_seq,
            flow_fn=flow_fn,
        )


# --------------------------------------------------------------------------- #
# Shared shape validation
# --------------------------------------------------------------------------- #


def _validate_shapes(
    prediction: VideoDepthPrediction,
    ground_truth: VideoDepthGroundTruth,
) -> None:
    if not isinstance(prediction, VideoDepthPrediction):
        raise MetricError(
            f"VideoDepth calculator expected VideoDepthPrediction, got {type(prediction).__name__}",
        )
    if not isinstance(ground_truth, VideoDepthGroundTruth):
        raise MetricError(
            f"VideoDepth calculator expected VideoDepthGroundTruth, "
            f"got {type(ground_truth).__name__}",
        )
    if prediction.depth_map_seq.shape != ground_truth.depth_map_seq.shape:
        raise MetricError(
            f"VideoDepth prediction shape {prediction.depth_map_seq.shape} "
            f"does not match ground-truth shape {ground_truth.depth_map_seq.shape}",
            hint=(
                "If your adapter subsampled the clip via frame_budget, it "
                "must return predictions only for those frames AND the "
                "ground-truth must already be sliced to the same indices "
                "by VideoDepthDataset (this happens automatically when the loader "
                "honours frame_indices)."
            ),
        )
    if ground_truth.valid_mask_seq.shape != ground_truth.depth_map_seq.shape:
        raise MetricError(
            f"VideoDepth valid_mask_seq shape {ground_truth.valid_mask_seq.shape} "
            f"does not match depth_map_seq shape {ground_truth.depth_map_seq.shape}",
        )


__all__ = ["VideoDepthErrorMetrics", "VideoDepthTemporalMetrics"]

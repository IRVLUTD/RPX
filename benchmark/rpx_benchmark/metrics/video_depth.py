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

from typing import Dict, Optional

import numpy as np

from ..api import TaskType, VideoDepthGroundTruth, VideoDepthPrediction
from ..exceptions import MetricError
from ..logging_utils import get_logger
from .depth_alignment import DEPTH_MAX_M, DEPTH_MIN_M
from .registry import MetricCalculator, register_metric

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Per-frame error metrics, aggregated over the clip
# --------------------------------------------------------------------------- #


# D435 paper-declared intrinsics for the 640×480 release. Used by the
# per-frame F-Score@5cm computation. Frames must be at this resolution
# for F-Score to be well-defined; a shape mismatch raises.
# Kept in sync with the Image Depth pipeline's ``depth_paper.py``
# (Naren's D1-F module on branch jishnu/depth_pipeline_check).
_D435_FX = _D435_FY = 615.0
_D435_CX = 320.0
_D435_CY = 240.0
_D435_HW = (480, 640)

#: Grasp-tolerance F-Score threshold in metres (paper §5.2).
_FSCORE_THRESHOLD_M = 0.05


def _backproject(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Back-project (H, W) depth to (N, 3) camera-frame points at valid pixels."""
    ys, xs = np.nonzero(valid)
    z = depth[valid].astype(np.float64, copy=False)
    return np.column_stack(
        (
            (xs.astype(np.float64) - _D435_CX) * z / _D435_FX,
            (ys.astype(np.float64) - _D435_CY) * z / _D435_FY,
            z,
        )
    )


def _fscore_5cm_per_frame(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> Optional[float]:
    """Bidirectional cKDTree F-Score at 5 cm — one frame.

    Returns ``None`` when the frame has no valid pixels or scipy is
    unavailable at the frame level; callers should skip ``None`` before
    averaging. Matches the definition in
    ``rpx_benchmark.metrics.depth_paper.point_cloud_fscore_5cm``.
    """
    if not valid.any():
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover — scipy is a standard dep
        return None

    gt_pts = _backproject(gt, valid)
    pred_pts = _backproject(pred, valid)
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return None

    def _matched(tree, pts, chunk: int = 65_536) -> float:
        matched = 0
        for start in range(0, len(pts), chunk):
            d, _ = tree.query(pts[start : start + chunk], k=1, workers=1)
            matched += int(np.count_nonzero(d < _FSCORE_THRESHOLD_M))
        return matched / len(pts)

    precision = _matched(cKDTree(gt_pts), pred_pts)
    recall = _matched(cKDTree(pred_pts), gt_pts)
    denom = precision + recall
    return 0.0 if denom == 0.0 else float(2.0 * precision * recall / denom)


def _per_frame_error_metrics(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    valid_seq: np.ndarray,
    *,
    compute_fscore: bool = True,
) -> Dict[str, float]:
    """Per-frame paper-spec metrics → arithmetic mean over the clip.

    Emits the full K=5 paper spatial vector plus δ₂/δ₃ diagnostics:
    ``absrel``, ``rmse``, ``silog``, ``delta1``, ``fscore_5cm``,
    ``delta2``, ``delta3``.

    SILog follows the KITTI display convention (``100 × sqrt(Var(log
    error))``) — matches ``depth_paper.silog`` in the Image Depth pipeline,
    so cells.parquet values are comparable across Image Depth and Video Depth.

    F@5cm requires 640×480 frames (paper-declared D435 intrinsics).
    Set ``compute_fscore=False`` to defer the expensive cKDTree pass;
    the caller can back-fill from saved predictions. Frames whose
    valid mask is fully zero are skipped from every metric's mean
    rather than counted as zero.
    """
    T = pred_seq.shape[0]
    per_frame: Dict[str, list] = {
        "absrel": [],
        "rmse": [],
        "silog": [],
        "delta1": [],
        "delta2": [],
        "delta3": [],
    }
    fscore_vals: list = []
    fscore_enabled = compute_fscore and pred_seq.shape[1:] == _D435_HW

    for t in range(T):
        p = pred_seq[t]
        g = gt_seq[t]
        v = (
            valid_seq[t]
            & np.isfinite(g)
            & (g > DEPTH_MIN_M)
            & (g < DEPTH_MAX_M)
        )
        if not v.any():
            continue
        if not np.isfinite(p).all():
            raise MetricError(
                "Video Depth prediction contains non-finite values",
                hint=(
                    "Fix the adapter output. Finite raw predictions are preserved "
                    "in the cache and uniformly clipped to [0.3, 5.0] only in "
                    "the evaluation copy."
                ),
            )
        p_eval = np.clip(p, DEPTH_MIN_M, DEPTH_MAX_M)
        p_v = p_eval[v]
        g_v = g[v]
        diff = p_v - g_v
        log_err = np.log(p_v) - np.log(g_v)

        per_frame["absrel"].append(float(np.mean(np.abs(diff) / g_v)))
        per_frame["rmse"].append(float(np.sqrt(np.mean(diff ** 2))))
        silog_var = max(
            float(np.mean(log_err ** 2) - np.mean(log_err) ** 2), 0.0
        )
        per_frame["silog"].append(float(100.0 * np.sqrt(silog_var)))
        ratio = np.maximum(p_v / g_v, g_v / p_v)
        per_frame["delta1"].append(float(np.mean(ratio < 1.25)))
        per_frame["delta2"].append(float(np.mean(ratio < 1.25 ** 2)))
        per_frame["delta3"].append(float(np.mean(ratio < 1.25 ** 3)))

        if fscore_enabled:
            fs = _fscore_5cm_per_frame(p_eval, g, v)
            if fs is not None:
                fscore_vals.append(fs)

    if not per_frame["absrel"]:
        raise MetricError(
            "Video Depth clip contains no GT pixels in the paper-valid "
            "0.3 < depth < 5.0 m interval",
        )
    out: Dict[str, float] = {k: float(np.mean(v)) for k, v in per_frame.items()}
    out["fscore_5cm"] = (
        float(np.mean(fscore_vals)) if fscore_vals else float("nan")
    )
    return out


@register_metric(TaskType.VIDEO_DEPTH)
class VideoDepthErrorMetrics(MetricCalculator):
    """Per-clip aggregate of the paper-spec per-frame depth metrics.

    Emits the full K=5 spatial vector plus δ₂/δ₃ diagnostics:

    * ``absrel``, ``rmse``, ``silog`` — spatial error family
    * ``delta1`` — headline threshold accuracy
    * ``fscore_5cm`` — 3D grasp-tolerance accuracy (M5 in the paper's
      locked K=5). Requires 640×480 frames per D435's paper-declared
      intrinsics.
    * ``delta2``, ``delta3`` — cross-paper compatibility diagnostics
      (saturated on SOTA models; kept for Eigen-split tables)

    Each metric is computed per frame and arithmetically averaged over
    the clip's valid frames. Frames with fully-empty valid masks are
    skipped from the mean rather than counted as zero.

    SILog follows the KITTI display convention (``100 × sqrt(Var(log
    error))``), matching ``depth_paper.silog`` in the Image Depth pipeline
    so cells.parquet values are directly comparable across Image Depth (per-frame)
    and Video Depth (per-clip average of per-frame) runs.
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
            compute_fscore=bool(ground_truth.compute_fscore),
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
                    "tgm": float("nan"), "tgse": float("nan"),
                    "tcc": float("nan"), "tmc": float("nan")}

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

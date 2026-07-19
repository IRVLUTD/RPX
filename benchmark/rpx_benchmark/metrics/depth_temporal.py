"""Clip-level temporal depth metrics for Video Depth.

The per-frame depth calculators in :mod:`rpx_benchmark.metrics.depth`
operate on a single ``(prediction, ground_truth)`` pair, which is the
right contract for Image Depth (frame-level depth).  Video Depth additionally scores a
model on how *temporally consistent* its depth sequence is.  Those
metrics need a whole clip (``T`` frames) plus, for some of them, the
T265 camera poses — they do not fit the per-sample
:class:`~rpx_benchmark.metrics.registry.MetricCalculator` interface, so
they live here as standalone sequence functions.

Locked MANOVA temporal tuple for Video Depth (``benchmark/docs/depth_metric_decisions.md``):

* ``tae`` — Temporal Alignment Error (pose-based geometric consistency).
  In the K = 7 MANOVA vector.
* ``opw`` — Optical-Flow Warping error (appearance-based consistency).
  In the K = 7 MANOVA vector.  Needs an optical-flow backend (RAFT);
  the math here is backend-agnostic, the backend is injected.

Emitted for diagnostics but **excluded** from the MANOVA K:

* ``tgm`` — Temporal Gradient Matching vs GT depth.
* ``tcc`` — Temporal Consistency (SSIM on depth-change maps).

Directionality (see :mod:`rpx_benchmark.metrics.specs`): ``tae``, ``opw``,
``tgm`` are lower-is-better; ``tcc`` is higher-is-better.

All metrics are computed only over valid pixels.  For the GT-referenced
metrics (``tgm``, ``tcc``) validity is the standard D435 gate
(:func:`~rpx_benchmark.metrics.depth_alignment.default_valid_mask`).
For the GT-free metrics (``tae``, ``opw``) validity is "the prediction
is positive and finite, and the warp/reprojection landed on a valid
pixel" — these measure the prediction's *self*-consistency under known
camera motion (``tae``) or scene appearance flow (``opw``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..deployment import _D435_CX, _D435_CY, _D435_FX, _D435_FY, se3_reproject_depth
from ..exceptions import MetricError
from ..logging_utils import get_logger
from .depth_alignment import default_valid_mask

log = get_logger(__name__)

__all__ = [
    "FlowFn",
    "temporal_alignment_error",
    "optical_flow_warping_error",
    "temporal_gradient_matching",
    "temporal_gradient_squared_error",
    "temporal_motion_consistency",
    "temporal_consistency_coefficient",
    "compute_temporal_depth_metrics",
    "range_stratified_tae",
    "range_stratified_per_frame_metrics",
    "TEMPORAL_GRADIENT_STATIC_THRESH_M",
    "DEPTH_RANGE_BINS",
]

#: A flow backend maps two RGB frames (H, W, 3) to a dense flow field
#: (H, W, 2) in pixels, expressed as the *backward* flow from the second
#: frame to the first: a pixel ``(y, x)`` in ``rgb_to`` corresponds to
#: ``(y + flow[y, x, 1], x + flow[y, x, 0])`` in ``rgb_from``.  This is
#: the convention :func:`optical_flow_warping_error` warps with.  RAFT's
#: forward pass produces exactly this when called as ``raft(rgb_to,
#: rgb_from)``.
FlowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: TGM only compares depth-change in *static* regions (|ΔGT| below this),
#: per the appendix definition — dynamic pixels carry real scene motion,
#: not model jitter.  0.05 m matches Video Depth Anything (Chen et al. 2025).
TEMPORAL_GRADIENT_STATIC_THRESH_M: float = 0.05

#: A clip shorter than this is not scored — temporal metrics are not
#: meaningful on a handful of frames (decisions doc §3.8).
MIN_CLIP_FRAMES: int = 30


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _absrel(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> Optional[float]:
    """Mean ``|pred − ref| / ref`` over ``valid`` pixels, or ``None`` if empty."""
    if not valid.any():
        return None
    p = pred[valid].astype(np.float64)
    r = ref[valid].astype(np.float64)
    return float(np.mean(np.abs(p - r) / r))


def _bilinear_sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``img`` (H, W) at float coords ``ys``/``xs``.

    Coordinates outside the image are clamped to the border.  Returns an
    array shaped like ``ys``.
    """
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = xs - x0
    wy = ys - y0

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    img64 = img.astype(np.float64)
    va = img64[y0c, x0c]
    vb = img64[y0c, x1c]
    vc = img64[y1c, x0c]
    vd = img64[y1c, x1c]

    top = va * (1 - wx) + vb * wx
    bot = vc * (1 - wx) + vd * wx
    return top * (1 - wy) + bot * wy


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between two equal-shaped arrays.

    Uses scikit-image when available (the community-standard impl) and
    falls back to a global single-window SSIM otherwise, so the metric is
    always computable without a hard skimage dependency.
    """
    try:
        from skimage.metrics import structural_similarity  # type: ignore

        rng = float(max(a.max() - a.min(), b.max() - b.min(), 1e-6))
        return float(structural_similarity(a, b, data_range=rng))
    except Exception:  # pragma: no cover - exercised only without skimage
        a64 = a.astype(np.float64).ravel()
        b64 = b.astype(np.float64).ravel()
        mu_a, mu_b = a64.mean(), b64.mean()
        va, vb = a64.var(), b64.var()
        cov = float(np.mean((a64 - mu_a) * (b64 - mu_b)))
        c1 = (0.01 * 7.0) ** 2
        c2 = (0.03 * 7.0) ** 2
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
        return float(num / den) if den else 0.0


def _as_seq(name: str, arr: np.ndarray, ndim: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != ndim:
        raise MetricError(
            f"{name} must be a {ndim}-D sequence, got shape {a.shape}.",
            hint="Expected (T, H, W) for depth/mask sequences, (T, H, W, 3) for RGB.",
        )
    if a.shape[0] < 2:
        raise MetricError(f"{name} needs at least 2 frames, got T={a.shape[0]}.")
    return a


# --------------------------------------------------------------------------- #
# TAE — Temporal Alignment Error (pose-based, GT-free).  In MANOVA K.
# --------------------------------------------------------------------------- #


def temporal_alignment_error(
    pred_seq: np.ndarray,
    poses: np.ndarray,
    *,
    fx: float = _D435_FX,
    fy: float = _D435_FY,
    cx: float = _D435_CX,
    cy: float = _D435_CY,
) -> float:
    """Bidirectional SE(3)-reprojection AbsRel of a predicted depth clip.

    For every consecutive frame pair ``(t, t+1)`` the prediction at one
    frame is back-projected to 3D, transported to the other camera via
    the known world-from-camera poses, and re-projected.  The depth it
    *predicts there* is compared (AbsRel) against the model's *own*
    prediction at that frame.  A geometrically consistent model has
    ``tae → 0``; a model whose per-frame depth disagrees in 3D under
    known camera motion scores high.  No ground truth is used (this is a
    self-consistency measure), matching ChronoDepth / Video Depth
    Anything.

    Parameters
    ----------
    pred_seq : np.ndarray
        ``(T, H, W)`` predicted depth in metres.
    poses : np.ndarray
        ``(T, 4, 4)`` world-from-camera SE(3) per frame.
    fx, fy, cx, cy : float
        Pinhole intrinsics (default D435).

    Returns
    -------
    float
        Mean bidirectional AbsRel over all valid consecutive pairs.
        ``nan`` if no pair yields a single valid reprojected pixel.
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] != pred_seq.shape[0] or poses.shape[1:] != (4, 4):
        raise MetricError(
            f"poses must be (T, 4, 4) with T={pred_seq.shape[0]}, got {poses.shape}.",
        )

    errors = []
    for t in range(pred_seq.shape[0] - 1):
        d0, d1 = pred_seq[t], pred_seq[t + 1]
        p0, p1 = poses[t], poses[t + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue

        # Forward: frame t -> frame t+1.
        reproj01, ok01 = se3_reproject_depth(d0, p0, p1, fx, fy, cx, cy)
        v01 = ok01 & np.isfinite(reproj01) & (reproj01 > 0) & (d1 > 0) & np.isfinite(d1)
        e01 = _absrel(reproj01, d1, v01)
        if e01 is not None:
            errors.append(e01)

        # Backward: frame t+1 -> frame t.
        reproj10, ok10 = se3_reproject_depth(d1, p1, p0, fx, fy, cx, cy)
        v10 = ok10 & np.isfinite(reproj10) & (reproj10 > 0) & (d0 > 0) & np.isfinite(d0)
        e10 = _absrel(reproj10, d0, v10)
        if e10 is not None:
            errors.append(e10)

    return float(np.mean(errors)) if errors else float("nan")


# --------------------------------------------------------------------------- #
# OPW — Optical-Flow Warping error (appearance-based, GT-free).  In MANOVA K.
# --------------------------------------------------------------------------- #


def optical_flow_warping_error(
    pred_seq: np.ndarray,
    rgb_seq: np.ndarray,
    flow_fn: Optional[FlowFn],
) -> float:
    """AbsRel between flow-warped consecutive predicted depth maps.

    For each pair ``(t, t+1)`` the backward flow ``rgb[t+1] → rgb[t]`` is
    used to sample the depth prediction at ``t`` into the pixel grid of
    ``t+1``; the warped depth is compared (AbsRel) against the prediction
    at ``t+1``.  This measures appearance-level temporal smoothness and
    needs no GT or pose — only an optical-flow backend.

    Parameters
    ----------
    pred_seq : np.ndarray
        ``(T, H, W)`` predicted depth in metres.
    rgb_seq : np.ndarray
        ``(T, H, W, 3)`` RGB uint8 frames (the flow source).
    flow_fn : FlowFn | None
        Backend that returns the **backward** flow (see :data:`FlowFn`).
        ``None`` raises :class:`MetricError` — OPW cannot be computed
        without a flow model.  RPX pins RAFT (``raft-sintel.pth``); wire
        it as ``flow_fn=raft_backward_flow``.

    Returns
    -------
    float
        Mean AbsRel over all valid pairs; ``nan`` if none are valid.
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    rgb_seq = _as_seq("rgb_seq", rgb_seq, 4)
    if rgb_seq.shape[0] != pred_seq.shape[0]:
        raise MetricError(
            f"rgb_seq T={rgb_seq.shape[0]} must match pred_seq T={pred_seq.shape[0]}.",
        )
    if flow_fn is None:
        raise MetricError(
            "optical_flow_warping_error requires a flow backend; flow_fn is None.",
            hint=(
                "OPW needs dense optical flow. RPX pins RAFT (raft-sintel.pth, "
                "Teed & Deng 2020). Pass flow_fn=<raft backward-flow callable>. "
                "See benchmark/docs/depth_metric_decisions.md §2 (optical-flow "
                "recommendation)."
            ),
        )

    h, w = pred_seq.shape[1:]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float64)

    errors = []
    for t in range(pred_seq.shape[0] - 1):
        flow = np.asarray(flow_fn(rgb_seq[t + 1], rgb_seq[t]), dtype=np.float64)
        if flow.shape != (h, w, 2):
            raise MetricError(
                f"flow_fn returned shape {flow.shape}, expected ({h}, {w}, 2).",
            )
        src_x = grid_x + flow[..., 0]
        src_y = grid_y + flow[..., 1]
        warped = _bilinear_sample(pred_seq[t], src_y, src_x)

        d1 = pred_seq[t + 1]
        inside = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
        valid = inside & (warped > 0) & np.isfinite(warped) & (d1 > 0) & np.isfinite(d1)
        e = _absrel(warped, d1, valid)
        if e is not None:
            errors.append(e)

    return float(np.mean(errors)) if errors else float("nan")


# --------------------------------------------------------------------------- #
# TGM — Temporal Gradient Matching (GT-referenced).  Diagnostic, not in K.
# --------------------------------------------------------------------------- #


def temporal_gradient_matching(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    *,
    static_thresh_m: float = TEMPORAL_GRADIENT_STATIC_THRESH_M,
) -> float:
    """Mean L1 between predicted and GT per-pixel depth-change in static regions.

    .. math::

        \\text{TGM} = \\frac{1}{T-1}\\sum_t
            \\big\\| \\,|\\hat d_{t+1}-\\hat d_t| - |d_{t+1}-d_t|\\, \\big\\|_1

    restricted to pixels where ``|d_{t+1} − d_t| < static_thresh_m`` (the
    scene is locally static, so any predicted change is model jitter).
    Flow-free, GT-referenced.  Lower is better.

    Returns ``nan`` if no static valid pixels exist across the clip.
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    gt_seq = _as_seq("gt_seq", gt_seq, 3)
    if pred_seq.shape != gt_seq.shape:
        raise MetricError(
            f"pred_seq {pred_seq.shape} and gt_seq {gt_seq.shape} must match.",
        )

    errors = []
    for t in range(pred_seq.shape[0] - 1):
        v0 = default_valid_mask(pred_seq[t], gt_seq[t])
        v1 = default_valid_mask(pred_seq[t + 1], gt_seq[t + 1])
        valid = v0 & v1
        if not valid.any():
            continue
        d_gt = np.abs(gt_seq[t + 1] - gt_seq[t])
        d_pred = np.abs(pred_seq[t + 1] - pred_seq[t])
        static = valid & (d_gt < static_thresh_m)
        if not static.any():
            continue
        errors.append(float(np.mean(np.abs(d_pred[static] - d_gt[static]))))

    return float(np.mean(errors)) if errors else float("nan")


# --------------------------------------------------------------------------- #
# TCC — Temporal Consistency Coefficient (GT-referenced).  Diagnostic, not in K.
# --------------------------------------------------------------------------- #


def temporal_consistency_coefficient(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
) -> float:
    """Mean SSIM between predicted and GT temporal depth-change maps.

    Captures the *structural pattern* of depth change over time rather
    than its per-pixel magnitude (which TGM measures).  Higher is better,
    in ``[-1, 1]``.  Returns ``nan`` if no pair has valid pixels.
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    gt_seq = _as_seq("gt_seq", gt_seq, 3)
    if pred_seq.shape != gt_seq.shape:
        raise MetricError(
            f"pred_seq {pred_seq.shape} and gt_seq {gt_seq.shape} must match.",
        )

    scores = []
    for t in range(pred_seq.shape[0] - 1):
        v0 = default_valid_mask(pred_seq[t], gt_seq[t])
        v1 = default_valid_mask(pred_seq[t + 1], gt_seq[t + 1])
        valid = v0 & v1
        if not valid.any():
            continue
        diff_pred = (pred_seq[t + 1] - pred_seq[t]) * valid
        diff_gt = (gt_seq[t + 1] - gt_seq[t]) * valid
        scores.append(_ssim(diff_pred.astype(np.float64), diff_gt.astype(np.float64)))

    return float(np.mean(scores)) if scores else float("nan")


# --------------------------------------------------------------------------- #
# TGSE — Temporal Gradient Squared Error (GT-referenced).  Diagnostic.
# Origin: VDPP (Yoon et al., arXiv 2026, arXiv:2604.06665).
# --------------------------------------------------------------------------- #


def temporal_gradient_squared_error(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
) -> float:
    """Mean squared error of temporal depth gradients (pred vs GT).

    .. math::

        \\text{TGSE} = \\frac{1}{T-1}\\sum_t \\frac{1}{|V_t|}
            \\sum_{(i,j) \\in V_t}
            \\left( \\hat d^{(t+1)}_{i,j} - \\hat d^{(t)}_{i,j}
                   - (d^{(t+1)}_{i,j} - d^{(t)}_{i,j}) \\right)^2

    Unlike TGM: (a) preserves sign of depth change (no inner absolute
    value), (b) uses L2 so severe flicker dominates the average.
    Lower is better.  Returns ``nan`` if no pair has valid pixels.

    Reference: Yoon et al. "VDPP: Video Depth Post-Processing for Speed
    and Scalability" (arXiv 2026, 2604.06665).
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    gt_seq = _as_seq("gt_seq", gt_seq, 3)
    if pred_seq.shape != gt_seq.shape:
        raise MetricError(
            f"pred_seq {pred_seq.shape} and gt_seq {gt_seq.shape} must match.",
        )

    errors = []
    for t in range(pred_seq.shape[0] - 1):
        v0 = default_valid_mask(pred_seq[t], gt_seq[t])
        v1 = default_valid_mask(pred_seq[t + 1], gt_seq[t + 1])
        valid = v0 & v1
        if not valid.any():
            continue
        delta_pred = pred_seq[t + 1][valid] - pred_seq[t][valid]
        delta_gt = gt_seq[t + 1][valid] - gt_seq[t][valid]
        errors.append(float(np.mean((delta_pred.astype(np.float64)
                                     - delta_gt.astype(np.float64)) ** 2)))

    return float(np.mean(errors)) if errors else float("nan")


# --------------------------------------------------------------------------- #
# TMC — Temporal Motion Consistency (GT-referenced).  Diagnostic.
# Origin: Zhang et al. ICCV 2019 (arXiv:1908.03706).
# --------------------------------------------------------------------------- #


def _dense_flow_on_depth(d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """Compute dense flow between two depth maps using Farneback.

    Returns (H, W, 2) flow field.  Falls back to zero-flow if OpenCV is
    not available.
    """
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return np.zeros((*d0.shape, 2), dtype=np.float32)
    # Normalize depth to [0, 255] uint8 for Farneback.
    def _to_u8(d: np.ndarray) -> np.ndarray:
        d64 = d.astype(np.float64)
        lo, hi = np.nanmin(d64), np.nanmax(d64)
        if hi - lo < 1e-8:
            return np.zeros_like(d64, dtype=np.uint8)
        return ((d64 - lo) / (hi - lo) * 255).astype(np.uint8)

    return cv2.calcOpticalFlowFarneback(
        _to_u8(d0), _to_u8(d1),
        None,  # type: ignore[arg-type]
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def temporal_motion_consistency(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
) -> float:
    """Mean SSIM between optical flows computed on depth maps (pred vs GT).

    Computes dense optical flow (Farneback) on consecutive depth frames
    for both prediction and GT, then takes the SSIM between the
    resulting flow fields.  Captures whether the apparent motion
    *pattern* in depth space matches GT.

    Higher is better, in ``[-1, 1]``.  Returns ``nan`` if no pair is
    computable or if OpenCV is unavailable.

    Reference: Zhang et al. "Exploiting temporal consistency for
    real-time video depth estimation" (ICCV 2019, arXiv:1908.03706).
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    gt_seq = _as_seq("gt_seq", gt_seq, 3)
    if pred_seq.shape != gt_seq.shape:
        raise MetricError(
            f"pred_seq {pred_seq.shape} and gt_seq {gt_seq.shape} must match.",
        )

    scores = []
    for t in range(pred_seq.shape[0] - 1):
        flow_pred = _dense_flow_on_depth(pred_seq[t], pred_seq[t + 1])
        flow_gt = _dense_flow_on_depth(gt_seq[t], gt_seq[t + 1])
        # SSIM on flow magnitude
        mag_pred = np.sqrt(flow_pred[..., 0] ** 2 + flow_pred[..., 1] ** 2).astype(np.float64)
        mag_gt = np.sqrt(flow_gt[..., 0] ** 2 + flow_gt[..., 1] ** 2).astype(np.float64)
        scores.append(_ssim(mag_pred, mag_gt))

    return float(np.mean(scores)) if scores else float("nan")


# --------------------------------------------------------------------------- #
# Range-stratified TAE and per-frame metrics
# --------------------------------------------------------------------------- #

#: Depth range bins for robotics-aware stratification (metres).
#: Near = tabletop manipulation, Mid = arm's reach, Far = room-scale.
DEPTH_RANGE_BINS = {
    "near": (0.3, 1.0),
    "mid": (1.0, 2.5),
    "far": (2.5, 5.0),
}

#: Minimum valid-pixel fraction per frame for a bin to be scored.
_MIN_BIN_PIXELS = 100


def range_stratified_tae(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    poses: np.ndarray,
    *,
    fx: float = _D435_FX,
    fy: float = _D435_FY,
    cx: float = _D435_CX,
    cy: float = _D435_CY,
) -> Dict[str, float]:
    """TAE stratified by GT-depth range bins.

    For each consecutive frame pair and each range bin, computes the
    TAE only over pixels whose GT depth at the source frame falls in
    that bin.  Returns ``{tae_near, tae_mid, tae_far}``; bins with
    insufficient valid pixels are ``nan``.
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    gt_seq = _as_seq("gt_seq", gt_seq, 3)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] != pred_seq.shape[0] or poses.shape[1:] != (4, 4):
        raise MetricError(
            f"poses must be (T, 4, 4) with T={pred_seq.shape[0]}, got {poses.shape}.",
        )

    bin_errors: Dict[str, list] = {f"tae_{k}": [] for k in DEPTH_RANGE_BINS}

    for t in range(pred_seq.shape[0] - 1):
        d0, d1 = pred_seq[t], pred_seq[t + 1]
        g0 = gt_seq[t]
        p0, p1 = poses[t], poses[t + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue

        # Forward reproject: frame t -> frame t+1
        reproj01, ok01 = se3_reproject_depth(d0, p0, p1, fx, fy, cx, cy)
        base_valid = ok01 & np.isfinite(reproj01) & (reproj01 > 0) & (d1 > 0) & np.isfinite(d1)

        for bin_name, (lo, hi) in DEPTH_RANGE_BINS.items():
            bin_mask = base_valid & (g0 >= lo) & (g0 < hi)
            if bin_mask.sum() < _MIN_BIN_PIXELS:
                continue
            e = _absrel(reproj01, d1, bin_mask)
            if e is not None:
                bin_errors[f"tae_{bin_name}"].append(e)

    return {
        k: float(np.mean(v)) if v else float("nan")
        for k, v in bin_errors.items()
    }


def range_stratified_per_frame_metrics(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
) -> Dict[str, float]:
    """Per-frame AbsRel/RMSE/δ₁ stratified by GT depth range, clip-averaged.

    Returns keys like ``absrel_near``, ``rmse_mid``, ``delta1_far``.
    Bins with insufficient valid pixels across the clip are ``nan``.
    """
    pred_seq = _as_seq("pred_seq", pred_seq, 3)
    gt_seq = _as_seq("gt_seq", gt_seq, 3)
    if pred_seq.shape != gt_seq.shape:
        raise MetricError(
            f"pred_seq {pred_seq.shape} and gt_seq {gt_seq.shape} must match.",
        )

    metric_names = ("absrel", "rmse", "delta1")
    accum: Dict[str, list] = {}
    for m in metric_names:
        for b in DEPTH_RANGE_BINS:
            accum[f"{m}_{b}"] = []

    for t in range(pred_seq.shape[0]):
        p = pred_seq[t].astype(np.float64)
        g = gt_seq[t].astype(np.float64)
        base_valid = default_valid_mask(pred_seq[t], gt_seq[t])

        for bin_name, (lo, hi) in DEPTH_RANGE_BINS.items():
            v = base_valid & (g >= lo) & (g < hi)
            if v.sum() < _MIN_BIN_PIXELS:
                continue
            pv, gv = p[v], g[v]
            accum[f"absrel_{bin_name}"].append(float(np.mean(np.abs(pv - gv) / gv)))
            accum[f"rmse_{bin_name}"].append(float(np.sqrt(np.mean((pv - gv) ** 2))))
            ratio = np.maximum(pv / gv, gv / pv)
            accum[f"delta1_{bin_name}"].append(float(np.mean(ratio < 1.25)))

    return {
        k: float(np.mean(v)) if v else float("nan")
        for k, v in accum.items()
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def compute_temporal_depth_metrics(
    pred_seq: np.ndarray,
    *,
    gt_seq: Optional[np.ndarray] = None,
    poses: Optional[np.ndarray] = None,
    rgb_seq: Optional[np.ndarray] = None,
    flow_fn: Optional[FlowFn] = None,
    fx: float = _D435_FX,
    fy: float = _D435_FY,
    cx: float = _D435_CX,
    cy: float = _D435_CY,
) -> Dict[str, float]:
    """Compute every available temporal depth metric for one clip.

    Each metric is computed only when its required inputs are present;
    missing inputs degrade gracefully (the metric is set to ``nan`` and a
    note is logged) rather than raising — a clip with no T265 poses still
    yields ``opw``/``tgm``/``tcc``.

    Inputs
    ------
    pred_seq : (T, H, W) predicted depth, metres. Required.
    gt_seq   : (T, H, W) GT depth, metres. Enables ``tgm`` and ``tcc``.
    poses    : (T, 4, 4) world-from-camera SE(3). Enables ``tae``.
    rgb_seq  : (T, H, W, 3) RGB. With ``flow_fn`` enables ``opw``.
    flow_fn  : optical-flow backend (see :data:`FlowFn`).

    Returns
    -------
    dict[str, float]
        Keys ``tae, opw, tgm, tcc`` (``nan`` where not computable). The
        MANOVA selection (``tae``, ``opw``) happens downstream in
        ``analyze_experiment.py``; this calculator emits all four.
    """
    out: Dict[str, float] = {
        "tae": float("nan"), "opw": float("nan"),
        "tgm": float("nan"), "tcc": float("nan"),
        "tgse": float("nan"), "tmc": float("nan"),
    }
    # Range-stratified keys initialized to nan.
    for b in DEPTH_RANGE_BINS:
        out[f"tae_{b}"] = float("nan")
    for m in ("absrel", "rmse", "delta1"):
        for b in DEPTH_RANGE_BINS:
            out[f"{m}_{b}"] = float("nan")

    if poses is not None:
        out["tae"] = temporal_alignment_error(pred_seq, poses, fx=fx, fy=fy, cx=cx, cy=cy)
        if gt_seq is not None:
            strat = range_stratified_tae(pred_seq, gt_seq, poses, fx=fx, fy=fy, cx=cx, cy=cy)
            out.update(strat)
    else:
        log.debug("temporal: poses absent — tae/tae_stratified skipped (nan)")

    if rgb_seq is not None and flow_fn is not None:
        out["opw"] = optical_flow_warping_error(pred_seq, rgb_seq, flow_fn)
    else:
        log.debug("temporal: rgb_seq/flow_fn absent — opw skipped (nan)")

    if gt_seq is not None:
        out["tgm"] = temporal_gradient_matching(pred_seq, gt_seq)
        out["tcc"] = temporal_consistency_coefficient(pred_seq, gt_seq)
        out["tgse"] = temporal_gradient_squared_error(pred_seq, gt_seq)
        out["tmc"] = temporal_motion_consistency(pred_seq, gt_seq)
        strat_pf = range_stratified_per_frame_metrics(pred_seq, gt_seq)
        out.update(strat_pf)
    else:
        log.debug("temporal: gt_seq absent — tgm/tcc/tgse/tmc/stratified skipped (nan)")

    return out

"""Tests for the SILog + F@5cm additions to VideoDepthErrorMetrics.

Closes the K=7 spec gap: previously VideoDepthErrorMetrics emitted only
absrel / rmse / delta1 / delta2 / delta3. The paper's locked K=5
spatial set (M3=SILog, M5=F@5cm) needed both. This module validates
the new emissions against hand-computable expected values.
"""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.api import VideoDepthGroundTruth, VideoDepthPrediction
from rpx_benchmark.exceptions import MetricError
from rpx_benchmark.metrics.video_depth import (
    VideoDepthErrorMetrics,
    _fscore_5cm_per_frame,
    _per_frame_error_metrics,
)

# --------------------------------------------------------------------------- #
# Fixture: 640×480 clip so F@5cm is well-defined (D435 intrinsics)
# --------------------------------------------------------------------------- #


def _clip_at_paper_resolution(T: int = 3, *, offset_m: float = 0.0):
    """Constant-2.0m GT clip + prediction offset by ``offset_m``."""
    H, W = 480, 640
    gt = np.full((T, H, W), 2.0, dtype=np.float32)
    pred = np.full_like(gt, 2.0 + offset_m)
    valid = np.ones((T, H, W), dtype=bool)
    return pred, gt, valid


# --------------------------------------------------------------------------- #
# SILog — hand-computable on constant-offset clip
# --------------------------------------------------------------------------- #


def test_silog_zero_on_perfect_prediction():
    """Perfect prediction → log_error is uniformly 0 → Var(log_err) = 0 → SILog = 0.
    """
    pred, gt, valid = _clip_at_paper_resolution(T=2, offset_m=0.0)
    out = _per_frame_error_metrics(pred, gt, valid, compute_fscore=False)
    assert out["silog"] == pytest.approx(0.0, abs=1e-9)


def test_silog_constant_offset_still_zero_variance():
    """A CONSTANT prediction bias (every pixel same offset) → uniform
    log-error → Var(log_error) = 0. SILog captures VARIANCE not mean —
    a scale-invariance property. Must be ~0 even with a 20% bias.
    """
    pred, gt, valid = _clip_at_paper_resolution(T=2, offset_m=0.4)  # 20% off
    out = _per_frame_error_metrics(pred, gt, valid, compute_fscore=False)
    # AbsRel should be 0.2 (20% error), RMSE 0.4, but SILog ~0.
    # SILog tolerance is looser than pytest.approx defaults because a
    # constant fp32 prediction still carries ~10⁻⁹ log-space variance
    # from the fp32 round-off in `2.4`. Real SILog values are in the
    # 5–20 (KITTI display) range; 0.1 is comfortably below that.
    assert out["absrel"] == pytest.approx(0.2, rel=1e-4)
    assert out["rmse"] == pytest.approx(0.4, rel=1e-4)
    assert out["silog"] < 0.1


def test_silog_kitti_display_convention():
    """SILog reports the KITTI display form ``100 × sqrt(variance)``.

    Constructed clip: half the pixels have +10% error, half have -10%.
    log-errors: +ln(1.1) and -ln(0.9). Mean ≈ 0.00475; Var ≈ 0.00905;
    sqrt ≈ 0.0951; ×100 ≈ 9.51.
    """
    H, W = 480, 640
    gt = np.full((1, H, W), 2.0, dtype=np.float32)
    pred = np.full_like(gt, 2.2)  # +10%
    pred[:, :, W // 2 :] = 1.8    # -10% on right half
    valid = np.ones_like(gt, dtype=bool)
    out = _per_frame_error_metrics(pred, gt, valid, compute_fscore=False)

    # Expected: variance of {ln(1.1), ln(0.9)} × 100
    log_hi = np.log(1.1)
    log_lo = np.log(0.9)
    mean = (log_hi + log_lo) / 2
    var = ((log_hi - mean) ** 2 + (log_lo - mean) ** 2) / 2
    expected = 100.0 * np.sqrt(var)
    assert out["silog"] == pytest.approx(expected, rel=1e-3)


# --------------------------------------------------------------------------- #
# F@5cm — perfect prediction → 1.0; large offset → 0.0
# --------------------------------------------------------------------------- #


def test_fscore_perfect_prediction_is_one():
    """Same 3D cloud for pred and GT → every point matches within 5cm
    (in fact, 0 cm). Bidirectional F-score = 1.0.
    """
    pred, gt, valid = _clip_at_paper_resolution(T=1, offset_m=0.0)
    fs = _fscore_5cm_per_frame(pred[0], gt[0], valid[0])
    assert fs == pytest.approx(1.0, abs=1e-6)


def test_fscore_10cm_offset_is_zero():
    """10 cm z-axis offset → every point is 10 cm away from its NN in
    the other cloud → nobody matches at the 5 cm threshold → F = 0.
    """
    pred, gt, valid = _clip_at_paper_resolution(T=1, offset_m=0.10)
    fs = _fscore_5cm_per_frame(pred[0], gt[0], valid[0])
    assert fs == pytest.approx(0.0, abs=1e-6)


def test_fscore_disabled_when_shape_wrong():
    """Non-640×480 clips have no paper-declared intrinsics; F@5cm is
    skipped (NaN) rather than computed against ambiguous ones.
    """
    pred = np.full((1, 240, 320), 2.0, dtype=np.float32)  # half-res
    gt = pred.copy()
    valid = np.ones_like(pred, dtype=bool)
    out = _per_frame_error_metrics(pred, gt, valid)
    assert np.isnan(out["fscore_5cm"])
    # But the other metrics still work
    assert out["absrel"] == pytest.approx(0.0, abs=1e-6)


def test_fscore_defer_flag_skips_expensive_pass():
    """compute_fscore=False lets the caller skip the CPU-heavy pass and
    back-fill later from saved predictions (Naren's D1-F pattern).
    """
    pred, gt, valid = _clip_at_paper_resolution(T=2, offset_m=0.0)
    fast = _per_frame_error_metrics(pred, gt, valid, compute_fscore=False)
    assert np.isnan(fast["fscore_5cm"])
    # But the K=5 spatial vector is still fully populated
    for k in ("absrel", "rmse", "silog", "delta1"):
        assert k in fast and not np.isnan(fast[k])


# --------------------------------------------------------------------------- #
# End-to-end through VideoDepthErrorMetrics (the registered calculator)
# --------------------------------------------------------------------------- #


def test_calculator_emits_full_k5_plus_diagnostics():
    """The registered calculator's output dict must contain every key
    downstream Φ / J aggregation depends on.
    """
    pred, gt, valid = _clip_at_paper_resolution(T=3, offset_m=0.05)
    prediction = VideoDepthPrediction(depth_map_seq=pred)
    ground_truth = VideoDepthGroundTruth(
        depth_map_seq=gt,
        valid_mask_seq=valid,
        frame_indices=np.arange(3, dtype=np.int32),
        compute_fscore=True,
    )
    calc = VideoDepthErrorMetrics()
    out = calc.compute(prediction, ground_truth)

    # Full K=5 spatial (paper's locked set) + δ₂/δ₃ diagnostics
    for k in ("absrel", "rmse", "silog", "delta1", "delta2", "delta3", "fscore_5cm"):
        assert k in out, f"VideoDepthErrorMetrics missing key: {k}"


def test_calculator_empty_valid_fails_explicitly():
    """An unevaluable clip must not be recorded as a perfect result."""
    H, W = 480, 640
    pred = np.zeros((2, H, W), dtype=np.float32)
    gt = np.zeros_like(pred)
    valid = np.zeros_like(pred, dtype=bool)
    with pytest.raises(MetricError, match="no GT pixels"):
        _per_frame_error_metrics(pred, gt, valid)


def test_nonpositive_prediction_is_clipped_for_evaluation():
    pred, gt, valid = _clip_at_paper_resolution(T=1)
    pred[0, 0, 0] = 0.0
    out = _per_frame_error_metrics(pred, gt, valid, compute_fscore=False)
    expected = ((2.0 - 0.3) / 2.0) / (480 * 640)
    assert out["absrel"] == pytest.approx(expected)


def test_spatial_metrics_apply_strict_paper_gt_range():
    pred = np.asarray([[[4.0, 1.0, 1.0, 4.0]]], dtype=np.float32)
    gt = np.asarray([[[0.3, 1.0, 1.0, 5.0]]], dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    out = _per_frame_error_metrics(pred, gt, valid, compute_fscore=False)
    assert out["absrel"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)

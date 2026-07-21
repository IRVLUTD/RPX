"""Tests for the per-scene-phase pooled depth alignment.

Covers:

* The closed-form least-squares solver recovers a known affine
  transform within float-precision tolerance.
* The pooled solver produces the same result as solving on the
  flattened concatenation directly (sanity).
* ``mode="none"`` and ``mode="median"`` short-circuit correctly.
* Degenerate inputs (empty valid mask, all-zero valid mask) return
  the input unchanged rather than raising.
* The ``BenchmarkModel.depth_output_kind`` attribute defaults to
  ``"metric"`` and accepts ``"relative"`` overrides.

The pooled solver is the substrate for both D1-F (per-scene-phase
across ~250 frames) and D1-V (per-clip across all frames). The
existing per-clip D1-V wrapper now delegates to this same code path
— that delegation is also tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.api import BenchmarkModel, TaskType
from rpx_benchmark.metrics.depth_alignment import (
    align_pred_to_gt,
    align_pred_to_gt_pooled,
)

# --------------------------------------------------------------------------- #
# align_pred_to_gt_pooled
# --------------------------------------------------------------------------- #


def test_pooled_ls_affine_recovers_known_transform():
    """Plant a pred with known (s, t) relative to GT, solve, recover."""
    rng = np.random.default_rng(42)
    gt = rng.uniform(0.5, 4.0, size=(8, 16, 16)).astype(np.float32)
    s_true, t_true = 2.5, -0.4
    pred = ((gt - t_true) / s_true).astype(np.float32)
    valid = np.ones_like(gt, dtype=bool)

    aligned = align_pred_to_gt_pooled(pred, gt, mode="ls_affine", valid_seq=valid)
    err = float(np.mean(np.abs(aligned - gt)))
    assert err < 1e-4, f"alignment error {err} too large"


def test_pooled_ls_log_recovers_known_log_depth_transform():
    """FE2E's raw signal is affine to log depth, pooled by RPX cell."""
    rng = np.random.default_rng(7)
    gt = rng.uniform(0.5, 4.0, size=(5, 9, 11)).astype(np.float32)
    scale, shift = 1.7, -0.25
    pred = ((np.log(gt) - shift) / scale).astype(np.float32)
    valid = np.ones_like(gt, dtype=bool)

    aligned = align_pred_to_gt_pooled(pred, gt, mode="ls_log", valid_seq=valid)
    np.testing.assert_allclose(aligned, gt, rtol=2e-5, atol=2e-5)


def test_per_frame_ls_log_recovers_known_log_depth_transform():
    gt = np.linspace(0.4, 4.5, 80, dtype=np.float32).reshape(8, 10)
    pred = ((np.log(gt) + 0.6) / 2.2).astype(np.float32)
    valid = np.ones_like(gt, dtype=bool)

    aligned = align_pred_to_gt(pred, gt, mode="ls_log", valid=valid)
    np.testing.assert_allclose(aligned, gt, rtol=2e-5, atol=2e-5)


def test_pooled_alignment_matches_solving_on_flattened_concatenation():
    """Pooled solve over (T, H, W) must equal solving over flat 1-D arrays."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 4.0, size=(4, 8, 8)).astype(np.float32)
    pred = (0.3 * gt + 1.5).astype(np.float32)
    valid = np.ones_like(gt, dtype=bool)

    aligned_pooled = align_pred_to_gt_pooled(pred, gt, mode="ls_affine", valid_seq=valid)

    # Manually flatten and solve via the per-frame variant — they should
    # converge to the same coefficients because pooling all frames is
    # equivalent to one big frame.
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_gt = gt.reshape(-1, gt.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    aligned_manual = align_pred_to_gt(flat_pred, flat_gt, mode="ls_affine", valid=flat_valid)

    np.testing.assert_allclose(
        aligned_pooled.reshape(-1, aligned_pooled.shape[-1]),
        aligned_manual,
        atol=1e-5,
    )


def test_pooled_mode_none_is_identity():
    pred = np.full((2, 4, 4), 2.5, dtype=np.float32)
    gt = np.full_like(pred, 1.0)
    valid = np.ones_like(pred, dtype=bool)
    out = align_pred_to_gt_pooled(pred, gt, mode="none", valid_seq=valid)
    assert out is pred  # identity short-circuit returns the input


def test_pooled_median_mode_one_global_scale():
    """Median mode rescales by the ratio of pooled medians; degenerate
    case to verify pooled vs per-frame difference shows up correctly."""
    pred = np.full((3, 4, 4), 2.0, dtype=np.float32)
    gt = np.full_like(pred, 6.0)
    valid = np.ones_like(pred, dtype=bool)
    out = align_pred_to_gt_pooled(pred, gt, mode="median", valid_seq=valid)
    # median(gt)/median(pred) = 6/2 = 3 → aligned should be ~6.0
    np.testing.assert_allclose(out, 6.0, atol=1e-5)


def test_pooled_returns_input_when_valid_mask_empty():
    """All-zero valid mask must NOT crash the solver."""
    pred = np.full((2, 4, 4), 1.0, dtype=np.float32)
    gt = np.full_like(pred, 2.0)
    valid = np.zeros_like(pred, dtype=bool)
    out = align_pred_to_gt_pooled(pred, gt, mode="ls_affine", valid_seq=valid)
    # No valid pixels → cannot solve → return input unchanged
    np.testing.assert_array_equal(out, pred)


def test_pooled_rejects_shape_mismatch():
    from rpx_benchmark.exceptions import ConfigError

    pred = np.zeros((4, 8, 8), dtype=np.float32)
    gt = np.zeros((4, 8, 9), dtype=np.float32)  # wrong width
    with pytest.raises(ConfigError, match="shape"):
        align_pred_to_gt_pooled(pred, gt, mode="ls_affine")


def test_pooled_uses_default_valid_mask_when_none_provided():
    """Caller can omit ``valid_seq``; default_valid_mask is applied."""
    # GT zeros are excluded by default mask, so pooled solve sees only
    # the non-zero pixels.
    gt = np.zeros((2, 4, 4), dtype=np.float32)
    gt[0, 0, :] = 1.5  # inject one row of valid GT in metres-range
    pred = (0.5 * gt + 0.2).astype(np.float32)
    out = align_pred_to_gt_pooled(pred, gt, mode="ls_affine", valid_seq=None)
    # Only the valid row contributed; the aligned values for that row
    # should be close to gt.
    assert np.mean(np.abs(out[0, 0, :] - gt[0, 0, :])) < 1e-3


# --------------------------------------------------------------------------- #
# BenchmarkModel.depth_output_kind contract
# --------------------------------------------------------------------------- #


def test_benchmark_model_depth_output_kind_defaults_to_metric():
    """The default keeps every existing adapter behaving like a metric
    model — no behaviour change for the 7 metric models in the roster."""

    class _MetricModel(BenchmarkModel):
        task = TaskType.MONOCULAR_DEPTH

        def setup(self) -> None:
            pass

        def predict(self, batch):
            return []

    m = _MetricModel()
    assert m.depth_output_kind == "metric"


def test_benchmark_model_depth_output_kind_can_be_set_relative():
    """Affine-invariant model adapters opt in by setting the attribute."""

    class _RelativeModel(BenchmarkModel):
        task = TaskType.MONOCULAR_DEPTH
        depth_output_kind = "relative"

        def setup(self) -> None:
            pass

        def predict(self, batch):
            return []

    m = _RelativeModel()
    assert m.depth_output_kind == "relative"


# --------------------------------------------------------------------------- #
# video_loader.align_scale_and_shift_per_clip still works (delegation)
# --------------------------------------------------------------------------- #


def test_video_loader_per_clip_wrapper_still_works():
    """The D1-V per-clip wrapper now delegates to the pooled solver —
    behaviour from the existing video_loader contract must be preserved."""
    from rpx_benchmark.video_loader import align_scale_and_shift_per_clip

    rng = np.random.default_rng(7)
    gt = rng.uniform(0.5, 4.0, size=(4, 16, 16)).astype(np.float32)
    s_true, t_true = 1.7, 0.3
    pred = ((gt - t_true) / s_true).astype(np.float32)
    valid = np.ones_like(gt, dtype=bool)

    aligned = align_scale_and_shift_per_clip(pred, gt, valid)
    err = float(np.mean(np.abs(aligned - gt)))
    assert err < 1e-4, f"per-clip alignment error {err} too large"


def test_video_loader_per_clip_wrapper_preserves_dtype():
    """Per-clip wrapper preserves the input dtype, even though the
    solver internals are float64."""
    from rpx_benchmark.video_loader import align_scale_and_shift_per_clip

    pred = np.full((2, 4, 4), 1.0, dtype=np.float32)
    gt = np.full_like(pred, 2.0)
    valid = np.ones_like(pred, dtype=bool)
    out = align_scale_and_shift_per_clip(pred, gt, valid)
    assert out.dtype == np.float32

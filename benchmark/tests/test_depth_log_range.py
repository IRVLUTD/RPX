"""Tests for RMSElog, SILog, and range-stratified depth metrics."""

from __future__ import annotations

import math

import numpy as np

from rpx_benchmark.api import DepthGroundTruth, DepthPrediction
from rpx_benchmark.metrics.depth import DepthLogMetrics, DepthRangeStratified


def _make_pair(pred_vals, gt_vals):
    pred = np.array(pred_vals, dtype=np.float32).reshape(-1, 1)
    gt = np.array(gt_vals, dtype=np.float32).reshape(-1, 1)
    return DepthPrediction(depth_map=pred), DepthGroundTruth(depth_map=gt)


# ── DepthLogMetrics ──────────────────────────────────────────────────

class TestDepthLogMetrics:
    calc = DepthLogMetrics()

    def test_perfect_prediction(self):
        pred, gt = _make_pair([2.0] * 20, [2.0] * 20)
        m = self.calc.compute(pred, gt)
        assert m["rmselog"] == 0.0
        assert m["silog"] == 0.0

    def test_constant_scale_silog_zero(self):
        """pred = 2×gt everywhere → log_diff is constant → Var = 0 → SILog = 0."""
        gt_vals = [0.5, 1.0, 2.0, 3.0, 4.0] * 4  # all in valid range
        pred_vals = [v * 2.0 for v in gt_vals]
        pred, gt = _make_pair(pred_vals, gt_vals)
        m = self.calc.compute(pred, gt)
        assert math.isclose(m["silog"], 0.0, abs_tol=1e-10), \
            f"Constant scale should give SILog=0, got {m['silog']}"
        # rmselog = |log(2)| ≈ 0.693
        assert math.isclose(m["rmselog"], math.log(2), rel_tol=1e-4)

    def test_all_invalid_sentinel(self):
        pred, gt = _make_pair([1.0], [0.0])
        m = self.calc.compute(pred, gt)
        assert m["rmselog"] == 0.0
        assert m["silog"] == 0.0

    def test_out_of_range_excluded(self):
        """GT outside (0.3, 5.0)m must be excluded."""
        pred, gt = _make_pair([0.1, 6.0], [0.1, 6.0])
        m = self.calc.compute(pred, gt)
        assert m["rmselog"] == 0.0  # sentinel — no valid pixels


# ── DepthRangeStratified ─────────────────────────────────────────────

class TestDepthRangeStratified:
    calc = DepthRangeStratified()

    def test_all_near_range(self):
        """All pixels in near range (0.3–1.0m) → near metrics valid, mid/far NaN."""
        n = 200  # above _MIN_PIXELS_PER_BIN = 100
        gt_vals = [0.5] * n
        pred_vals = [0.5] * n
        pred, gt = _make_pair(pred_vals, gt_vals)
        m = self.calc.compute(pred, gt)
        assert m["absrel_near"] == 0.0
        assert m["delta1_near"] == 1.0
        assert math.isnan(m["absrel_mid"])
        assert math.isnan(m["absrel_far"])

    def test_mixed_ranges(self):
        """Pixels spanning near, mid, far should produce non-NaN for each."""
        n = 150
        gt_vals = [0.5] * n + [1.5] * n + [3.0] * n
        pred_vals = gt_vals[:]  # perfect
        pred, gt = _make_pair(pred_vals, gt_vals)
        m = self.calc.compute(pred, gt)
        for suffix in ("near", "mid", "far"):
            assert m[f"absrel_{suffix}"] == 0.0, f"absrel_{suffix} should be 0"
            assert m[f"delta1_{suffix}"] == 1.0, f"delta1_{suffix} should be 1"

    def test_too_few_pixels_gives_nan(self):
        """Fewer than _MIN_PIXELS_PER_BIN in a bin → NaN."""
        gt_vals = [0.5] * 50  # below 100 threshold
        pred_vals = [0.5] * 50
        pred, gt = _make_pair(pred_vals, gt_vals)
        m = self.calc.compute(pred, gt)
        assert math.isnan(m["absrel_near"]), "Too few pixels should give NaN"

    def test_keys_emitted(self):
        """Check all expected keys are present."""
        n = 200
        gt_vals = [2.0] * n
        pred_vals = [2.0] * n
        pred, gt = _make_pair(pred_vals, gt_vals)
        m = self.calc.compute(pred, gt)
        for metric in ("absrel", "rmse", "rmselog", "silog", "delta1"):
            for bin_name in ("near", "mid", "far"):
                assert f"{metric}_{bin_name}" in m, f"Missing key {metric}_{bin_name}"

"""Unit tests for depth metric computation."""

from __future__ import annotations

import math

import numpy as np

from rpx_benchmark.evaluators import depth_metrics


def test_perfect_prediction_yields_zero_error():
    gt = np.ones((10, 10), dtype=np.float32) * 2.0
    pred = gt.copy()
    m = depth_metrics(pred, gt)
    assert m["absrel"] == 0.0
    assert m["rmse"] == 0.0
    assert m["delta1"] == 1.0
    assert m["delta2"] == 1.0
    assert m["delta3"] == 1.0


def test_scaled_prediction_yields_expected_absrel():
    """pred = 0.9 * gt → AbsRel = 0.1 exactly, δ1=1.0 (0.9 < 1.25)."""
    gt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    pred = 0.9 * gt
    m = depth_metrics(pred, gt)
    assert math.isclose(m["absrel"], 0.1, rel_tol=1e-6)
    assert m["delta1"] == 1.0


def test_invalid_gt_pixels_are_masked():
    """Pixels with gt == 0 must be excluded from metrics."""
    gt = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    pred = np.array([[1.0, 999.0], [999.0, 2.0]], dtype=np.float32)
    m = depth_metrics(pred, gt)
    assert m["absrel"] == 0.0
    assert m["rmse"] == 0.0


def test_all_invalid_returns_sentinel():
    gt = np.zeros((5, 5), dtype=np.float32)
    pred = np.ones_like(gt)
    m = depth_metrics(pred, gt)
    # Function returns δ=1.0 and error=0 for a totally-invalid frame
    # as a safe no-op aggregation value.
    assert m["delta1"] == 1.0
    assert m["absrel"] == 0.0

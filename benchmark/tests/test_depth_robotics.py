"""Tests for robotics-deployment depth metrics (Chamfer, F-score, normals, boundary)."""

from __future__ import annotations

import math

import numpy as np

from rpx_benchmark.api import DepthGroundTruth, DepthPrediction
from rpx_benchmark.metrics.depth_robotics import (
    DepthBoundaryF1,
    DepthGrasp3D,
    DepthNormals,
)


def _make_pair(pred_arr, gt_arr):
    return (
        DepthPrediction(depth_map=np.asarray(pred_arr, dtype=np.float32)),
        DepthGroundTruth(depth_map=np.asarray(gt_arr, dtype=np.float32)),
    )


# ── DepthGrasp3D (Chamfer-L1 + F-score) ──────────────────────────────

class TestDepthGrasp3D:
    calc = DepthGrasp3D()

    def test_perfect_depth(self):
        """Identical pred/gt → Chamfer=0, F-score=1 at all thresholds."""
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        pred, gt = _make_pair(depth, depth)
        m = self.calc.compute(pred, gt)
        assert m["chamfer_l1"] == 0.0
        assert m["fscore_1cm"] == 1.0
        assert m["fscore_5cm"] == 1.0
        assert m["fscore_10cm"] == 1.0

    def test_small_error_fscore_degrades(self):
        """10cm uniform error at 2m depth → F@1cm should be low, F@10cm higher."""
        gt = np.full((480, 640), 2.0, dtype=np.float32)
        pred = np.full((480, 640), 2.1, dtype=np.float32)  # 10cm off
        p, g = _make_pair(pred, gt)
        m = self.calc.compute(p, g)
        assert m["chamfer_l1"] > 0
        assert m["fscore_1cm"] < m["fscore_10cm"]

    def test_too_few_valid_pixels(self):
        """Mostly invalid → NaN chamfer, 0 F-score."""
        gt = np.zeros((10, 10), dtype=np.float32)
        pred = np.ones((10, 10), dtype=np.float32)
        p, g = _make_pair(pred, gt)
        m = self.calc.compute(p, g)
        assert math.isnan(m["chamfer_l1"])
        assert m["fscore_5cm"] == 0.0


# ── DepthNormals ─────────────────────────────────────────────────────

class TestDepthNormals:
    calc = DepthNormals()

    def test_flat_wall_normals(self):
        """Flat wall at z=2m → normals should be [0,0,-1], MAE ≈ 0."""
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        pred, gt = _make_pair(depth, depth)
        m = self.calc.compute(pred, gt)
        # Perfect pred = perfect gt normals → angular error ≈ 0
        assert m["normal_mae"] < 1.0, f"Flat wall should have near-zero MAE, got {m['normal_mae']}"
        assert m["normal_acc1125"] > 0.9

    def test_all_invalid_sentinel(self):
        gt = np.zeros((10, 10), dtype=np.float32)
        pred = np.ones((10, 10), dtype=np.float32)
        p, g = _make_pair(pred, gt)
        m = self.calc.compute(p, g)
        assert math.isnan(m["normal_mae"])


# ── DepthBoundaryF1 ──────────────────────────────────────────────────

class TestDepthBoundaryF1:
    calc = DepthBoundaryF1()

    def test_identical_depth_perfect_f1(self):
        """Identical pred/gt → same contours → F1 = 1.0."""
        depth = np.full((50, 50), 2.0, dtype=np.float32)
        # Add a depth step to create a contour
        depth[25:, :] = 4.0
        pred, gt = _make_pair(depth, depth)
        m = self.calc.compute(pred, gt)
        assert m["boundary_f1"] == 1.0

    def test_no_contours_perfect(self):
        """Flat depth everywhere → no contours in either → F1 = 1.0."""
        depth = np.full((50, 50), 2.0, dtype=np.float32)
        pred, gt = _make_pair(depth, depth)
        m = self.calc.compute(pred, gt)
        assert m["boundary_f1"] == 1.0

    def test_missing_boundary_low_f1(self):
        """GT has a boundary, pred doesn't → F1 should be low."""
        gt_d = np.full((50, 50), 2.0, dtype=np.float32)
        gt_d[25:, :] = 4.0  # 2× depth step → strong contour
        pred_d = np.full((50, 50), 3.0, dtype=np.float32)  # smooth, no step
        pred, gt = _make_pair(pred_d, gt_d)
        m = self.calc.compute(pred, gt)
        assert m["boundary_f1"] < 0.5, f"Missing boundary should give low F1, got {m['boundary_f1']}"

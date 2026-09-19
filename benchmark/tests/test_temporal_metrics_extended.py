"""Tests for the extended temporal depth metrics: TGSE, TMC, range-stratified.

Golden values are computed by hand on small synthetic clips so the tests
are fully deterministic and offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.metrics.depth_temporal import (
    DEPTH_RANGE_BINS,
    compute_temporal_depth_metrics,
    range_stratified_per_frame_metrics,
    range_stratified_tae,
    range_stratified_tgm_tgse,
    temporal_gradient_squared_error,
    temporal_motion_consistency,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _constant_clip(T: int, H: int, W: int, value: float) -> np.ndarray:
    """Depth clip where every pixel = value."""
    return np.full((T, H, W), value, dtype=np.float32)


def _identity_poses(T: int) -> np.ndarray:
    """T × (4, 4) identity poses."""
    return np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))


def _translated_poses(T: int, step: float = 0.01) -> np.ndarray:
    """Poses with small x-translation per frame."""
    poses = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    for t in range(T):
        poses[t, 0, 3] = t * step
    return poses


# --------------------------------------------------------------------------- #
# TGSE tests
# --------------------------------------------------------------------------- #


class TestTGSE:
    def test_perfect_match_is_zero(self):
        """Identical pred and GT sequences → TGSE = 0."""
        T, H, W = 5, 10, 10
        seq = np.random.default_rng(42).uniform(0.5, 3.0, (T, H, W)).astype(np.float32)
        assert temporal_gradient_squared_error(seq, seq) == pytest.approx(0.0, abs=1e-10)

    def test_constant_pred_varying_gt(self):
        """Constant prediction, GT changes by 0.1 per frame → TGSE = 0.01."""
        T, H, W = 5, 10, 10
        pred = _constant_clip(T, H, W, 1.0)
        gt = np.stack([np.full((H, W), 1.0 + t * 0.1) for t in range(T)]).astype(np.float32)
        # Δpred = 0, Δgt = 0.1 → per-pixel: (0 - 0.1)^2 = 0.01
        val = temporal_gradient_squared_error(pred, gt)
        assert val == pytest.approx(0.01, abs=1e-6)

    def test_preserves_sign_unlike_tgm(self):
        """TGSE is sensitive to direction: pred increases while GT decreases."""
        T, H, W = 3, 10, 10
        # pred goes up by 0.1, GT goes down by 0.1
        pred = np.stack([np.full((H, W), 1.0 + t * 0.1) for t in range(T)]).astype(np.float32)
        gt = np.stack([np.full((H, W), 1.0 - t * 0.1) for t in range(T)]).astype(np.float32)
        # Δpred = +0.1, Δgt = -0.1 → (0.1 - (-0.1))^2 = 0.04
        val = temporal_gradient_squared_error(pred, gt)
        assert val == pytest.approx(0.04, abs=1e-6)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(Exception):
            temporal_gradient_squared_error(np.zeros((3, 10, 10)), np.zeros((3, 10, 20)))


# --------------------------------------------------------------------------- #
# TMC tests
# --------------------------------------------------------------------------- #


class TestTMC:
    def test_perfect_match_is_one(self):
        """Identical sequences → TMC ≈ 1.0."""
        T, H, W = 4, 20, 20
        rng = np.random.default_rng(42)
        seq = rng.uniform(0.5, 4.0, (T, H, W)).astype(np.float32)
        val = temporal_motion_consistency(seq, seq)
        assert val == pytest.approx(1.0, abs=0.05)

    def test_constant_clip_returns_finite(self):
        """Constant depth clip should return finite TMC (flow = 0)."""
        T, H, W = 4, 20, 20
        seq = _constant_clip(T, H, W, 2.0)
        val = temporal_motion_consistency(seq, seq)
        assert np.isfinite(val)


# --------------------------------------------------------------------------- #
# Range-stratified per-frame metrics
# --------------------------------------------------------------------------- #


class TestRangeStratified:
    def test_all_near(self):
        """All GT pixels at 0.5m → only _near keys populated."""
        T, H, W = 3, 20, 20
        pred = _constant_clip(T, H, W, 0.6)
        gt = _constant_clip(T, H, W, 0.5)
        result = range_stratified_per_frame_metrics(pred, gt)
        assert np.isfinite(result["absrel_near"])
        assert result["absrel_near"] == pytest.approx(0.2, abs=1e-4)  # |0.6-0.5|/0.5
        assert np.isnan(result["absrel_mid"])
        assert np.isnan(result["absrel_far"])

    def test_mixed_ranges(self):
        """Pixels at different depths populate different bins."""
        T, H, W = 3, 200, 1  # 200 pixels per frame
        gt = np.zeros((T, H, W), dtype=np.float32)
        pred = np.zeros((T, H, W), dtype=np.float32)
        # Near: first 100 pixels at 0.5m
        gt[:, :100, :] = 0.5
        pred[:, :100, :] = 0.5
        # Far: next 100 pixels at 3.0m
        gt[:, 100:, :] = 3.0
        pred[:, 100:, :] = 3.3
        result = range_stratified_per_frame_metrics(pred, gt)
        assert np.isfinite(result["absrel_near"])
        assert result["absrel_near"] == pytest.approx(0.0, abs=1e-4)
        assert np.isfinite(result["absrel_far"])
        assert result["absrel_far"] == pytest.approx(0.1, abs=1e-4)  # |3.3-3.0|/3.0


# --------------------------------------------------------------------------- #
# Range-stratified TAE
# --------------------------------------------------------------------------- #


class TestRangeStratifiedTAE:
    def test_identity_pose_consistent(self):
        """With identity poses, TAE_* should be ~0 for consistent depth."""
        T, H, W = 5, 20, 20
        pred = _constant_clip(T, H, W, 1.0)
        gt = _constant_clip(T, H, W, 1.0)  # all near→mid boundary
        poses = _identity_poses(T)
        result = range_stratified_tae(pred, gt, poses)
        # 1.0m is exactly at the near/mid boundary → falls in mid bin [1.0, 2.5)
        assert "tae_mid" in result
        # With identity poses and constant depth, reprojected depth = pred → TAE = 0.
        for k, v in result.items():
            if np.isfinite(v):
                assert v == pytest.approx(0.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Range-stratified TGM and TGSE (pose-free / flow-free)
# --------------------------------------------------------------------------- #


class TestRangeStratifiedTGMTGSE:
    def test_perfect_match_zero_across_bins(self):
        """Identical pred and GT → tgm and tgse are 0 in every populated bin."""
        T, H, W = 4, 40, 40
        # Static clip at 1.5m (mid bin), all identical → both metrics 0.
        d = _constant_clip(T, H, W, 1.5)
        result = range_stratified_tgm_tgse(d, d)
        assert result["tgm_mid"] == pytest.approx(0.0, abs=1e-6)
        assert result["tgse_mid"] == pytest.approx(0.0, abs=1e-6)
        # Empty bins are nan.
        assert np.isnan(result["tgm_near"])
        assert np.isnan(result["tgse_far"])

    def test_bin_selection_by_source_depth(self):
        """Pred error only at far range → only *_far bins show finite non-zero."""
        T, H, W = 3, 200, 1
        gt = np.zeros((T, H, W), dtype=np.float32)
        pred = np.zeros((T, H, W), dtype=np.float32)
        # Near pixels: constant at 0.5m, pred matches exactly.
        gt[:, :100, :] = 0.5
        pred[:, :100, :] = 0.5
        # Far pixels: constant GT at 3.0m; pred flickers between 3.0 and 3.2.
        gt[:, 100:, :] = 3.0
        pred[0, 100:, :] = 3.0
        pred[1, 100:, :] = 3.2
        pred[2, 100:, :] = 3.0

        result = range_stratified_tgm_tgse(pred, gt)
        # Near bin: no change in either → tgm_near and tgse_near = 0.
        assert result["tgm_near"] == pytest.approx(0.0, abs=1e-6)
        assert result["tgse_near"] == pytest.approx(0.0, abs=1e-6)
        # Far bin: predicted flicker at bin-source, GT static → tgm_far > 0, tgse_far > 0.
        assert np.isfinite(result["tgm_far"]) and result["tgm_far"] > 0.0
        assert np.isfinite(result["tgse_far"]) and result["tgse_far"] > 0.0
        # Mid bin empty (no pixels there) → nan.
        assert np.isnan(result["tgm_mid"])
        assert np.isnan(result["tgse_mid"])

    def test_all_six_keys_present(self):
        T, H, W = 3, 20, 20
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.5, 3.0, (T, H, W)).astype(np.float32)
        gt = rng.uniform(0.5, 3.0, (T, H, W)).astype(np.float32)
        result = range_stratified_tgm_tgse(pred, gt)
        assert set(result) == {
            "tgm_near", "tgm_mid", "tgm_far",
            "tgse_near", "tgse_mid", "tgse_far",
        }

    def test_shape_mismatch_raises(self):
        pred = np.zeros((3, 10, 10), dtype=np.float32)
        gt = np.zeros((3, 8, 8), dtype=np.float32)
        with pytest.raises(Exception, match="must match"):
            range_stratified_tgm_tgse(pred, gt)


# --------------------------------------------------------------------------- #
# compute_temporal_depth_metrics now includes new keys
# --------------------------------------------------------------------------- #


class TestOrchestratorExtended:
    def test_all_new_keys_present(self):
        """Orchestrator emits tgse, tmc, and range-stratified keys."""
        T, H, W = 5, 20, 20
        rng = np.random.default_rng(42)
        pred = rng.uniform(0.5, 3.0, (T, H, W)).astype(np.float32)
        gt = rng.uniform(0.5, 3.0, (T, H, W)).astype(np.float32)
        poses = _identity_poses(T)

        result = compute_temporal_depth_metrics(
            pred, gt_seq=gt, poses=poses,
        )
        # Check new keys are present.
        assert "tgse" in result
        assert "tmc" in result
        assert "tae_near" in result
        assert "tae_mid" in result
        assert "tae_far" in result
        assert "absrel_near" in result
        assert "rmse_far" in result
        assert "delta1_mid" in result

    def test_gt_absent_new_keys_nan(self):
        """Without GT, new GT-referenced keys should be nan."""
        T, H, W = 5, 20, 20
        pred = np.random.default_rng(42).uniform(0.5, 3.0, (T, H, W)).astype(np.float32)
        poses = _identity_poses(T)

        result = compute_temporal_depth_metrics(pred, poses=poses)
        assert np.isnan(result["tgse"])
        assert np.isnan(result["tmc"])
        assert np.isnan(result["absrel_near"])

"""Tests for ``scripts/pose_comprehensive_metrics.py``.

Pin the per-pair metrics + AUC math against simple analytic cases so a
refactor can't silently change rotation / translation conventions or
the AUC normalisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# pose_comprehensive_metrics lives under scripts/, not in the rpx_benchmark package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pose_comprehensive_metrics import (  # noqa: E402
    auc_pose_error,
    rotation_error_deg,
    translation_angular_deg,
    translation_l2,
)

# ────────────────────────  per-pair metrics  ──────────────────────────────


def test_rotation_error_deg_identity_is_zero():
    R = np.eye(3)
    assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-9)


def test_rotation_error_deg_matches_known_angle():
    """A 30° rotation about Z should yield exactly 30° geodesic error."""
    theta = np.deg2rad(30.0)
    R_pred = np.eye(3)
    R_gt = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    assert rotation_error_deg(R_pred, R_gt) == pytest.approx(30.0, abs=1e-6)


def test_rotation_error_deg_caps_at_180():
    """A 180° rotation about any axis yields exactly 180° error."""
    R_pred = np.eye(3)
    R_gt = np.diag([1, -1, -1]).astype(np.float64)  # 180° about X
    assert rotation_error_deg(R_pred, R_gt) == pytest.approx(180.0, abs=1e-6)


def test_translation_l2_known_distance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    assert translation_l2(a, b) == pytest.approx(1.0, abs=1e-9)


def test_translation_angular_deg_orthogonal():
    """Orthogonal unit vectors → 90° angular error."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert translation_angular_deg(a, b) == pytest.approx(90.0, abs=1e-6)


def test_translation_angular_deg_scale_invariant():
    """Same direction, different magnitudes → 0° angular error."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([5.0, 0.0, 0.0])
    assert translation_angular_deg(a, b) == pytest.approx(0.0, abs=1e-9)


def test_translation_angular_deg_zero_norm_returns_zero():
    """Zero-norm input should not blow up — return 0 by contract."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert translation_angular_deg(a, b) == 0.0


# ────────────────────────  AUC  ────────────────────────────────────────────


def test_auc_perfect_predictions_is_one():
    """When every pair has 0° error, AUC is 1.0 at every threshold."""
    errors = np.zeros(100)
    auc = auc_pose_error(errors, thresholds=(5.0, 10.0, 20.0))
    for k in ("auc_5deg", "auc_10deg", "auc_20deg"):
        assert auc[k] == pytest.approx(1.0, abs=1e-9), f"{k}: {auc[k]}"


def test_auc_all_above_threshold_is_zero():
    """When every pair fails by way more than the threshold, AUC is 0."""
    errors = np.full(100, 100.0)  # 100° error everywhere
    auc = auc_pose_error(errors, thresholds=(5.0, 10.0, 20.0))
    for k in ("auc_5deg", "auc_10deg", "auc_20deg"):
        assert auc[k] == pytest.approx(0.0, abs=1e-9), f"{k}: {auc[k]}"


def test_auc_empty_input_is_zero():
    auc = auc_pose_error(np.array([]), thresholds=(5.0,))
    assert auc["auc_5deg"] == 0.0


def test_auc_monotone_in_threshold():
    """AUC at higher thresholds should be ≥ AUC at lower ones (more pairs
    pass at higher thresholds, so cumulative fraction is higher)."""
    rng = np.random.default_rng(5_062_026)
    errors = rng.uniform(0, 30, size=500)
    auc = auc_pose_error(errors, thresholds=(5.0, 10.0, 20.0))
    assert auc["auc_5deg"] <= auc["auc_10deg"] + 1e-9
    assert auc["auc_10deg"] <= auc["auc_20deg"] + 1e-9

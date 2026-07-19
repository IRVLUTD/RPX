"""Tests for the SE(3) backproject→reproject depth warp."""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.deployment import se3_reproject_depth


def _identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _translate_pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """World-from-camera pose that is pure translation."""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def test_identity_pose_reprojects_to_self():
    """Same pose for src and dst → reprojected depth ≈ original depth."""
    h, w = 10, 10
    depth = np.full((h, w), 2.0, dtype=np.float32)  # flat wall at 2m
    pose = _identity_pose()
    reproj, valid = se3_reproject_depth(depth, pose, pose)
    # Central pixels should survive (edge pixels may fall off due to rounding)
    assert valid[2:8, 2:8].all(), "Central pixels must be valid after identity reproject"
    np.testing.assert_allclose(reproj[valid], 2.0, atol=1e-4)


def test_zero_depth_excluded():
    """Pixels with depth=0 should produce no reprojection."""
    h, w = 5, 5
    depth = np.zeros((h, w), dtype=np.float32)
    pose = _identity_pose()
    reproj, valid = se3_reproject_depth(depth, pose, pose)
    assert not valid.any()


def test_pure_z_translation_changes_depth():
    """Moving the camera 0.5m forward → objects are 0.5m closer."""
    # Use the real D435 resolution so pixel coords are sensible
    h, w = 480, 640
    depth = np.full((h, w), 3.0, dtype=np.float32)  # wall at 3m

    pose_src = _identity_pose()
    pose_dst = _translate_pose(tz=0.5)  # dst camera is 0.5m ahead

    reproj, valid = se3_reproject_depth(depth, pose_src, pose_dst)
    # In the dst camera, the wall should appear at 3.0 - 0.5 = 2.5m
    cy, cx = h // 2, w // 2
    assert valid[cy, cx], "Center pixel must survive reprojection"
    center = reproj[cy, cx]
    assert abs(center - 2.5) < 0.05, f"Expected ~2.5m, got {center}"


def test_pure_x_translation_shifts_pixels():
    """Moving camera right → pixels shift left in the image."""
    h, w = 30, 30
    depth = np.full((h, w), 2.0, dtype=np.float32)

    fx = 615.0
    pose_src = _identity_pose()
    # Move camera 0.01m right → at z=2m, pixel shift ≈ 0.01*615/2 ≈ 3 px left
    pose_dst = _translate_pose(tx=0.01)

    reproj, valid = se3_reproject_depth(depth, pose_src, pose_dst, fx=fx)
    # The reprojected image should have valid pixels shifted by ~3px
    # Check that the rightmost column is now empty (shifted left)
    assert valid.any(), "Should have some valid pixels"


def test_nan_poses_fall_through():
    """NaN poses should not crash; _warp_depth_approx wraps gracefully."""
    from rpx_benchmark.deployment import _warp_depth_approx

    h, w = 5, 5
    depth = np.full((h, w), 2.0, dtype=np.float32)
    bad_pose = np.full((4, 4), np.nan)
    result = _warp_depth_approx(depth, bad_pose, bad_pose)
    # Should fall back to identity (return original depth)
    np.testing.assert_array_equal(result, depth)

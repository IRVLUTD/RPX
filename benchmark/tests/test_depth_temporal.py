"""Tests for clip-level temporal depth metrics (Video Depth)."""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.exceptions import MetricError
from rpx_benchmark.metrics.depth_temporal import (
    compute_temporal_depth_metrics,
    optical_flow_warping_error,
    temporal_alignment_error,
    temporal_consistency_coefficient,
    temporal_gradient_matching,
)

H = W = 16


def _const(value: float) -> np.ndarray:
    return np.full((H, W), value, dtype=np.float32)


def _identity_poses(t: int) -> np.ndarray:
    return np.stack([np.eye(4, dtype=np.float64) for _ in range(t)])


# --------------------------------------------------------------------------- #
# TAE
# --------------------------------------------------------------------------- #


def test_tae_identity_poses_same_depth_is_zero():
    seq = np.stack([_const(2.0), _const(2.0)])
    tae = temporal_alignment_error(seq, _identity_poses(2))
    assert tae == pytest.approx(0.0, abs=1e-6)


def test_tae_identity_poses_diff_depth_is_deterministic():
    # Identity reprojection: reproj01 == pred[0]; compared to pred[1].
    # Forward AbsRel = |2.0-2.2|/2.2; backward = |2.2-2.0|/2.0; tae = mean.
    seq = np.stack([_const(2.0), _const(2.2)])
    expected = 0.5 * (abs(2.0 - 2.2) / 2.2 + abs(2.2 - 2.0) / 2.0)
    tae = temporal_alignment_error(seq, _identity_poses(2))
    assert tae == pytest.approx(expected, rel=1e-4)


def test_tae_nan_poses_yields_nan():
    seq = np.stack([_const(2.0), _const(2.1)])
    poses = np.full((2, 4, 4), np.nan)
    assert np.isnan(temporal_alignment_error(seq, poses))


def test_tae_rejects_bad_pose_shape():
    seq = np.stack([_const(2.0), _const(2.1)])
    with pytest.raises(MetricError):
        temporal_alignment_error(seq, np.zeros((2, 3, 3)))


# --------------------------------------------------------------------------- #
# TGM
# --------------------------------------------------------------------------- #


def test_tgm_perfect_prediction_is_zero():
    gt = np.stack([_const(2.0), _const(2.0)])
    assert temporal_gradient_matching(gt.copy(), gt) == pytest.approx(0.0)


def test_tgm_static_scene_penalises_predicted_change():
    # GT static (grad_gt = 0 -> all pixels "static"); prediction drifts 0.1 m.
    gt = np.stack([_const(2.0), _const(2.0)])
    pred = np.stack([_const(2.0), _const(2.1)])
    assert temporal_gradient_matching(pred, gt) == pytest.approx(0.1, rel=1e-4)


def test_tgm_ignores_dynamic_pixels():
    # GT moves 0.5 m (> 0.05 static thresh) everywhere -> no static pixels -> nan.
    gt = np.stack([_const(2.0), _const(2.5)])
    pred = np.stack([_const(2.0), _const(2.4)])
    assert np.isnan(temporal_gradient_matching(pred, gt))


# --------------------------------------------------------------------------- #
# TCC
# --------------------------------------------------------------------------- #


def test_tcc_identical_change_maps_is_one():
    gt = np.stack([_const(2.0), _const(2.3), _const(2.1)])
    tcc = temporal_consistency_coefficient(gt.copy(), gt)
    assert tcc == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# OPW
# --------------------------------------------------------------------------- #


def test_opw_requires_flow_backend():
    seq = np.stack([_const(2.0), _const(2.1)])
    rgb = np.zeros((2, H, W, 3), dtype=np.uint8)
    with pytest.raises(MetricError, match="flow backend"):
        optical_flow_warping_error(seq, rgb, None)


def test_opw_zero_flow_is_absrel_between_frames():
    # Zero backward flow -> warped depth_t == depth_t; compared to depth_{t+1}.
    seq = np.stack([_const(2.0), _const(2.2)])
    rgb = np.zeros((2, H, W, 3), dtype=np.uint8)
    zero_flow = lambda a, b: np.zeros((H, W, 2), dtype=np.float64)  # noqa: E731
    expected = abs(2.0 - 2.2) / 2.2
    assert optical_flow_warping_error(seq, rgb, zero_flow) == pytest.approx(expected, rel=1e-4)


def test_opw_warps_with_known_shift():
    # A horizontal depth ramp + a backward flow of (+1, 0): pixel x in frame
    # t+1 samples frame t at x+1. Warped depth at column x == ramp(x+1).
    ramp = np.tile(np.arange(1, W + 1, dtype=np.float32), (H, 1)) * 0.1 + 1.0  # 1.1..2.6
    seq = np.stack([ramp, ramp.copy()])
    rgb = np.zeros((2, H, W, 3), dtype=np.uint8)
    shift = lambda a, b: np.dstack(  # noqa: E731
        [np.ones((H, W)), np.zeros((H, W))]
    ).astype(np.float64)
    # Warp shifts sampling right by 1; last column clamps to border.
    opw = optical_flow_warping_error(seq, rgb, shift)
    assert opw > 0.0 and np.isfinite(opw)


# --------------------------------------------------------------------------- #
# Orchestrator — graceful degradation
# --------------------------------------------------------------------------- #


def test_orchestrator_without_poses_or_flow():
    pred = np.stack([_const(2.0), _const(2.1)])
    gt = np.stack([_const(2.0), _const(2.0)])
    out = compute_temporal_depth_metrics(pred, gt_seq=gt)
    assert {"tae", "opw", "tgm", "tcc", "tgse", "tmc"} <= set(out)
    assert np.isnan(out["tae"]) and np.isnan(out["opw"])
    assert all(np.isfinite(out[key]) for key in ("tgm", "tcc", "tgse", "tmc"))


def test_orchestrator_with_poses_computes_tae():
    pred = np.stack([_const(2.0), _const(2.1)])
    out = compute_temporal_depth_metrics(pred, poses=_identity_poses(2))
    assert np.isfinite(out["tae"])
    assert np.isnan(out["opw"]) and np.isnan(out["tgm"]) and np.isnan(out["tcc"])


def test_orchestrator_with_flow_computes_opw():
    pred = np.stack([_const(2.0), _const(2.2)])
    rgb = np.zeros((2, H, W, 3), dtype=np.uint8)
    zero_flow = lambda a, b: np.zeros((H, W, 2), dtype=np.float64)  # noqa: E731
    out = compute_temporal_depth_metrics(pred, rgb_seq=rgb, flow_fn=zero_flow)
    assert np.isfinite(out["opw"])

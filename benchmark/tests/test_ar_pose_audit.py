"""Tests for the SOS ALVAR-board camera-pose audit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rpx_benchmark.ar_pose_audit import (
    BoardObservation,
    alvar_dictionary,
    compare_pose_tracks,
    make_detector,
    marker_object_corners,
)


def _pose(translation=(0.0, 0.0, 0.0), rotation_deg=0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


def test_marker_geometry_is_board_centred_and_metric():
    marker_zero = marker_object_corners(0)
    assert marker_zero.shape == (4, 3)
    assert np.linalg.norm(marker_zero[1] - marker_zero[0]) == pytest.approx(0.059)
    assert marker_zero[:, 0].mean() == pytest.approx(0.0295 - 0.352)
    assert marker_zero[:, 1].mean() == pytest.approx(0.252 - 0.0295)


def test_canonical_alvar_code_round_trips_through_detector():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV contrib is not installed")
    marker = cv2.aruco.generateImageMarker(alvar_dictionary(), 13, 210, borderBits=2)
    canvas = np.full((900, 900), 255, dtype=np.uint8)
    canvas[345:555, 345:555] = marker
    _corners, ids, _rejected = make_detector().detectMarkers(canvas)
    assert ids.ravel().tolist() == [13]


def test_identical_relative_tracks_have_perfect_agreement():
    # Let the board be the world frame. camera_from_board is inverse c2w.
    poses = [_pose(), _pose((0.1, -0.02, 0.0), 5.0), _pose((0.2, -0.03, 0.01), 10.0)]
    observations = [BoardObservation(np.linalg.inv(pose), (0, 1), 0.2, 8) for pose in poses]
    metrics, rows = compare_pose_tracks(
        [0, 1, 2], poses, observations, pose_axis_convention="opencv"
    )
    assert metrics["anchor_relative_translation_rmse_m"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["anchor_relative_rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["static_board_translation_drift_rmse_m"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["static_board_translation_jitter"]["p95"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["static_board_rotation_jitter"]["p95"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["static_board_robust_outlier_rate"] == 0.0
    assert all("world_board_x_m" in row for row in rows)
    assert len(rows) == 3


def test_central_pose_jitter_is_not_biased_by_noisy_first_observation():
    camera_poses = [_pose(), _pose(), _pose(), _pose()]
    board_poses = [_pose((0.2, 0.0, 0.0)), _pose(), _pose(), _pose()]
    observations = [
        BoardObservation(np.linalg.inv(board_pose), (0, 1), 0.2, 8)
        for board_pose in board_poses
    ]
    metrics, rows = compare_pose_tracks(
        [0, 1, 2, 3], camera_poses, observations, pose_axis_convention="opencv"
    )
    assert metrics["static_board_translation_drift_rmse_m"] == pytest.approx(0.17320508)
    assert metrics["static_board_translation_jitter"]["median"] == pytest.approx(0.0)
    assert metrics["static_board_translation_jitter"]["p95"] == pytest.approx(0.17)
    assert rows[0]["static_board_robust_outlier"] is True


def test_world_origin_cancels_without_fitting_sensor_transform():
    ar_poses = [_pose(), _pose((0.1, 0.0, 0.0), 2.0), _pose((0.2, 0.02, 0.0), 4.0)]
    arbitrary_world_from_board = _pose((4.0, -2.0, 1.0), 35.0)
    cam_poses = [arbitrary_world_from_board @ pose for pose in ar_poses]
    observations = [BoardObservation(np.linalg.inv(pose), (2, 3, 4), 0.3, 12) for pose in ar_poses]
    metrics, _ = compare_pose_tracks(
        [10, 20, 30], cam_poses, observations, pose_axis_convention="opencv"
    )
    assert metrics["anchor_relative_translation_rmse_m"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["anchor_relative_rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-6)


def test_track_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="counts differ"):
        compare_pose_tracks([0], [_pose()], [], pose_axis_convention="opencv")


def test_t265_axis_conversion_matches_opencv_motion():
    opencv_poses = [_pose(), _pose((0.1, -0.03, 0.2), 8.0)]
    basis = np.diag([1.0, -1.0, -1.0, 1.0])
    raw_t265 = [basis @ pose @ basis for pose in opencv_poses]
    observations = [BoardObservation(np.linalg.inv(pose), (0, 1), 0.2, 8) for pose in opencv_poses]
    metrics, _ = compare_pose_tracks([0, 1], raw_t265, observations)
    assert metrics["anchor_relative_translation_rmse_m"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["anchor_relative_rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-6)

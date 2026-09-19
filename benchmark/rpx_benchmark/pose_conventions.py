"""Coordinate conventions shared by RPX camera-pose benchmarks."""

from __future__ import annotations

import numpy as np

from .exceptions import ConfigError

# librealsense T265: X right, Y up, Z back.
# OpenCV/VGGT:       X right, Y down, Z forward.
T265_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def t265_c2w_to_opencv(pose: np.ndarray) -> np.ndarray:
    """Express a raw T265 camera-to-world pose in OpenCV coordinates.

    RPX stores the unmodified librealsense position and quaternion. Changing
    the basis of both sides of the camera-to-world transform gives ``S T S``.
    This is a fixed convention conversion, not an estimated calibration.
    """

    transform = np.asarray(pose, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ConfigError(f"expected a 4x4 pose, got {transform.shape}")
    return T265_TO_OPENCV @ transform @ T265_TO_OPENCV


def relative_pose_from_raw_t265(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Return frame B in frame A, expressed in the OpenCV camera basis."""

    a = t265_c2w_to_opencv(pose_a)
    b = t265_c2w_to_opencv(pose_b)
    return np.linalg.inv(a) @ b


__all__ = ["T265_TO_OPENCV", "relative_pose_from_raw_t265", "t265_c2w_to_opencv"]

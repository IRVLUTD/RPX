"""AR-board consistency audit for RPX single-object camera poses.

The single-object captures contain an 18-marker FewSOL board in the D435 RGB
images and a synchronized camera-to-world pose saved from the T265.  The RPX
capture rig is documented by the dataset owners as already camera-frame
aligned.  This module deliberately does *not* estimate a D435-to-T265
extrinsic.  Instead it compares relative SE(3) motions, which also cancels the
unrelated board and T265 world origins.

The ALVAR ``MarkerData`` codes below are the canonical IDs 0--17 distributed
with ``ros-perception/ar_track_alvar``.  They are represented as an OpenCV
custom dictionary (5x5 payload, two-bit black border), allowing an offline
audit directly from RPX ``rgb.tar`` files without ROS.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

# ALVAR MarkerData IDs 0--17, sampled from the canonical marker generator.
_ALVAR_5X5_BITS = (
    "1101111011101011111111111",
    "1101111011101010011011101",
    "1101111011101011011010110",
    "1101111011101010111110100",
    "1101111011101010111001110",
    "1101111011101011011101100",
    "1101111011101010011100111",
    "1101111011101011111000101",
    "1101111011101010010111110",
    "1101111011101011110011100",
    "1101111011101010110010111",
    "1101111011101011010110101",
    "1101111011101011010001111",
    "1101111011101010110101101",
    "1101111011101011110100110",
    "1101111011101010010000100",
    "0001011001101011111111111",
    "0001011001101010011011101",
)

# FewSOL marker board geometry, metres.  Marker coordinates are measured from
# the upper-left of the 70.4 x 50.4 cm board; +Y below is converted to +Y up.
BOARD_WIDTH_M = 0.704
BOARD_HEIGHT_M = 0.504
MARKER_SIZE_M = 0.059
_MARKER_CENTRES_CM = (
    (2.95, 2.95),
    (15.85, 2.95),
    (28.75, 2.95),
    (41.55, 2.95),
    (54.45, 2.95),
    (67.45, 2.95),
    (67.45, 13.75),
    (67.45, 25.00),
    (67.45, 36.20),
    (67.45, 47.45),
    (54.45, 47.45),
    (41.55, 47.45),
    (28.70, 47.45),
    (15.80, 47.45),
    (2.95, 47.45),
    (2.95, 36.75),
    (2.95, 25.55),
    (2.95, 14.25),
)

# FewSOL's published D435 calibration.  CLI callers can override every value.
DEFAULT_CAMERA_MATRIX = np.array(
    [[611.10888672, 0.0, 315.51083374], [0.0, 610.02844238, 237.73669434], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# librealsense pose coordinates are X right, Y up, Z back; OpenCV RGB camera
# coordinates are X right, Y down, Z forward.  This fixed basis change is not a
# fitted camera extrinsic.  save_device_data.py writes the raw T265 quaternion
# and translation, so it must be applied unless poses were converted later.
_T265_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class BoardObservation:
    """One D435 observation of the static marker board."""

    camera_from_board: np.ndarray
    marker_ids: tuple[int, ...]
    reprojection_rmse_px: float
    inlier_corners: int


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "AR pose auditing requires OpenCV contrib; install "
            "`pip install 'rpx-benchmark[ar-pose-audit]'`."
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV was installed without the aruco module; use opencv-contrib-python-headless."
        )
    return cv2


def alvar_dictionary() -> Any:
    """Build the canonical ALVAR 0--17 dictionary for OpenCV detection."""

    cv2 = _cv2()
    rows = []
    for code in _ALVAR_5X5_BITS:
        bits = np.fromiter((int(bit) for bit in code), dtype=np.uint8).reshape(5, 5)
        rows.append(cv2.aruco.Dictionary_getByteListFromBits(bits))
    byte_list = np.concatenate(rows, axis=0)
    return cv2.aruco.Dictionary(byte_list, 5, 2)


def make_detector() -> Any:
    """Create an ALVAR detector tuned for the 640x480 RPX RGB frames."""

    cv2 = _cv2()
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 1.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.errorCorrectionRate = 0.8
    return cv2.aruco.ArucoDetector(alvar_dictionary(), params)


def marker_object_corners(marker_id: int) -> np.ndarray:
    """Return TL, TR, BR, BL corners in the board-centred coordinate frame."""

    if not 0 <= marker_id < len(_MARKER_CENTRES_CM):
        raise ValueError(f"unknown FewSOL marker id: {marker_id}")
    x_cm, y_cm = _MARKER_CENTRES_CM[marker_id]
    cx = x_cm * 0.01 - BOARD_WIDTH_M / 2.0
    cy = BOARD_HEIGHT_M / 2.0 - y_cm * 0.01
    half = MARKER_SIZE_M / 2.0
    return np.array(
        [
            [cx - half, cy + half, 0.0],
            [cx + half, cy + half, 0.0],
            [cx + half, cy - half, 0.0],
            [cx - half, cy - half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_board_pose(
    image: np.ndarray,
    detector: Any,
    camera_matrix: np.ndarray = DEFAULT_CAMERA_MATRIX,
    distortion: np.ndarray | None = None,
    *,
    reprojection_threshold_px: float = 4.0,
) -> BoardObservation | None:
    """Detect visible ALVAR tags and estimate ``T_camera_from_board``."""

    cv2 = _cv2()
    corners, ids, _rejected = detector.detectMarkers(image)
    if ids is None or not len(ids):
        return None

    # A duplicated ID is geometrically ambiguous; keep the largest detection.
    by_id: dict[int, np.ndarray] = {}
    for marker_id, image_corners in zip(ids.ravel().tolist(), corners, strict=True):
        marker_id = int(marker_id)
        candidate = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
        previous = by_id.get(marker_id)
        if previous is None or abs(cv2.contourArea(candidate.astype(np.float32))) > abs(
            cv2.contourArea(previous.astype(np.float32))
        ):
            by_id[marker_id] = candidate

    candidate_ids = tuple(sorted(by_id))
    object_points = np.concatenate([marker_object_corners(i) for i in candidate_ids])
    image_points = np.concatenate([by_id[i] for i in candidate_ids])
    distortion = (
        np.zeros(5, dtype=np.float64)
        if distortion is None
        else np.asarray(distortion, dtype=np.float64)
    )
    K = np.asarray(camera_matrix, dtype=np.float64)

    if len(candidate_ids) == 1:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        inliers = np.arange(4, dtype=np.int32).reshape(-1, 1)
    else:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            distortion,
            reprojectionError=float(reprojection_threshold_px),
            confidence=0.999,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok or inliers is None or len(inliers) < 4:
        return None

    keep = inliers.ravel()
    marker_ids = tuple(sorted({candidate_ids[int(index) // 4] for index in keep}))
    ok, rvec, tvec = cv2.solvePnP(
        object_points[keep],
        image_points[keep],
        K,
        distortion,
        rvec,
        tvec,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points[keep], rvec, tvec, K, distortion)
    residual = projected.reshape(-1, 2) - image_points[keep]
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    return BoardObservation(transform, marker_ids, rmse, int(len(keep)))


def pose_from_npz_bytes(payload: bytes) -> np.ndarray:
    """Load an RPX T265 camera-to-world pose from an NPZ payload."""

    with np.load(io.BytesIO(payload)) as data:
        position = np.asarray(data["position"], dtype=np.float64).reshape(3)
        quaternion = np.asarray(data["orientation"], dtype=np.float64).reshape(4)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_pose(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.linalg.inv(reference) @ current


def t265_to_opencv_pose(pose: np.ndarray) -> np.ndarray:
    """Express a raw librealsense T265 pose in the OpenCV axis convention."""

    return _T265_TO_OPENCV @ np.asarray(pose, dtype=np.float64) @ _T265_TO_OPENCV


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    """Return robust, publication-friendly statistics for one error series."""

    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: None for key in ("rmse", "mean", "median", "std", "p95", "max")}
    return {
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _robust_outliers(values: Sequence[float], *, z_threshold: float = 3.5) -> np.ndarray:
    """Flag high residuals using a median/MAD threshold without assuming Gaussian noise."""

    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return np.zeros(0, dtype=bool)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if float(np.ptp(array)) < 1e-9:
        return np.zeros(len(array), dtype=bool)
    if mad < 1e-12:
        return array > median + 1e-9
    robust_z = (array - median) / (1.4826 * mad)
    return robust_z > z_threshold


def _slope_norm(frame_ids: Sequence[int], vectors: np.ndarray) -> float | None:
    """Magnitude of a least-squares vector trend, expressed per source frame."""

    if len(frame_ids) < 2:
        return None
    x = np.asarray(frame_ids, dtype=np.float64)
    x -= x.mean()
    denominator = float(x @ x)
    if denominator < 1e-12:
        return None
    centred = np.asarray(vectors, dtype=np.float64) - np.mean(vectors, axis=0)
    slope = (x[:, None] * centred).sum(axis=0) / denominator
    return float(np.linalg.norm(slope))


def _rotation_medoid(rotations: Rotation) -> Rotation:
    """Choose the observed orientation with minimum median geodesic distance."""

    quaternions = rotations.as_quat()
    cosine_half_angles = np.clip(np.abs(quaternions @ quaternions.T), 0.0, 1.0)
    distances = 2.0 * np.arccos(cosine_half_angles)
    return rotations[int(np.argmin(np.median(distances, axis=1)))]


def compare_pose_tracks(
    frame_ids: Sequence[int],
    cam_poses: Sequence[np.ndarray],
    board_observations: Sequence[BoardObservation],
    *,
    pose_axis_convention: str = "t265",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare AR and saved motion without estimating a sensor transform."""

    if not (len(frame_ids) == len(cam_poses) == len(board_observations)):
        raise ValueError("frame, camera-pose and board-observation counts differ")
    if not frame_ids:
        return {"matched_frames": 0}, []

    if pose_axis_convention == "t265":
        cam_poses = [t265_to_opencv_pose(pose) for pose in cam_poses]
    elif pose_axis_convention != "opencv":
        raise ValueError("pose_axis_convention must be 't265' or 'opencv'")

    ar_poses = [np.linalg.inv(observation.camera_from_board) for observation in board_observations]
    cam_anchor, ar_anchor = cam_poses[0], ar_poses[0]
    cam_relative = [relative_pose(cam_anchor, pose) for pose in cam_poses]
    ar_relative = [relative_pose(ar_anchor, pose) for pose in ar_poses]

    frame_rows: list[dict[str, Any]] = []
    translation_errors = []
    rotation_errors = []
    cam_translations = []
    ar_translations = []
    cam_rotvecs = []
    ar_rotvecs = []
    for frame_id, cam_rel, ar_rel, observation in zip(
        frame_ids, cam_relative, ar_relative, board_observations, strict=True
    ):
        trans_error = float(np.linalg.norm(cam_rel[:3, 3] - ar_rel[:3, 3]))
        rot_error = rotation_error_deg(cam_rel, ar_rel)
        translation_errors.append(trans_error)
        rotation_errors.append(rot_error)
        cam_translations.append(cam_rel[:3, 3])
        ar_translations.append(ar_rel[:3, 3])
        cam_rotvecs.append(Rotation.from_matrix(cam_rel[:3, :3]).as_rotvec())
        ar_rotvecs.append(Rotation.from_matrix(ar_rel[:3, :3]).as_rotvec())
        frame_rows.append(
            {
                "frame_idx": int(frame_id),
                "detected": True,
                "marker_count": len(observation.marker_ids),
                "marker_ids": ";".join(str(i) for i in observation.marker_ids),
                "reprojection_rmse_px": observation.reprojection_rmse_px,
                "translation_error_m": trans_error,
                "rotation_error_deg": rot_error,
            }
        )

    cam_t = np.asarray(cam_translations)
    ar_t = np.asarray(ar_translations)
    cam_r = np.asarray(cam_rotvecs)
    ar_r = np.asarray(ar_rotvecs)

    step_translation_errors = []
    step_rotation_errors = []
    step_translation_errors_per_frame = []
    step_rotation_errors_per_frame = []
    step_gaps = []
    for index in range(1, len(frame_ids)):
        cam_step = relative_pose(cam_poses[index - 1], cam_poses[index])
        ar_step = relative_pose(ar_poses[index - 1], ar_poses[index])
        step_translation_errors.append(float(np.linalg.norm(cam_step[:3, 3] - ar_step[:3, 3])))
        step_rotation_errors.append(rotation_error_deg(cam_step, ar_step))
        gap = frame_ids[index] - frame_ids[index - 1]
        step_gaps.append(gap)
        step_translation_errors_per_frame.append(step_translation_errors[-1] / max(gap, 1))
        step_rotation_errors_per_frame.append(step_rotation_errors[-1] / max(gap, 1))

    # If the frames are aligned and the board is static, this composition must
    # be constant.  No world-origin or hand-eye fit is involved.
    world_from_boards = [
        cam @ obs.camera_from_board for cam, obs in zip(cam_poses, board_observations, strict=True)
    ]
    board_anchor = world_from_boards[0]
    board_translation_drift = [
        float(np.linalg.norm(pose[:3, 3] - board_anchor[:3, 3])) for pose in world_from_boards
    ]
    board_rotation_drift = [rotation_error_deg(board_anchor, pose) for pose in world_from_boards]

    # Jitter is measured around a sequence-level central pose rather than the
    # first observation, which may itself be noisy.  The translation median is
    # robust to isolated PnP failures; the geodesic rotation medoid avoids
    # quaternion sign ambiguity and is not pulled toward planar-PnP flips.
    board_positions = np.asarray([pose[:3, 3] for pose in world_from_boards])
    board_rotations = Rotation.from_matrix([pose[:3, :3] for pose in world_from_boards])
    board_position_centre = np.median(board_positions, axis=0)
    board_rotation_centre = _rotation_medoid(board_rotations)
    board_translation_jitter = np.linalg.norm(
        board_positions - board_position_centre[None, :], axis=1
    )
    board_rotation_jitter = np.degrees(
        (board_rotation_centre.inv() * board_rotations).magnitude()
    )
    translation_outliers = _robust_outliers(board_translation_jitter)
    rotation_outliers = _robust_outliers(board_rotation_jitter)
    combined_outliers = translation_outliers | rotation_outliers
    rotation_residual_vectors = (
        board_rotation_centre.inv() * board_rotations
    ).as_rotvec()

    for row, position, pose, translation_jitter, rotation_jitter, is_outlier in zip(
        frame_rows,
        board_positions,
        world_from_boards,
        board_translation_jitter,
        board_rotation_jitter,
        combined_outliers,
        strict=True,
    ):
        row.update(
            {
                "world_board_x_m": float(position[0]),
                "world_board_y_m": float(position[1]),
                "world_board_z_m": float(position[2]),
                "world_board_qx": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[0]),
                "world_board_qy": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[1]),
                "world_board_qz": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[2]),
                "world_board_qw": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[3]),
                "static_board_translation_jitter_m": float(translation_jitter),
                "static_board_rotation_jitter_deg": float(rotation_jitter),
                "static_board_robust_outlier": bool(is_outlier),
            }
        )

    def mean(values: Sequence[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def rmse(values: Sequence[float]) -> float | None:
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    summary: dict[str, Any] = {
        "matched_frames": len(frame_ids),
        "first_matched_frame": int(frame_ids[0]),
        "last_matched_frame": int(frame_ids[-1]),
        "translation_correlation": {
            "x": _pearson(cam_t[:, 0], ar_t[:, 0]),
            "y": _pearson(cam_t[:, 1], ar_t[:, 1]),
            "z": _pearson(cam_t[:, 2], ar_t[:, 2]),
            "all_components": _pearson(cam_t, ar_t),
            "magnitude": _pearson(np.linalg.norm(cam_t, axis=1), np.linalg.norm(ar_t, axis=1)),
        },
        "rotation_vector_correlation": {
            "x": _pearson(cam_r[:, 0], ar_r[:, 0]),
            "y": _pearson(cam_r[:, 1], ar_r[:, 1]),
            "z": _pearson(cam_r[:, 2], ar_r[:, 2]),
            "all_components": _pearson(cam_r, ar_r),
            "magnitude": _pearson(np.linalg.norm(cam_r, axis=1), np.linalg.norm(ar_r, axis=1)),
        },
        "anchor_relative_translation_rmse_m": rmse(translation_errors),
        "anchor_relative_translation_mean_m": mean(translation_errors),
        "anchor_relative_rotation_mean_deg": mean(rotation_errors),
        "anchor_relative_rotation_rmse_deg": rmse(rotation_errors),
        "step_translation_rmse_m": rmse(step_translation_errors),
        "step_rotation_rmse_deg": rmse(step_rotation_errors),
        "step_translation_rmse_m_per_frame": rmse(step_translation_errors_per_frame),
        "step_rotation_rmse_deg_per_frame": rmse(step_rotation_errors_per_frame),
        "mean_detected_frame_gap": mean(step_gaps),
        "static_board_translation_drift_rmse_m": rmse(board_translation_drift),
        "static_board_rotation_drift_rmse_deg": rmse(board_rotation_drift),
        "static_board_translation_jitter": _distribution(board_translation_jitter),
        "static_board_rotation_jitter": _distribution(board_rotation_jitter),
        "static_board_translation_jitter_robust_inliers": _distribution(
            board_translation_jitter[~combined_outliers]
        ),
        "static_board_rotation_jitter_robust_inliers": _distribution(
            board_rotation_jitter[~combined_outliers]
        ),
        "static_board_translation_drift_slope_m_per_frame": _slope_norm(
            frame_ids, board_positions
        ),
        "static_board_rotation_drift_slope_deg_per_frame": (
            None
            if (slope := _slope_norm(frame_ids, rotation_residual_vectors)) is None
            else float(np.degrees(slope))
        ),
        "static_board_robust_outlier_rate": float(np.mean(combined_outliers)),
        "mean_marker_count": mean([len(obs.marker_ids) for obs in board_observations]),
        "mean_reprojection_rmse_px": mean([obs.reprojection_rmse_px for obs in board_observations]),
    }
    return summary, frame_rows


def _member_map(archive: tarfile.TarFile, suffix: str) -> dict[int, tarfile.TarInfo]:
    result = {}
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith(suffix):
            try:
                result[int(Path(member.name).stem)] = member
            except ValueError:
                continue
    return result


def audit_sequence(
    sequence_dir: str | Path,
    camera_matrix: np.ndarray = DEFAULT_CAMERA_MATRIX,
    distortion: np.ndarray | None = None,
    *,
    max_frames: int | None = None,
    reprojection_threshold_px: float = 4.0,
    pose_axis_convention: str = "t265",
    min_markers: int = 2,
    max_reprojection_rmse_px: float = 2.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit one ``objects/<id>/0`` directory directly from tar shards."""

    sequence_dir = Path(sequence_dir)
    rgb_path = sequence_dir / "rgb.tar"
    pose_path = sequence_dir / "labels" / "cam_pose" / "v1.tar"
    if not rgb_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"missing RGB or camera-pose archive under {sequence_dir}")

    detector = make_detector()
    frame_ids: list[int] = []
    cam_poses: list[np.ndarray] = []
    observations: list[BoardObservation] = []
    all_rows: dict[int, dict[str, Any]] = {}
    raw_detections = 0
    cv2 = _cv2()
    with tarfile.open(rgb_path) as rgb_tar, tarfile.open(pose_path) as pose_tar:
        rgb_members = _member_map(rgb_tar, ".png")
        pose_members = _member_map(pose_tar, ".npz")
        candidates = sorted(set(rgb_members) & set(pose_members))
        if max_frames is not None:
            candidates = candidates[:max_frames]
        for frame_id in candidates:
            rgb_file = rgb_tar.extractfile(rgb_members[frame_id])
            pose_file = pose_tar.extractfile(pose_members[frame_id])
            if rgb_file is None or pose_file is None:
                continue
            encoded = np.frombuffer(rgb_file.read(), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            observation = estimate_board_pose(
                image,
                detector,
                camera_matrix,
                distortion,
                reprojection_threshold_px=reprojection_threshold_px,
            )
            accepted = observation is not None
            reject_reason = ""
            if observation is not None:
                raw_detections += 1
                if len(observation.marker_ids) < min_markers:
                    accepted = False
                    reject_reason = "insufficient_inlier_markers"
                elif observation.reprojection_rmse_px > max_reprojection_rmse_px:
                    accepted = False
                    reject_reason = "high_reprojection_error"
            all_rows[frame_id] = {
                "frame_idx": frame_id,
                "raw_detected": observation is not None,
                "detected": accepted,
                "reject_reason": reject_reason,
            }
            if observation is not None:
                all_rows[frame_id].update(
                    {
                        "marker_count": len(observation.marker_ids),
                        "marker_ids": ";".join(str(i) for i in observation.marker_ids),
                        "reprojection_rmse_px": observation.reprojection_rmse_px,
                    }
                )
            if accepted and observation is not None:
                frame_ids.append(frame_id)
                cam_poses.append(pose_from_npz_bytes(pose_file.read()))
                observations.append(observation)

    metrics, matched_rows = compare_pose_tracks(
        frame_ids,
        cam_poses,
        observations,
        pose_axis_convention=pose_axis_convention,
    )
    for row in matched_rows:
        all_rows[int(row["frame_idx"])].update(row)
    metrics.update(
        {
            "object_id": sequence_dir.parent.name,
            "sequence": str(sequence_dir),
            "total_paired_frames": len(all_rows),
            "detection_coverage": len(frame_ids) / max(len(all_rows), 1),
            "raw_detected_frames": raw_detections,
            "raw_detection_coverage": raw_detections / max(len(all_rows), 1),
        }
    )
    rows = [dict(object_id=sequence_dir.parent.name, **all_rows[i]) for i in sorted(all_rows)]
    return metrics, rows


def aggregate_sequence_metrics(sequences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce frame-weighted aggregate values from per-sequence summaries."""

    total = sum(int(item.get("total_paired_frames", 0)) for item in sequences)
    matched = sum(int(item.get("matched_frames", 0)) for item in sequences)
    raw_matched = sum(int(item.get("raw_detected_frames", 0)) for item in sequences)
    scalar_keys = (
        "anchor_relative_translation_rmse_m",
        "anchor_relative_rotation_rmse_deg",
        "step_translation_rmse_m",
        "step_rotation_rmse_deg",
        "step_translation_rmse_m_per_frame",
        "step_rotation_rmse_deg_per_frame",
        "static_board_translation_drift_rmse_m",
        "static_board_rotation_drift_rmse_deg",
        "static_board_translation_drift_slope_m_per_frame",
        "static_board_rotation_drift_slope_deg_per_frame",
        "static_board_robust_outlier_rate",
        "mean_marker_count",
        "mean_reprojection_rmse_px",
    )
    result: dict[str, Any] = {
        "sequences": len(sequences),
        "sequences_with_detections": sum(
            int(item.get("matched_frames", 0)) > 0 for item in sequences
        ),
        "total_frames": total,
        "matched_frames": matched,
        "detection_coverage": matched / max(total, 1),
        "raw_detected_frames": raw_matched,
        "raw_detection_coverage": raw_matched / max(total, 1),
    }
    for key in scalar_keys:
        values = [
            (float(item[key]), int(item.get("matched_frames", 0)))
            for item in sequences
            if item.get(key) is not None and int(item.get("matched_frames", 0)) > 0
        ]
        result[key] = (
            float(
                np.average([value for value, _ in values], weights=[weight for _, weight in values])
            )
            if values
            else None
        )
    for family in ("translation_correlation", "rotation_vector_correlation"):
        result[family] = {}
        for component in ("x", "y", "z", "all_components", "magnitude"):
            values = [
                (float(item[family][component]), int(item["matched_frames"]))
                for item in sequences
                if item.get(family, {}).get(component) is not None
            ]
            result[family][component] = (
                float(
                    np.average(
                        [value for value, _ in values], weights=[weight for _, weight in values]
                    )
                )
                if values
                else None
            )
    return result


def write_results(
    output_dir: str | Path,
    sequence_metrics: Sequence[Mapping[str, Any]],
    frame_rows: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write JSON, CSV and a concise Markdown report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = list(frame_rows)
    aggregate = aggregate_sequence_metrics(sequence_metrics)
    valid_rows = [row for row in frame_rows if row.get("detected")]
    for family, column in (
        ("static_board_translation_jitter", "static_board_translation_jitter_m"),
        ("static_board_rotation_jitter", "static_board_rotation_jitter_deg"),
    ):
        aggregate[family] = _distribution(
            [float(row[column]) for row in valid_rows if row.get(column) is not None]
        )
    robust_inlier_rows = [
        row for row in valid_rows if not bool(row.get("static_board_robust_outlier"))
    ]
    for family, column in (
        (
            "static_board_translation_jitter_robust_inliers",
            "static_board_translation_jitter_m",
        ),
        (
            "static_board_rotation_jitter_robust_inliers",
            "static_board_rotation_jitter_deg",
        ),
    ):
        aggregate[family] = _distribution(
            [float(row[column]) for row in robust_inlier_rows if row.get(column) is not None]
        )
    aggregate["static_board_robust_outlier_rate"] = (
        float(np.mean([bool(row.get("static_board_robust_outlier")) for row in valid_rows]))
        if valid_rows
        else None
    )
    payload = {
        "protocol": dict(metadata),
        "aggregate": aggregate,
        "per_sequence": list(sequence_metrics),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    flattened = []
    nested_families = (
        "translation_correlation",
        "rotation_vector_correlation",
        "static_board_translation_jitter",
        "static_board_rotation_jitter",
        "static_board_translation_jitter_robust_inliers",
        "static_board_rotation_jitter_robust_inliers",
    )
    for item in sequence_metrics:
        row = {key: value for key, value in item.items() if not isinstance(value, Mapping)}
        for family in nested_families:
            for key, value in item.get(family, {}).items():
                row[f"{family}.{key}"] = value
        flattened.append(row)
    _write_csv(output_dir / "per_sequence.csv", flattened)
    _write_csv(output_dir / "per_frame.csv", frame_rows)

    def fmt(value: Any, digits: int = 4) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    report = f"""# RPX SOS AR-board / camera-pose audit

## Protocol

- Dataset: `{metadata.get("dataset_repo", "unknown")}` at revision `{metadata.get("dataset_revision", "unknown")}`
- Sequences: {aggregate["sequences"]} ({aggregate["sequences_with_detections"]} with detections)
- Frames inspected: {aggregate["total_frames"]}
- Frames with a valid AR-board pose: {aggregate["matched_frames"]} ({aggregate["detection_coverage"]:.1%})
- Frames with any raw tag pose: {aggregate["raw_detected_frames"]} ({aggregate["raw_detection_coverage"]:.1%})
- Camera matrix: `{metadata["camera_matrix"]}`
- Sensor-frame assumption: D435 RGB and saved T265 camera poses are already aligned.
- Sensor transform fitting: **none**.
- World-origin handling: relative poses from the first detected frame in each sequence.
- Primary quality gate: at least {metadata["min_markers"]} RANSAC-consistent markers and reprojection RMSE <= {metadata["max_reprojection_rmse_px"]} px.

## Aggregate results

| Measure | Value |
|---|---:|
| Translation correlation, all components | {fmt(aggregate["translation_correlation"]["all_components"])} |
| Translation-magnitude correlation | {fmt(aggregate["translation_correlation"]["magnitude"])} |
| Rotation-vector correlation, all components | {fmt(aggregate["rotation_vector_correlation"]["all_components"])} |
| Rotation-magnitude correlation | {fmt(aggregate["rotation_vector_correlation"]["magnitude"])} |
| Anchor-relative translation RMSE | {fmt(aggregate["anchor_relative_translation_rmse_m"])} m |
| Anchor-relative rotation RMSE | {fmt(aggregate["anchor_relative_rotation_rmse_deg"])} deg |
| Step translation RMSE | {fmt(aggregate["step_translation_rmse_m"])} m |
| Step rotation RMSE | {fmt(aggregate["step_rotation_rmse_deg"])} deg |
| Gap-normalized step translation RMSE | {fmt(aggregate["step_translation_rmse_m_per_frame"])} m/frame |
| Gap-normalized step rotation RMSE | {fmt(aggregate["step_rotation_rmse_deg_per_frame"])} deg/frame |
| Static-board translation drift RMSE | {fmt(aggregate["static_board_translation_drift_rmse_m"])} m |
| Static-board rotation drift RMSE | {fmt(aggregate["static_board_rotation_drift_rmse_deg"])} deg |
| Static-board translation jitter, median / P95 | {fmt(aggregate["static_board_translation_jitter"]["median"])} / {fmt(aggregate["static_board_translation_jitter"]["p95"])} m |
| Static-board rotation jitter, median / P95 | {fmt(aggregate["static_board_rotation_jitter"]["median"])} / {fmt(aggregate["static_board_rotation_jitter"]["p95"])} deg |
| Robust-inlier translation jitter, median / P95 | {fmt(aggregate["static_board_translation_jitter_robust_inliers"]["median"])} / {fmt(aggregate["static_board_translation_jitter_robust_inliers"]["p95"])} m |
| Robust-inlier rotation jitter, median / P95 | {fmt(aggregate["static_board_rotation_jitter_robust_inliers"]["median"])} / {fmt(aggregate["static_board_rotation_jitter_robust_inliers"]["p95"])} deg |
| Translation drift slope | {fmt(aggregate["static_board_translation_drift_slope_m_per_frame"], 6)} m/frame |
| Rotation drift slope | {fmt(aggregate["static_board_rotation_drift_slope_deg_per_frame"], 6)} deg/frame |
| Robust residual outlier rate | {fmt(aggregate["static_board_robust_outlier_rate"])} |
| Mean marker count | {fmt(aggregate["mean_marker_count"], 2)} |
| Mean marker reprojection RMSE | {fmt(aggregate["mean_reprojection_rmse_px"])} px |

## Automated finding

The AR-derived and stored motion tracks are strongly correlated (translation
`{fmt(aggregate["translation_correlation"]["all_components"])}`, rotation
`{fmt(aggregate["rotation_vector_correlation"]["all_components"])}`). Across accepted frames,
the fixed-board residual has median/P95 translation
`{fmt(aggregate["static_board_translation_jitter"]["median"])} / {fmt(aggregate["static_board_translation_jitter"]["p95"])} m`
and median/P95 rotation
`{fmt(aggregate["static_board_rotation_jitter"]["median"])} / {fmt(aggregate["static_board_rotation_jitter"]["p95"])} deg`.
After removing the automatically identified median/MAD residual outliers, those P95 values are
`{fmt(aggregate["static_board_translation_jitter_robust_inliers"]["p95"])} m` and
`{fmt(aggregate["static_board_rotation_jitter_robust_inliers"]["p95"])} deg`.

These values show measurable disagreement, but they are an upper bound on T265 pose error rather
than a pure T265 accuracy measurement. The released files do not contain the capture device's
factory distortion coefficients or a numerical D435-to-T265 extrinsic. The residual therefore also
contains tag-corner/PnP uncertainty, approximate-intrinsics error, synchronization error and any
unmodelled sensor-frame offset. Large orientation branches in the per-sequence PDF should be
treated as planar-PnP/reference failures until independently verified.

## Interpretation

Relative-motion metrics remove only the arbitrary world origin; they do not learn or apply a
D435-to-T265 transform. The static-board residual is an independent consistency check: with
correct synchronization, conventions and the stated camera-frame alignment, the composed board
pose should be constant. Review per-sequence and per-frame files before drawing conclusions from
the aggregate correlations, especially where marker coverage or reprojection quality is low.

The first-frame drift metrics are directly comparable to anchor-based ATE. The central-pose jitter
metrics use the sequence median position and rotation medoid, making them less sensitive to one
bad anchor. Gap-normalized step errors are the RPE-style local-motion measurements. A robust
outlier is a translation or rotation residual more than 3.5 scaled MAD above its sequence median.
"""
    (output_dir / "report.md").write_text(report)
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

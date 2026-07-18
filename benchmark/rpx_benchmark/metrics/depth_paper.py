"""Canonical RPX D1-F paper metrics and fixed camera calibration.

The RPX paper declares one RGB-camera calibration for the released 640x480
benchmark.  The release does not include per-device calibration files, so this
module deliberately fails on any other resolution rather than inventing a
rescaling rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..api import DepthGroundTruth, DepthPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricSuite

DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 5.0
FSCORE_THRESHOLD_M = 0.05
EXPECTED_HW = (480, 640)


@dataclass(frozen=True)
class D1Calibration:
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    version: str = "rpx-paper-d1-rgb-640x480-v1"
    provenance: str = "RPX-4.pdf Appendix I.1; fixed paper-declared RGB intrinsics"

    def to_dict(self) -> dict[str, object]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "version": self.version,
            "provenance": self.provenance,
            "intrinsics_policy": "fixed paper-declared calibration; never rescaled",
            "coordinate_system": "RGB camera",
            "pixel_origin": "top-left pixel centre",
            "released_depth_alignment": "depth aligned to RGB color stream",
            "per_device_calibration": "not present in released HF manifests",
            "extrinsics_used": False,
        }


D1_CALIBRATION = D1Calibration()
PAPER_METRIC_KEYS = (
    "absrel",
    "rmse",
    "silog",
    "delta1",
    "irmse",
    "fscore_5cm",
)


def valid_depth_mask(gt_m: np.ndarray) -> np.ndarray:
    """Paper validity mask: finite GT strictly inside the D435 range."""
    return np.isfinite(gt_m) & (gt_m > DEPTH_MIN_M) & (gt_m < DEPTH_MAX_M)


def _point_cloud(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(valid)
    z = depth[valid].astype(np.float64, copy=False)
    return np.column_stack(
        (
            (x.astype(np.float64) - D1_CALIBRATION.cx) * z / D1_CALIBRATION.fx,
            (y.astype(np.float64) - D1_CALIBRATION.cy) * z / D1_CALIBRATION.fy,
            z,
        )
    )


def _matched_fraction(tree, points: np.ndarray, *, chunk_size: int = 65_536) -> float:
    matched = 0
    for start in range(0, len(points), chunk_size):
        distances, _ = tree.query(
            points[start : start + chunk_size],
            k=1,
            workers=1,
        )
        matched += int(np.count_nonzero(distances < FSCORE_THRESHOLD_M))
    return float(matched / len(points))


def point_cloud_fscore_5cm(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Bidirectional nearest-neighbour F-score at a strict 5 cm threshold."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - Docker and test env include scipy
        raise MetricError(
            "F-Score@5cm requires scipy.spatial.cKDTree.",
            hint="Install scipy>=1.10 or use the RPX depth-paper Docker image.",
        ) from exc

    gt_points = _point_cloud(gt_m, valid)
    pred_points = _point_cloud(pred_m, valid)
    if len(gt_points) == 0 or len(pred_points) == 0:
        raise MetricError("F-Score@5cm received an empty valid point cloud.")

    precision = _matched_fraction(cKDTree(gt_points), pred_points)
    recall = _matched_fraction(cKDTree(pred_points), gt_points)
    denom = precision + recall
    return 0.0 if denom == 0.0 else float(2.0 * precision * recall / denom)


def compute_d1_paper_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute the six D1-F headline metrics for one 640x480 frame."""
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(ground_truth, dtype=np.float32)
    if pred.shape != gt.shape:
        raise MetricError(
            f"Depth prediction shape {pred.shape} does not match ground-truth shape {gt.shape}."
        )
    if gt.shape != EXPECTED_HW:
        raise MetricError(
            f"RPX D1-F paper calibration requires depth shape {EXPECTED_HW}; got {gt.shape}.",
            hint="Do not rescale the fixed paper intrinsics; use the released 640x480 frames.",
        )

    valid = valid_depth_mask(gt)
    if not valid.any():
        raise MetricError("Depth frame contains no finite GT pixels in the 0.3-5.0 m range.")
    pred_valid = pred[valid]
    if not np.isfinite(pred_valid).all() or np.any(pred_valid <= 0):
        raise MetricError(
            "Depth prediction has non-finite or non-positive values on paper-valid GT pixels."
        )

    gt_valid = gt[valid]
    diff = pred_valid - gt_valid
    log_error = np.log(pred_valid) - np.log(gt_valid)
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    silog_variance = max(
        float(np.mean(log_error**2) - np.mean(log_error) ** 2),
        0.0,
    )
    metrics = {
        "absrel": float(np.mean(np.abs(diff) / gt_valid)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "silog": float(100.0 * np.sqrt(silog_variance)),
        "delta1": float(np.mean(ratio < 1.25)),
        "irmse": float(np.sqrt(np.mean((1.0 / pred_valid - 1.0 / gt_valid) ** 2))),
        "fscore_5cm": point_cloud_fscore_5cm(pred, gt, valid),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise MetricError("D1-F paper metric computation produced a non-finite value.")
    return metrics


class PaperDepthMetricSuite(MetricSuite):
    """Metric suite dedicated to the strict RPX D1-F paper protocol."""

    def __init__(self) -> None:
        super().__init__(TaskType.MONOCULAR_DEPTH)

    def evaluate(self, prediction: object, ground_truth: object) -> dict[str, float]:
        if not isinstance(prediction, DepthPrediction):
            raise MetricError(
                f"PaperDepthMetricSuite expected DepthPrediction, got {type(prediction).__name__}."
            )
        if not isinstance(ground_truth, DepthGroundTruth):
            raise MetricError(
                "PaperDepthMetricSuite expected DepthGroundTruth, "
                f"got {type(ground_truth).__name__}."
            )
        return compute_d1_paper_metrics(prediction.depth_map, ground_truth.depth_map)

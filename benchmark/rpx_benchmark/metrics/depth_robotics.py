"""Robotics-deployment depth metric calculators.

Camera-frame Chamfer-L1 + grasp F-score, surface normals (MAE, %11.25°),
and scale-invariant boundary F1.  All are **pose-free** (intrinsics
only) — they measure shape/reachability, not pose drift.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..api import DepthGroundTruth, DepthPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric

# D435 intrinsics at 640×480
_FX: float = 615.0
_FY: float = 615.0
_CX: float = 320.0
_CY: float = 240.0

# Max points for Chamfer/F-score (computational tractability)
_MAX_POINTS: int = 50_000


def _backproject(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Backproject valid depth pixels to 3D in camera frame.

    Returns (N, 3) float64 array of 3D points.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                        np.arange(h, dtype=np.float64))
    z = depth[valid].astype(np.float64)
    x = (u[valid] - _CX) * z / _FX
    y = (v[valid] - _CY) * z / _FY
    return np.stack([x, y, z], axis=-1)


def _subsample(pts: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) <= max_n:
        return pts
    idx = rng.choice(len(pts), size=max_n, replace=False)
    return pts[idx]


def _chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer-L1 distance between two point clouds.

    Uses brute-force nearest-neighbour (OK for ≤50K points).
    """
    # a→b
    # Chunked to avoid OOM on large clouds
    chunk = 5000
    dists_a = np.empty(len(a), dtype=np.float64)
    for i in range(0, len(a), chunk):
        end = min(i + chunk, len(a))
        diff = a[i:end, None, :] - b[None, :, :]  # (chunk, M, 3)
        dists_a[i:end] = np.min(np.sum(np.abs(diff), axis=-1), axis=1)

    dists_b = np.empty(len(b), dtype=np.float64)
    for i in range(0, len(b), chunk):
        end = min(i + chunk, len(b))
        diff = b[i:end, None, :] - a[None, :, :]
        dists_b[i:end] = np.min(np.sum(np.abs(diff), axis=-1), axis=1)

    return float(np.mean(dists_a) + np.mean(dists_b))


def _fscore(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    """F-score: fraction of points in each cloud within *threshold* of the other."""
    chunk = 5000

    # a→b distances (L2)
    dists_a = np.empty(len(a), dtype=np.float64)
    for i in range(0, len(a), chunk):
        end = min(i + chunk, len(a))
        diff = a[i:end, None, :] - b[None, :, :]
        dists_a[i:end] = np.min(np.sqrt(np.sum(diff ** 2, axis=-1)), axis=1)

    # b→a
    dists_b = np.empty(len(b), dtype=np.float64)
    for i in range(0, len(b), chunk):
        end = min(i + chunk, len(b))
        diff = b[i:end, None, :] - a[None, :, :]
        dists_b[i:end] = np.min(np.sqrt(np.sum(diff ** 2, axis=-1)), axis=1)

    precision = float(np.mean(dists_a < threshold))
    recall = float(np.mean(dists_b < threshold))
    if precision + recall < 1e-12:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthGrasp3D(MetricCalculator):
    """Camera-frame Chamfer-L1 and grasp F-score @ {1, 2, 5, 10 cm}.

    Back-projects predicted and GT depth to 3D using D435 intrinsics
    only (no camera pose) — measures shape/reachability, not pose drift.
    """

    name = "depth_grasp_3d"

    def compute(
        self,
        prediction: DepthPrediction,
        ground_truth: DepthGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, DepthPrediction):
            raise MetricError(
                f"DepthGrasp3D expected DepthPrediction, got {type(prediction).__name__}",
            )
        if not isinstance(ground_truth, DepthGroundTruth):
            raise MetricError(
                f"DepthGrasp3D expected DepthGroundTruth, got {type(ground_truth).__name__}",
            )

        pred = np.asarray(prediction.depth_map, dtype=np.float32)
        gt = np.asarray(ground_truth.depth_map, dtype=np.float32)

        from .depth_alignment import default_valid_mask

        valid = default_valid_mask(pred, gt)
        if valid.sum() < 10:
            return {
                "chamfer_l1": float("nan"),
                "fscore_1cm": 0.0,
                "fscore_2cm": 0.0,
                "fscore_5cm": 0.0,
                "fscore_10cm": 0.0,
            }

        # Use independent RNGs so identical pred/gt get the same subset
        rng_pred = np.random.default_rng(42)
        rng_gt = np.random.default_rng(42)
        pts_pred = _subsample(_backproject(pred, valid), _MAX_POINTS, rng_pred)
        pts_gt = _subsample(_backproject(gt, valid), _MAX_POINTS, rng_gt)

        chamfer = _chamfer_l1(pts_pred, pts_gt)

        return {
            "chamfer_l1": chamfer,
            "fscore_1cm": _fscore(pts_pred, pts_gt, 0.01),
            "fscore_2cm": _fscore(pts_pred, pts_gt, 0.02),
            "fscore_5cm": _fscore(pts_pred, pts_gt, 0.05),
            "fscore_10cm": _fscore(pts_pred, pts_gt, 0.10),
        }


# --------------------------------------------------------------------------- #
# Surface normals from depth
# --------------------------------------------------------------------------- #

def _depth_to_normals(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute surface normals from depth via finite differences + backprojection.

    Returns (H, W, 3) normals and (H, W) bool mask of pixels where
    normals are reliable (both neighbours valid and away from holes).
    """
    h, w = depth.shape
    z = depth.astype(np.float64)

    # Backproject to 3D (per-pixel camera-frame positions)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                        np.arange(h, dtype=np.float64))
    x = (u - _CX) * z / _FX
    y = (v - _CY) * z / _FY

    # Finite-difference tangent vectors (central where possible)
    dx = np.zeros((h, w, 3), dtype=np.float64)
    dy = np.zeros((h, w, 3), dtype=np.float64)

    dx[:, 1:-1, 0] = x[:, 2:] - x[:, :-2]
    dx[:, 1:-1, 1] = y[:, 2:] - y[:, :-2]
    dx[:, 1:-1, 2] = z[:, 2:] - z[:, :-2]

    dy[1:-1, :, 0] = x[2:, :] - x[:-2, :]
    dy[1:-1, :, 1] = y[2:, :] - y[:-2, :]
    dy[1:-1, :, 2] = z[2:, :] - z[:-2, :]

    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1e-8)
    normals = normals / norm

    # Mask: valid depth at this pixel and all 4 finite-diff neighbours;
    # exclude 1px border and k=2px around holes.
    neighbour_valid = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1] & valid[2:, 1:-1]
        & valid[1:-1, :-2] & valid[1:-1, 2:]
    )
    normal_valid = np.zeros((h, w), dtype=bool)
    normal_valid[1:-1, 1:-1] = neighbour_valid

    return normals, normal_valid


@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthNormals(MetricCalculator):
    """Surface normal MAE and %11.25° from depth-derived normals.

    Normals are derived via finite differences on the back-projected
    3D positions (intrinsics only, no pose). Pixels near depth holes
    and image borders are masked out to avoid amplifying D435 noise.
    """

    name = "depth_normals"

    def compute(
        self,
        prediction: DepthPrediction,
        ground_truth: DepthGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, DepthPrediction):
            raise MetricError(
                f"DepthNormals expected DepthPrediction, got {type(prediction).__name__}",
            )
        if not isinstance(ground_truth, DepthGroundTruth):
            raise MetricError(
                f"DepthNormals expected DepthGroundTruth, got {type(ground_truth).__name__}",
            )

        pred = np.asarray(prediction.depth_map, dtype=np.float32)
        gt = np.asarray(ground_truth.depth_map, dtype=np.float32)

        from .depth_alignment import default_valid_mask

        valid = default_valid_mask(pred, gt)

        pred_normals, pred_nvalid = _depth_to_normals(pred, valid)
        gt_normals, gt_nvalid = _depth_to_normals(gt, valid)

        both_valid = pred_nvalid & gt_nvalid
        if both_valid.sum() < 10:
            return {"normal_mae": float("nan"), "normal_acc1125": 0.0}

        # Cosine similarity → angular error (degrees)
        cos_sim = np.sum(
            pred_normals[both_valid] * gt_normals[both_valid], axis=-1
        )
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(np.abs(cos_sim)))  # abs: sign ambiguity

        normal_mae = float(np.mean(angles_deg))
        normal_acc1125 = float(np.mean(angles_deg < 11.25))

        return {
            "normal_mae": normal_mae,
            "normal_acc1125": normal_acc1125,
        }


# --------------------------------------------------------------------------- #
# Scale-invariant boundary F1
# --------------------------------------------------------------------------- #

def _depth_ratio_contours(depth: np.ndarray, threshold_pct: float) -> np.ndarray:
    """Binary occluding-contour map from depth ratios.

    A pixel (i,j) is an occluding contour if any 4-connected neighbour
    satisfies d(neighbour)/d(pixel) > 1 + threshold_pct/100.
    """
    t = 1.0 + threshold_pct / 100.0
    h, w = depth.shape
    contour = np.zeros((h, w), dtype=bool)

    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        sy = slice(max(0, dy), h + min(0, dy))
        sx = slice(max(0, dx), w + min(0, dx))
        ny = slice(max(0, -dy), h + min(0, -dy))
        nx = slice(max(0, -dx), w + min(0, -dx))
        d_self = depth[sy, sx]
        d_neigh = depth[ny, nx]
        valid_pair = (d_self > 1e-6) & (d_neigh > 1e-6)
        ratio = np.ones_like(d_self)
        ratio[valid_pair] = d_neigh[valid_pair] / d_self[valid_pair]
        contour[sy, sx] |= (ratio > t) & valid_pair

    return contour


@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthBoundaryF1(MetricCalculator):
    """Scale-invariant occluding-contour F1 from depth ratios.

    Averages F1 over thresholds t ∈ {5, 10, 15, 20, 25} (percent).
    No GT boundary annotations needed — contours are derived from
    depth ratios on both predicted and GT depth maps.
    """

    name = "depth_boundary_f1"

    _thresholds = (5, 10, 15, 20, 25)

    def compute(
        self,
        prediction: DepthPrediction,
        ground_truth: DepthGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, DepthPrediction):
            raise MetricError(
                f"DepthBoundaryF1 expected DepthPrediction, got {type(prediction).__name__}",
            )
        if not isinstance(ground_truth, DepthGroundTruth):
            raise MetricError(
                f"DepthBoundaryF1 expected DepthGroundTruth, got {type(ground_truth).__name__}",
            )

        pred = np.asarray(prediction.depth_map, dtype=np.float32)
        gt = np.asarray(ground_truth.depth_map, dtype=np.float32)

        f1_scores = []
        for t in self._thresholds:
            pred_c = _depth_ratio_contours(pred, t)
            gt_c = _depth_ratio_contours(gt, t)

            tp = float((pred_c & gt_c).sum())
            fp = float((pred_c & ~gt_c).sum())
            fn = float((~pred_c & gt_c).sum())

            if tp + fp < 1e-6 and tp + fn < 1e-6:
                # No contours in either → perfect agreement
                f1_scores.append(1.0)
            elif tp < 1e-6:
                f1_scores.append(0.0)
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1_scores.append(2 * prec * rec / (prec + rec))

        return {"boundary_f1": float(np.mean(f1_scores))}

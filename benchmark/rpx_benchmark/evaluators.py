"""Legacy metric module (compat shim).

``MetricSuite`` and ``BenchmarkResult`` have moved to
:mod:`rpx_benchmark.metrics`. This file still re-exports them so old
imports continue to work, and it still hosts the raw numpy metric
*functions* (``depth_metrics``, ``detection_metrics``, etc.) used
internally by the calculator classes.

New code should import from :mod:`rpx_benchmark.metrics` instead.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

# Re-export the new registry-backed facade so old ``from
# rpx_benchmark.evaluators import MetricSuite`` imports keep working.
from .metrics.registry import BenchmarkResult, MetricSuite  # noqa: F401

# ------------------------------------------------------------------ #
# Task-specific metric functions
# ------------------------------------------------------------------ #


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """AbsRel, RMSE, and threshold accuracy (δ<1.25) for monocular depth."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    valid = gt > 0
    if valid.sum() == 0:
        return {"rmse": 0.0, "absrel": 0.0, "delta1": 1.0, "delta2": 1.0, "delta3": 1.0}

    pred_v, gt_v = pred[valid], gt[valid]
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
    absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))

    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25**2))
    delta3 = float(np.mean(thresh < 1.25**3))

    return {"rmse": rmse, "absrel": absrel, "delta1": delta1, "delta2": delta2, "delta3": delta3}


def detection_metrics(
    pred_boxes: np.ndarray,
    pred_scores: Sequence[float],
    pred_labels: Sequence[str],
    gt_boxes: np.ndarray,
    gt_labels: Sequence[str],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Greedy matching at a single IoU threshold → precision/recall/F1."""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_indices = np.argsort(-np.asarray(pred_scores))
    matched_gt: set[int] = set()
    tp = 0
    for idx in pred_indices:
        box = pred_boxes[idx]
        label = pred_labels[idx]
        best_iou, best_j = 0.0, -1
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels, strict=False)):
            if j in matched_gt or gt_label != label:
                continue
            iou = bbox_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Per-class IoU averaged across classes present in GT (mIoU)."""
    pred_mask = np.asarray(pred_mask, dtype=np.int32)
    gt_mask = np.asarray(gt_mask, dtype=np.int32)
    classes = np.unique(gt_mask)
    classes = classes[classes >= 0]  # ignore background = -1 if used

    if len(classes) == 0:
        return {"miou": 1.0}

    ious = []
    for c in classes:
        pred_c = pred_mask == c
        gt_c = gt_mask == c
        intersection = float((pred_c & gt_c).sum())
        union = float((pred_c | gt_c).sum())
        ious.append(intersection / union if union > 0 else 1.0)

    return {"miou": float(np.mean(ious))}


def tracking_metrics(
    pred_tracks: Sequence,
    gt_tracks: Sequence,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """MOTA and IDF1 for multi-object tracking (single-frame / tracklet level).

    This is a simplified tracklet-level computation suitable for per-sample
    evaluation across a video clip.  A full MOTA/IDF1 computation requires
    frame-by-frame association across the full sequence and is typically run
    by the PhaseStratifiedRunner over an entire scene.
    """
    # Build frame-level sets from Tracklet objects
    # Each Tracklet has boxes: T x 4, so T = number of frames it covers.
    # We evaluate over each frame index present in GT.

    # Determine number of frames
    max_t = max((t.boxes.shape[0] for t in gt_tracks), default=0)
    if max_t == 0:
        return {"mota": 1.0, "idf1": 1.0}

    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0  # ID switches — approximated here

    # id assignment tracking: pred_track_id -> last matched gt_track_id
    pred_last_match: Dict[str, str] = {}

    for t in range(max_t):
        gt_boxes_t = []
        gt_ids_t = []
        for tr in gt_tracks:
            if t < tr.boxes.shape[0]:
                gt_boxes_t.append(tr.boxes[t])
                gt_ids_t.append(tr.track_id)

        pred_boxes_t = []
        pred_ids_t = []
        for tr in pred_tracks:
            if t < tr.boxes.shape[0]:
                pred_boxes_t.append(tr.boxes[t])
                pred_ids_t.append(tr.track_id)

        total_gt += len(gt_boxes_t)

        if len(pred_boxes_t) == 0:
            total_fn += len(gt_boxes_t)
            continue

        # Greedy matching
        matched_gt_idx: set[int] = set()
        for _p_idx, (p_box, p_id) in enumerate(zip(pred_boxes_t, pred_ids_t, strict=False)):
            best_iou, best_g = 0.0, -1
            for g_idx, g_box in enumerate(gt_boxes_t):
                if g_idx in matched_gt_idx:
                    continue
                iou = bbox_iou(np.array(p_box), np.array(g_box))
                if iou > best_iou:
                    best_iou, best_g = iou, g_idx
            if best_iou >= iou_threshold and best_g >= 0:
                matched_gt_idx.add(best_g)
                g_id = gt_ids_t[best_g]
                if p_id in pred_last_match and pred_last_match[p_id] != g_id:
                    total_idsw += 1
                pred_last_match[p_id] = g_id

        total_fn += len(gt_boxes_t) - len(matched_gt_idx)
        total_fp += len(pred_boxes_t) - len(matched_gt_idx)

    mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)
    mota = float(np.clip(mota, -1.0, 1.0))

    # IDF1: identity recall × precision harmonic mean (simplified: treat matched as IDTP)
    total_tp = total_gt - total_fn
    idf1 = (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1)
    idf1 = float(np.clip(idf1, 0.0, 1.0))

    return {
        "mota": mota,
        "idf1": idf1,
        "fp": float(total_fp),
        "fn": float(total_fn),
        "idsw": float(total_idsw),
    }


def pose_metrics(
    pred_rot: np.ndarray,
    pred_trans: np.ndarray,
    gt_rot: np.ndarray,
    gt_trans: np.ndarray,
) -> Dict[str, float]:
    """Rotation geodesic error (degrees) and translation L2 error (meters)."""
    pred_rot = np.asarray(pred_rot, dtype=np.float64)
    gt_rot = np.asarray(gt_rot, dtype=np.float64)

    # Handle quaternion input (4-vector)
    if pred_rot.shape == (4,):
        pred_rot = _quat_to_rotmat(pred_rot)
    if gt_rot.shape == (4,):
        gt_rot = _quat_to_rotmat(gt_rot)

    # Geodesic distance: arccos((trace(R^T R') - 1) / 2)
    relative = pred_rot.T @ gt_rot
    trace_val = float(np.trace(relative))
    cos_angle = (trace_val - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_angle)))

    trans_err_m = float(
        np.linalg.norm(
            np.asarray(pred_trans, dtype=np.float64) - np.asarray(gt_trans, dtype=np.float64)
        )
    )

    return {"rotation_error_deg": rot_err_deg, "translation_error_m": trans_err_m}


def grounding_metrics(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Visual grounding: top-1 IoU and accuracy at IoU threshold.

    Standard referring grounding protocol: a prediction is correct if the
    highest-scoring predicted box has IoU ≥ threshold with any GT box.
    """
    if len(pred_boxes) == 0:
        return {"grounding_iou": 0.0, "grounding_acc": 0.0}
    if len(gt_boxes) == 0:
        return {"grounding_iou": 1.0, "grounding_acc": 1.0}

    # Evaluate the first (highest-scoring) prediction against all GT boxes
    top_box = pred_boxes[0]
    best_iou = max(bbox_iou(top_box, gt_box) for gt_box in gt_boxes)
    acc = 1.0 if best_iou >= iou_threshold else 0.0

    return {"grounding_iou": float(best_iou), "grounding_acc": float(acc)}


def sparse_depth_metrics(
    pred_coords: np.ndarray,
    pred_depths: np.ndarray,
    gt_coords: np.ndarray,
    gt_depths: np.ndarray,
    radius: float = 2.0,
) -> Dict[str, float]:
    """AbsRel and RMSE at sparse GT pixel locations.

    We match predicted depths to GT coordinates by nearest-neighbor within
    `radius` pixels.
    """
    if len(gt_coords) == 0:
        return {"sparse_absrel": 0.0, "sparse_rmse": 0.0}

    abs_errors = []
    sq_errors = []
    for (gx, gy), gd in zip(gt_coords, gt_depths, strict=False):
        if len(pred_coords) == 0:
            abs_errors.append(gd)
            sq_errors.append(gd**2)
            continue
        dists = np.sqrt(((pred_coords - np.array([gx, gy])) ** 2).sum(axis=1))
        nearest = int(np.argmin(dists))
        if dists[nearest] <= radius:
            pd = pred_depths[nearest]
        else:
            pd = 0.0  # no prediction at this location
        abs_errors.append(abs(pd - gd) / max(gd, 1e-6))
        sq_errors.append((pd - gd) ** 2)

    return {
        "sparse_absrel": float(np.mean(abs_errors)),
        "sparse_rmse": float(np.sqrt(np.mean(sq_errors))),
    }


def nvs_metrics(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> Dict[str, float]:
    """PSNR and SSIM for novel-view synthesis."""
    pred = np.asarray(pred_rgb, dtype=np.float32)
    gt = np.asarray(gt_rgb, dtype=np.float32)

    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(10 * np.log10(255.0**2 / mse)) if mse > 0 else 100.0

    ssim = _ssim(pred, gt)

    return {"psnr": psnr, "ssim": ssim}


def keypoint_metrics(
    pred_p0: np.ndarray,
    pred_p1: np.ndarray,
    gt_p0: np.ndarray,
    gt_p1: np.ndarray,
    visibility: np.ndarray | None = None,
    px_threshold: float = 3.0,
) -> Dict[str, float]:
    """Keypoint matching accuracy: % correspondences within px_threshold pixels.

    We match each GT keypoint pair to the nearest predicted pair in image0
    and check if the predicted point in image1 is within threshold pixels of GT.
    """
    if len(gt_p0) == 0:
        return {"keypoint_acc": 1.0, "mean_match_error": 0.0}
    if len(pred_p0) == 0:
        return {"keypoint_acc": 0.0, "mean_match_error": float("inf")}

    mask = visibility if visibility is not None else np.ones(len(gt_p0), dtype=bool)
    gt_p0_v = gt_p0[mask]
    gt_p1_v = gt_p1[mask]

    if len(gt_p0_v) == 0:
        return {"keypoint_acc": 1.0, "mean_match_error": 0.0}

    errors = []
    for gp0, gp1 in zip(gt_p0_v, gt_p1_v, strict=False):
        dists0 = np.sqrt(((pred_p0 - gp0) ** 2).sum(axis=1))
        nearest = int(np.argmin(dists0))
        err = float(np.sqrt(((pred_p1[nearest] - gp1) ** 2).sum()))
        errors.append(err)

    errors_arr = np.array(errors)
    acc = float(np.mean(errors_arr < px_threshold))
    mean_err = float(np.mean(errors_arr))

    return {"keypoint_acc": acc, "mean_match_error": mean_err}


# ------------------------------------------------------------------ #
# Geometry helpers
# ------------------------------------------------------------------ #


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _ssim(
    pred: np.ndarray, gt: np.ndarray, k1: float = 0.01, k2: float = 0.03, L: float = 255.0
) -> float:
    """Simplified global SSIM (no sliding window) for NVS evaluation."""
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    mu_pred, mu_gt = pred.mean(), gt.mean()
    sigma_pred = pred.std()
    sigma_gt = gt.std()
    sigma_pred_gt = float(np.mean((pred - mu_pred) * (gt - mu_gt)))
    numerator = (2 * mu_pred * mu_gt + c1) * (2 * sigma_pred_gt + c2)
    denominator = (mu_pred**2 + mu_gt**2 + c1) * (sigma_pred**2 + sigma_gt**2 + c2)
    return float(numerator / denominator) if denominator > 0 else 1.0

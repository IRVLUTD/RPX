"""Paper-protocol metrics for RPX D3 multi-object tracking.

RPX publishes temporally consistent instance masks: a positive pixel value is
the same physical object ID throughout a ``(scene, phase)`` clip.  D3 models
are initialized from the first-frame mask.  For the MOTChallenge metric set we
derive tight boxes from those masks and delegate the sequence calculations to
the official TrackEval implementation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..exceptions import MetricError

PAPER_TRACKING_METRICS = ("hota", "deta", "assa", "idf1", "mota", "idsw")
MOT_IOU_THRESHOLD = 0.5


def _validate_mask_sequence(masks: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    if not masks:
        raise MetricError(f"{name} mask sequence is empty.")
    result: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for index, value in enumerate(masks):
        mask = np.asarray(value)
        if mask.ndim != 2:
            raise MetricError(f"{name}[{index}] must be 2-D, got shape {mask.shape}.")
        if not np.issubdtype(mask.dtype, np.integer):
            raise MetricError(f"{name}[{index}] must have an integer dtype, got {mask.dtype}.")
        if np.any(mask < 0):
            raise MetricError(f"{name}[{index}] contains negative instance IDs.")
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise MetricError(
                f"{name}[{index}] has shape {mask.shape}; expected {expected_shape}."
            )
        result.append(mask.astype(np.int32, copy=False))
    return result


def _boxes_and_ids(mask: np.ndarray, id_map: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    source_ids = np.unique(mask)
    source_ids = source_ids[source_ids > 0]
    boxes: list[list[float]] = []
    ids: list[int] = []
    for source_id in source_ids.tolist():
        ys, xs = np.nonzero(mask == source_id)
        if xs.size == 0:
            continue
        # Exclusive maxima follow the common xyxy convention and make a
        # one-pixel-wide object have positive area.
        boxes.append(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
        )
        ids.append(id_map[int(source_id)])
    return np.asarray(ids, dtype=np.int64), np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _box_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)
    left_top = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    right_bottom = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    intersection_wh = np.maximum(0.0, right_bottom - left_top)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    gt_area = np.prod(np.maximum(0.0, gt_boxes[:, 2:] - gt_boxes[:, :2]), axis=1)
    pred_area = np.prod(np.maximum(0.0, pred_boxes[:, 2:] - pred_boxes[:, :2]), axis=1)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _trackeval_data(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
) -> dict[str, Any]:
    gt = _validate_mask_sequence(gt_masks, "ground_truth")
    pred = _validate_mask_sequence(pred_masks, "prediction")
    if len(gt) != len(pred):
        raise MetricError(
            f"prediction has {len(pred)} frames; ground truth has {len(gt)} frames."
        )
    if pred[0].shape != gt[0].shape:
        raise MetricError(
            f"prediction shape {pred[0].shape} does not match GT shape {gt[0].shape}."
        )

    gt_source_ids = sorted({int(v) for mask in gt for v in np.unique(mask) if v > 0})
    pred_source_ids = sorted({int(v) for mask in pred for v in np.unique(mask) if v > 0})
    gt_id_map = {source_id: index for index, source_id in enumerate(gt_source_ids)}
    pred_id_map = {source_id: index for index, source_id in enumerate(pred_source_ids)}

    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarity_scores: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0
    for gt_mask, pred_mask in zip(gt, pred, strict=True):
        frame_gt_ids, gt_boxes = _boxes_and_ids(gt_mask, gt_id_map)
        frame_pred_ids, pred_boxes = _boxes_and_ids(pred_mask, pred_id_map)
        gt_ids.append(frame_gt_ids)
        tracker_ids.append(frame_pred_ids)
        similarity_scores.append(_box_iou_matrix(gt_boxes, pred_boxes))
        num_gt_dets += len(frame_gt_ids)
        num_tracker_dets += len(frame_pred_ids)

    return {
        "num_timesteps": len(gt),
        "num_gt_ids": len(gt_source_ids),
        "num_tracker_ids": len(pred_source_ids),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
    }


def paper_tracking_metrics(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
) -> dict[str, float]:
    """Compute HOTA, DetA, AssA, IDF1, MOTA and ID switches for one clip.

    TrackEval is intentionally a runtime dependency rather than a locally
    reimplemented approximation.  HOTA is the standard mean over association
    thresholds 0.05:0.05:0.95; CLEAR and Identity use the MOTChallenge IoU
    threshold of 0.5.
    """

    try:
        # TrackEval still references NumPy's removed aliases at the pinned
        # upstream revision. Restore those aliases locally before importing it;
        # this does not change any calculation.
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError as exc:
        raise MetricError(
            "Paper D3 metrics require the official TrackEval package.",
            hint="Use the RPX tracking Docker image; it pins TrackEval.",
        ) from exc

    data = _trackeval_data(gt_masks=gt_masks, pred_masks=pred_masks)
    clear_result = CLEAR({"THRESHOLD": MOT_IOU_THRESHOLD, "PRINT_CONFIG": False}).eval_sequence(
        data
    )
    identity_result = Identity(
        {"THRESHOLD": MOT_IOU_THRESHOLD, "PRINT_CONFIG": False}
    ).eval_sequence(data)
    hota_result = HOTA().eval_sequence(data)

    metrics = {
        "hota": float(np.mean(hota_result["HOTA"])),
        "deta": float(np.mean(hota_result["DetA"])),
        "assa": float(np.mean(hota_result["AssA"])),
        "idf1": float(identity_result["IDF1"]),
        "mota": float(clear_result["MOTA"]),
        "idsw": float(clear_result["IDSW"]),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise MetricError(f"TrackEval returned non-finite metrics: {metrics}")
    return metrics

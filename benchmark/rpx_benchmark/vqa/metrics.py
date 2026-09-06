"""Benchmark metrics and smoke-gate aggregation for RPX VQA."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ..exceptions import MetricError
from .contract import ATTRIBUTE_TYPES, BBOX_TYPES, BINARY_TYPES, VQASample
from .outputs import ParsedOutput, normalize_label


def bbox_iou(
    predicted: tuple[float, float, float, float], ground_truth: tuple[int, int, int, int]
) -> float:
    """IoU for the dataset's inclusive pixel xyxy convention."""
    px0, py0, px1, py1 = predicted
    gx0, gy0, gx1, gy1 = ground_truth
    ix0, iy0, ix1, iy1 = max(px0, gx0), max(py0, gy0), min(px1, gx1), min(py1, gy1)
    intersection = max(0.0, ix1 - ix0 + 1) * max(0.0, iy1 - iy0 + 1)
    pred_area = max(0.0, px1 - px0 + 1) * max(0.0, py1 - py0 + 1)
    gt_area = (gx1 - gx0 + 1) * (gy1 - gy0 + 1)
    union = pred_area + gt_area - intersection
    return intersection / union if union else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_predictions(
    pairs: Iterable[tuple[VQASample, ParsedOutput]], iou_threshold: float = 0.5
) -> dict[str, Any]:
    rows = list(pairs)
    if not rows:
        raise MetricError("no VQA predictions to score", hint="provide one output per request")
    valid = [float(parsed.valid) for _, parsed in rows]
    by_type: dict[str, list[float]] = defaultdict(list)
    binary: list[float] = []
    attribute: list[float] = []
    ious: list[float] = []
    bbox_hits: list[float] = []
    bbox_labels: list[float] = []
    for sample, parsed in rows:
        score = 0.0
        if parsed.valid and sample.question_type in BINARY_TYPES:
            score = float(parsed.label == sample.answer)
            binary.append(score)
        elif parsed.valid and sample.question_type in ATTRIBUTE_TYPES:
            score = float(parsed.label == normalize_label(sample.answer))
            attribute.append(score)
        elif sample.question_type in BBOX_TYPES:
            assert sample.answer_bbox is not None
            iou = 0.0
            if parsed.valid and parsed.bbox is not None:
                iou = bbox_iou(parsed.bbox, sample.answer_bbox)
            ious.append(iou)
            score = float(iou >= iou_threshold)
            bbox_hits.append(score)
            if parsed.label is not None:
                bbox_labels.append(float(parsed.label == normalize_label(sample.answer)))
        by_type[sample.question_type].append(score)
    return {
        "count": len(rows),
        "parse_rate": _mean(valid),
        "binary_accuracy": _mean(binary),
        "attribute_exact_match": _mean(attribute),
        "bbox_accuracy_at_0_5": _mean(bbox_hits),
        "bbox_mean_iou": _mean(ious),
        "bbox_label_accuracy": _mean(bbox_labels),
        "per_type": {key: _mean(value) for key, value in sorted(by_type.items())},
    }

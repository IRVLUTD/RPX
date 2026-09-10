"""Benchmark metrics and smoke-gate aggregation for RPX VQA."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ..exceptions import MetricError
from .contract import ATTRIBUTE_TYPES, BBOX_TYPES, BINARY_TYPES, VQASample
from .outputs import ParsedOutput, normalize_label

COCO_IOU_THRESHOLDS = tuple(value / 100 for value in range(50, 100, 5))


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


def label_token_f1(predicted: str | None, expected: str) -> float:
    """Transparent lexical overlap for the single-reference label contract.

    This complements, but never replaces, normalized exact match.  It is not
    called semantic accuracy: RPX has one reference label per row rather than
    the multiple human answers needed by the conventional VQA soft score.
    """
    predicted_tokens = normalize_label(predicted or "").split()
    expected_tokens = normalize_label(expected).split()
    if not predicted_tokens or not expected_tokens:
        return 0.0
    predicted_counts: dict[str, int] = defaultdict(int)
    expected_counts: dict[str, int] = defaultdict(int)
    for token in predicted_tokens:
        predicted_counts[token] += 1
    for token in expected_tokens:
        expected_counts[token] += 1
    overlap = sum(
        min(count, expected_counts.get(token, 0))
        for token, count in predicted_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


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
    bbox_label_token_f1: list[float] = []
    bbox_joint_hits: list[float] = []
    bbox_hits_by_threshold: dict[float, list[float]] = {
        threshold: [] for threshold in (0.25, 0.5, 0.75, *COCO_IOU_THRESHOLDS)
    }
    detailed_by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
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
            label_exact = float(parsed.label == normalize_label(sample.answer))
            label_f1 = label_token_f1(parsed.label, sample.answer)
            # Missing/unparseable labels count as zero; excluding them would
            # inflate textual correctness precisely when protocol compliance
            # is poor.
            bbox_labels.append(label_exact)
            bbox_label_token_f1.append(label_f1)
            bbox_joint_hits.append(float(label_exact == 1.0 and iou >= iou_threshold))
            for threshold, values in bbox_hits_by_threshold.items():
                values.append(float(iou >= threshold))
            detailed_by_type[sample.question_type].append(
                {
                    "valid": float(parsed.valid),
                    "iou": iou,
                    "label_exact": label_exact,
                    "label_token_f1": label_f1,
                }
            )
        by_type[sample.question_type].append(score)
    per_type_detailed = {}
    for question_type, values in sorted(detailed_by_type.items()):
        per_type_detailed[question_type] = {
            "count": len(values),
            "parse_rate": _mean([value["valid"] for value in values]),
            "bbox_mean_iou": _mean([value["iou"] for value in values]),
            "bbox_accuracy_at_0_25": _mean(
                [float(value["iou"] >= 0.25) for value in values]
            ),
            "bbox_accuracy_at_0_5": _mean(
                [float(value["iou"] >= 0.5) for value in values]
            ),
            "bbox_accuracy_at_0_75": _mean(
                [float(value["iou"] >= 0.75) for value in values]
            ),
            "label_normalized_exact_match": _mean(
                [value["label_exact"] for value in values]
            ),
            "label_token_f1": _mean([value["label_token_f1"] for value in values]),
        }
    return {
        "count": len(rows),
        "primary_metric": "bbox_accuracy_at_0_5",
        "bbox_scoring_uses_generated_label": False,
        "parse_rate": _mean(valid),
        "binary_accuracy": _mean(binary),
        "attribute_exact_match": _mean(attribute),
        "bbox_accuracy_at_0_5": _mean(bbox_hits),
        "bbox_accuracy_at_0_25": _mean(bbox_hits_by_threshold[0.25]),
        "bbox_accuracy_at_0_75": _mean(bbox_hits_by_threshold[0.75]),
        "bbox_mean_iou": _mean(ious),
        "bbox_mean_accuracy_50_95": _mean(
            [_mean(bbox_hits_by_threshold[threshold]) for threshold in COCO_IOU_THRESHOLDS]
        ),
        # Retain the old key for report compatibility, but make its exact
        # single-reference meaning explicit in the new key.
        "bbox_label_accuracy": _mean(bbox_labels),
        "bbox_label_normalized_exact_match": _mean(bbox_labels),
        "bbox_label_token_f1": _mean(bbox_label_token_f1),
        "bbox_joint_label_exact_and_iou_at_0_5": _mean(bbox_joint_hits),
        "per_type": {key: _mean(value) for key, value in sorted(by_type.items())},
        "per_type_detailed": per_type_detailed,
        "metric_notes": {
            "bbox_mean_accuracy_50_95": (
                "Mean single-box success rate over IoU thresholds 0.50:0.05:0.95; "
                "not detection mAP because predictions have no confidence scores."
            ),
            "bbox_label_normalized_exact_match": (
                "Informational only: one-reference normalized exact match; bbox scoring "
                "does not depend on the generated label."
            ),
            "bbox_label_token_f1": (
                "Informational only: lexical token overlap; bbox scoring does not depend "
                "on the generated label."
            ),
            "bbox_joint_label_exact_and_iou_at_0_5": (
                "Informational diagnostic only; the benchmark's bbox result is "
                "bbox_accuracy_at_0_5."
            ),
        },
    }

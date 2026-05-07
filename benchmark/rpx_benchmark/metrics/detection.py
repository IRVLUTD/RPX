"""Object detection + open-vocab detection metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import DetectionGroundTruth, DetectionPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.OBJECT_DETECTION)
@register_metric(TaskType.OPEN_VOCAB_DETECTION)
class DetectionMetrics(MetricCalculator):
    """Precision / recall / F1 at a single IoU threshold.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU (0.0–1.0) for a prediction to count as a true
        positive. Defaults to 0.5, the COCO baseline.

    Notes
    -----
    Greedy matching by descending prediction score. A GT box is
    consumed after the first predicted box matches it, so subsequent
    predictions for the same object register as false positives
    (standard VOC/COCO evaluation rule).
    """

    name = "detection_metrics"

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def compute(
        self,
        prediction: DetectionPrediction,
        ground_truth: DetectionGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, DetectionPrediction):
            raise MetricError(
                f"DetectionMetrics expected DetectionPrediction, got {type(prediction).__name__}",
            )
        if not isinstance(ground_truth, DetectionGroundTruth):
            raise MetricError(
                f"DetectionMetrics expected DetectionGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )

        from ..evaluators import detection_metrics

        return detection_metrics(
            pred_boxes=prediction.boxes,
            pred_scores=prediction.scores,
            pred_labels=prediction.labels,
            gt_boxes=ground_truth.boxes,
            gt_labels=ground_truth.labels,
            iou_threshold=self.iou_threshold,
        )

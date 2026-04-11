"""Visual grounding metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import TaskType, VisualGroundingGroundTruth, VisualGroundingPrediction
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.VISUAL_GROUNDING)
class GroundingIoU(MetricCalculator):
    """Top-1 IoU and accuracy at a threshold for a single referred object.

    Parameters
    ----------
    iou_threshold : float
        IoU threshold for the ``grounding_acc`` indicator. Default
        0.5, matching standard referring expression comprehension
        protocols.
    """

    name = "grounding_iou"

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def compute(
        self,
        prediction: VisualGroundingPrediction,
        ground_truth: VisualGroundingGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, VisualGroundingPrediction):
            raise MetricError(
                f"GroundingIoU expected VisualGroundingPrediction, got "
                f"{type(prediction).__name__}",
            )
        if not isinstance(ground_truth, VisualGroundingGroundTruth):
            raise MetricError(
                f"GroundingIoU expected VisualGroundingGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )
        from ..evaluators import grounding_metrics
        return grounding_metrics(
            pred_boxes=prediction.boxes,
            gt_boxes=ground_truth.boxes,
            iou_threshold=self.iou_threshold,
        )

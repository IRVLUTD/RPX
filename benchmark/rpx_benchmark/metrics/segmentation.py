"""Instance/semantic segmentation metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import SegmentationGroundTruth, SegmentationPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.OBJECT_SEGMENTATION)
class SegmentationMIoU(MetricCalculator):
    """Mean Intersection-over-Union (mIoU) across GT classes.

    Notes
    -----
    Averaged over the set of class ids present in the ground-truth
    mask (background class id ``-1`` is ignored). Classes in the
    prediction that do not appear in GT do not contribute — they only
    reduce the IoU of other classes they overlap with.
    """

    name = "segmentation_miou"

    def compute(
        self,
        prediction: SegmentationPrediction,
        ground_truth: SegmentationGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, SegmentationPrediction):
            raise MetricError(
                f"SegmentationMIoU expected SegmentationPrediction, got "
                f"{type(prediction).__name__}",
            )
        if not isinstance(ground_truth, SegmentationGroundTruth):
            raise MetricError(
                f"SegmentationMIoU expected SegmentationGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )

        from ..evaluators import segmentation_metrics

        return segmentation_metrics(prediction.mask, ground_truth.mask)

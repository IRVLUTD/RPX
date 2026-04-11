"""Keypoint correspondence metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import (
    KeypointCorrespondenceGroundTruth,
    KeypointCorrespondencePrediction,
    TaskType,
)
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.KEYPOINT_MATCHING)
class KeypointAccuracy(MetricCalculator):
    """% of predicted matches within a pixel threshold + mean error.

    Parameters
    ----------
    px_threshold : float
        Pixel radius in frame B at which a predicted correspondence is
        counted as correct. Default 3 px matches the standard
        ScanNet/InteriorNet evaluation protocol.
    """

    name = "keypoint_accuracy"

    def __init__(self, px_threshold: float = 3.0) -> None:
        self.px_threshold = px_threshold

    def compute(
        self,
        prediction: KeypointCorrespondencePrediction,
        ground_truth: KeypointCorrespondenceGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, KeypointCorrespondencePrediction):
            raise MetricError(
                f"KeypointAccuracy expected "
                f"KeypointCorrespondencePrediction, got "
                f"{type(prediction).__name__}",
            )
        if not isinstance(ground_truth, KeypointCorrespondenceGroundTruth):
            raise MetricError(
                f"KeypointAccuracy expected "
                f"KeypointCorrespondenceGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )
        from ..evaluators import keypoint_metrics
        return keypoint_metrics(
            pred_p0=prediction.points0,
            pred_p1=prediction.points1,
            gt_p0=ground_truth.points0,
            gt_p1=ground_truth.points1,
            visibility=ground_truth.visibility,
            px_threshold=self.px_threshold,
        )

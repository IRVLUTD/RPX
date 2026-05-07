"""Multi-object tracking metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import TaskType, TrackletGroundTruth, TrackletPrediction
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.OBJECT_TRACKING)
class TrackletMetrics(MetricCalculator):
    """Simplified MOTA + IDF1 over a single tracklet sample.

    Parameters
    ----------
    iou_threshold : float
        IoU required to count a predicted detection as a hit for a
        given GT box. Default 0.5.

    Notes
    -----
    This is the per-sample version called by the runner. A full MOTA
    computation (with cross-scene identity switches) lives in
    :mod:`rpx_benchmark.deployment` and runs on the whole dataset.
    """

    name = "tracklet_metrics"

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def compute(
        self,
        prediction: TrackletPrediction,
        ground_truth: TrackletGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, TrackletPrediction):
            raise MetricError(
                f"TrackletMetrics expected TrackletPrediction, got {type(prediction).__name__}",
            )
        if not isinstance(ground_truth, TrackletGroundTruth):
            raise MetricError(
                f"TrackletMetrics expected TrackletGroundTruth, got {type(ground_truth).__name__}",
            )
        from ..evaluators import tracking_metrics

        return tracking_metrics(
            prediction.tracks,
            ground_truth.tracks,
            iou_threshold=self.iou_threshold,
        )

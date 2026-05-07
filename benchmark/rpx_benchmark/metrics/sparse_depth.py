"""Sparse depth metric calculators (RGB + sparse GT depth points)."""

from __future__ import annotations

from typing import Dict

from ..api import SparseDepthGroundTruth, SparseDepthPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.SPARSE_DEPTH)
class SparseDepthError(MetricCalculator):
    """AbsRel + RMSE computed only at the provided sparse GT locations.

    Parameters
    ----------
    radius : float
        Maximum pixel distance between a prediction point and the
        matched GT point. Predictions outside this radius contribute a
        full-magnitude error, which penalises models that fail to
        localise the sparse points.
    """

    name = "sparse_depth_error"

    def __init__(self, radius: float = 2.0) -> None:
        self.radius = radius

    def compute(
        self,
        prediction: SparseDepthPrediction,
        ground_truth: SparseDepthGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, SparseDepthPrediction):
            raise MetricError(
                f"SparseDepthError expected SparseDepthPrediction, got {type(prediction).__name__}",
            )
        if not isinstance(ground_truth, SparseDepthGroundTruth):
            raise MetricError(
                f"SparseDepthError expected SparseDepthGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )
        from ..evaluators import sparse_depth_metrics

        return sparse_depth_metrics(
            pred_coords=prediction.coordinates,
            pred_depths=prediction.depths,
            gt_coords=ground_truth.coordinates,
            gt_depths=ground_truth.depths,
            radius=self.radius,
        )

"""Relative camera pose metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import RelativePoseGroundTruth, RelativePosePrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.RELATIVE_CAMERA_POSE)
class RelativePoseError(MetricCalculator):
    """Rotation geodesic (degrees) + translation L2 (metres).

    Notes
    -----
    Rotations are compared in SO(3) via the geodesic distance
    ``arccos((trace(R_pred^T R_gt) - 1) / 2)``. Quaternion inputs (4-
    vectors) are accepted and converted to rotation matrices.
    Translations are compared in metres with straight L2.
    """

    name = "relative_pose_error"

    def compute(
        self,
        prediction: RelativePosePrediction,
        ground_truth: RelativePoseGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, RelativePosePrediction):
            raise MetricError(
                f"RelativePoseError expected RelativePosePrediction, got "
                f"{type(prediction).__name__}",
            )
        if not isinstance(ground_truth, RelativePoseGroundTruth):
            raise MetricError(
                f"RelativePoseError expected RelativePoseGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )
        from ..evaluators import pose_metrics
        return pose_metrics(
            prediction.rotation, prediction.translation,
            ground_truth.rotation, ground_truth.translation,
        )

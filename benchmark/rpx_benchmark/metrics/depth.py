"""Monocular absolute depth metric calculators.

Registers :class:`DepthErrorMetrics` which computes AbsRel, RMSE, and
three thresholded accuracy ratios (δ<1.25, δ<1.25², δ<1.25³) against
the subset of pixels where the ground-truth depth map is valid
(``gt > 0``). This matches the paper's eval protocol: we mask out
D435-invalid pixels so models are not penalised where the sensor
itself produced no measurement.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..api import DepthGroundTruth, DepthPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.MONOCULAR_DEPTH)
class DepthErrorMetrics(MetricCalculator):
    """AbsRel / RMSE / δ-accuracy for monocular metric depth.

    Notes
    -----
    All metrics are computed in metres. Neither raw logits nor any
    kind of scale alignment is applied — we deliberately measure the
    model's *absolute* metric accuracy, which is the property that
    matters for robot policies that consume depth directly.

    For empty validity masks (fully invalid GT frames) we return the
    safe no-op sentinel ``{"absrel": 0, "rmse": 0, "delta1": 1, ...}``
    so a single bad frame does not break aggregate means. In practice
    the dataset pipeline should filter these out upstream.

    Metric definitions
    ------------------

    .. math::

        \\text{AbsRel} &= \\frac{1}{N} \\sum_i \\frac{|\\hat d_i - d_i|}{d_i}

        \\text{RMSE}   &= \\sqrt{\\frac{1}{N}\\sum_i(\\hat d_i - d_i)^2}

        \\delta_k      &= \\frac{1}{N}\\,\\#\\Big\\{i : \\max\\!\\big(\\tfrac{\\hat d_i}{d_i},
                          \\tfrac{d_i}{\\hat d_i}\\big) < 1.25^{\\,k}\\Big\\}
    """

    name = "depth_error_metrics"

    def compute(
        self,
        prediction: DepthPrediction,
        ground_truth: DepthGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, DepthPrediction):
            raise MetricError(
                f"DepthErrorMetrics expected DepthPrediction, got "
                f"{type(prediction).__name__}",
            )
        if not isinstance(ground_truth, DepthGroundTruth):
            raise MetricError(
                f"DepthErrorMetrics expected DepthGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )

        pred = np.asarray(prediction.depth_map, dtype=np.float32)
        gt = np.asarray(ground_truth.depth_map, dtype=np.float32)

        if pred.shape != gt.shape:
            raise MetricError(
                f"Depth prediction shape {pred.shape} does not match "
                f"ground-truth shape {gt.shape}",
                hint=(
                    "The adapter's OutputAdapter should resize predictions "
                    "back to the sample RGB resolution before returning."
                ),
            )

        valid = gt > 0
        if not valid.any():
            return {"rmse": 0.0, "absrel": 0.0,
                    "delta1": 1.0, "delta2": 1.0, "delta3": 1.0}

        pred_v = pred[valid]
        gt_v = gt[valid]

        # Guard against pathological predictions (0 or negative) that
        # would blow up the ratio-based thresholds.
        pred_v = np.clip(pred_v, a_min=1e-6, a_max=None)

        rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
        absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))

        thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
        delta1 = float(np.mean(thresh < 1.25))
        delta2 = float(np.mean(thresh < 1.25 ** 2))
        delta3 = float(np.mean(thresh < 1.25 ** 3))

        return {
            "rmse": rmse,
            "absrel": absrel,
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
        }

"""Novel view synthesis metric calculators."""

from __future__ import annotations

from typing import Dict

from ..api import NovelViewSynthesisGroundTruth, NovelViewSynthesisPrediction, TaskType
from ..exceptions import MetricError
from .registry import MetricCalculator, register_metric


@register_metric(TaskType.NOVEL_VIEW_SYNTHESIS)
class NVSQuality(MetricCalculator):
    """PSNR and simplified global SSIM for an RGB novel-view prediction.

    Notes
    -----
    The SSIM implementation is a deliberately simple global-statistics
    variant (no sliding window) so it runs without OpenCV / skimage.
    For publication we recommend re-computing SSIM offline with a full
    sliding-window implementation against the per-sample arrays
    captured in ``result.per_sample``.
    """

    name = "nvs_quality"

    def compute(
        self,
        prediction: NovelViewSynthesisPrediction,
        ground_truth: NovelViewSynthesisGroundTruth,
    ) -> Dict[str, float]:
        if not isinstance(prediction, NovelViewSynthesisPrediction):
            raise MetricError(
                f"NVSQuality expected NovelViewSynthesisPrediction, got "
                f"{type(prediction).__name__}",
            )
        if not isinstance(ground_truth, NovelViewSynthesisGroundTruth):
            raise MetricError(
                f"NVSQuality expected NovelViewSynthesisGroundTruth, got "
                f"{type(ground_truth).__name__}",
            )
        from ..evaluators import nvs_metrics

        return nvs_metrics(prediction.rgb, ground_truth.rgb)

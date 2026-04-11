"""Tests for the pluggable metric registry."""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.api import (
    DepthGroundTruth,
    DepthPrediction,
    SegmentationGroundTruth,
    SegmentationPrediction,
    TaskType,
)
from rpx_benchmark.exceptions import MetricError
from rpx_benchmark.metrics import (
    MetricCalculator,
    MetricSuite,
    available_metrics,
    compute_metrics,
    register_metric,
    unregister_metric,
)


def test_registry_populated_for_every_task():
    ams = available_metrics()
    # All 10 tasks ship with at least one calculator
    for t in TaskType:
        assert t in ams, f"{t.value} has no registered calculator"
        assert ams[t], f"{t.value} has an empty calculator list"


def test_compute_metrics_depth():
    pred = DepthPrediction(depth_map=np.full((8, 8), 2.0, dtype=np.float32))
    gt = DepthGroundTruth(depth_map=np.full((8, 8), 2.0, dtype=np.float32))
    out = compute_metrics(TaskType.MONOCULAR_DEPTH, pred, gt)
    assert out["absrel"] == 0.0
    assert out["rmse"] == 0.0
    assert out["delta1"] == 1.0
    assert out["delta2"] == 1.0
    assert out["delta3"] == 1.0


def test_compute_metrics_wrong_type_raises():
    pred = DepthPrediction(depth_map=np.ones((4, 4), dtype=np.float32))
    wrong_gt = "not a DepthGroundTruth"
    with pytest.raises(MetricError, match="DepthGroundTruth"):
        compute_metrics(TaskType.MONOCULAR_DEPTH, pred, wrong_gt)


def test_depth_shape_mismatch_raises_metric_error():
    pred = DepthPrediction(depth_map=np.ones((4, 4), dtype=np.float32))
    gt = DepthGroundTruth(depth_map=np.ones((8, 8), dtype=np.float32))
    with pytest.raises(MetricError, match="shape"):
        compute_metrics(TaskType.MONOCULAR_DEPTH, pred, gt)


def test_register_and_unregister_custom_metric():
    @register_metric(TaskType.MONOCULAR_DEPTH)
    class AlwaysFortyTwo(MetricCalculator):
        name = "always_forty_two"
        def compute(self, prediction, ground_truth):
            return {"always_forty_two": 42.0}

    try:
        pred = DepthPrediction(depth_map=np.ones((4, 4), dtype=np.float32))
        gt = DepthGroundTruth(depth_map=np.ones((4, 4), dtype=np.float32))
        out = compute_metrics(TaskType.MONOCULAR_DEPTH, pred, gt)
        assert out.get("always_forty_two") == 42.0
        # Original depth metrics should still be present
        assert "absrel" in out
        assert "rmse" in out
    finally:
        removed = unregister_metric(TaskType.MONOCULAR_DEPTH, "always_forty_two")
        assert removed


def test_register_metric_rejects_non_calculator():
    with pytest.raises(MetricError, match="MetricCalculator"):
        register_metric(TaskType.MONOCULAR_DEPTH)(dict)


def test_register_metric_rejects_non_task_type():
    with pytest.raises(MetricError, match="TaskType"):
        register_metric("not a task type")  # type: ignore[arg-type]


def test_metric_suite_facade_still_works():
    """Legacy code path: MetricSuite.for_task(task).evaluate(pred, gt)."""
    suite = MetricSuite.for_task(TaskType.MONOCULAR_DEPTH)
    pred = DepthPrediction(depth_map=np.full((4, 4), 2.0, dtype=np.float32))
    gt = DepthGroundTruth(depth_map=np.full((4, 4), 2.0, dtype=np.float32))
    metrics = suite.evaluate(pred, gt)
    assert "absrel" in metrics
    assert metrics["absrel"] == 0.0


def test_segmentation_metric_smoke():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:6, 2:6] = 1
    pred = SegmentationPrediction(mask=mask.copy())
    gt = SegmentationGroundTruth(mask=mask.copy())
    out = compute_metrics(TaskType.OBJECT_SEGMENTATION, pred, gt)
    assert out["miou"] == 1.0

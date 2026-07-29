from __future__ import annotations

import numpy as np

from rpx_benchmark.metrics.tracking_paper import paper_tracking_metrics


def _mask(instance_id: int) -> np.ndarray:
    value = np.zeros((12, 16), dtype=np.int32)
    value[2:8, 4:11] = instance_id
    return value


def test_paper_tracking_metrics_perfect_sequence() -> None:
    ground_truth = [_mask(7), _mask(7), _mask(7)]
    result = paper_tracking_metrics(ground_truth, ground_truth)
    assert result["mota"] == 1.0
    assert result["idf1"] == 1.0
    assert result["hota"] == 1.0
    assert result["idsw"] == 0.0


def test_paper_tracking_metrics_counts_identity_switch() -> None:
    ground_truth = [_mask(7), _mask(7), _mask(7)]
    prediction = [_mask(11), _mask(12), _mask(12)]
    result = paper_tracking_metrics(prediction, ground_truth)
    assert result["idsw"] == 1.0
    assert result["mota"] < 1.0
    assert result["idf1"] < 1.0
    assert result["hota"] < 1.0

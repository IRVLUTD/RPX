"""Acceptance tests for the strict RPX D1-F metric and resume protocol."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.adapters.batched_depth import BatchedDepthBenchmarkModel
from rpx_benchmark.api import DepthGroundTruth, DepthPrediction, Sample
from rpx_benchmark.exceptions import MetricError
from rpx_benchmark.metrics.depth_paper import (
    EXPECTED_HW,
    FAST_PAPER_METRIC_KEYS,
    PAPER_METRIC_KEYS,
    FastPaperDepthMetricSuite,
    PaperDepthMetricSuite,
    compute_d1_paper_metrics,
)


def _depth_pair(
    points: dict[tuple[int, int], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros(EXPECTED_HW, dtype=np.float32)
    pred = np.zeros(EXPECTED_HW, dtype=np.float32)
    for (y, x), (gt_value, pred_value) in points.items():
        gt[y, x] = gt_value
        pred[y, x] = pred_value
    return gt, pred


def _sample(frame: str = "000001") -> Sample:
    return Sample(
        id=f"scene_001__0__{frame}",
        rgb=np.zeros((*EXPECTED_HW, 3), dtype=np.uint8),
        ground_truth=None,
        metadata={"scene_id": "scene_001", "phase_idx": "0", "frame": frame},
    )


class _CountingAdapter:
    native_alignment = "none"
    native_precision = "fp16"

    def __init__(self) -> None:
        self.calls = 0
        self.frames = 0

    def __call__(self, rgbs):
        self.calls += 1
        self.frames += len(rgbs)
        base = np.linspace(1.0, 2.0, EXPECTED_HW[1], dtype=np.float32)
        depth = np.broadcast_to(base, EXPECTED_HW).copy()
        return [depth.copy() for _ in rgbs]


def test_paper_metrics_perfect_prediction() -> None:
    gt, pred = _depth_pair({(10, 10): (1.0, 1.0), (20, 20): (2.0, 2.0)})
    result = compute_d1_paper_metrics(pred, gt)
    assert set(result) == set(PAPER_METRIC_KEYS)
    assert result == {
        "absrel": 0.0,
        "rmse": 0.0,
        "silog": 0.0,
        "delta1": 1.0,
        "irmse": 0.0,
        "fscore_5cm": 1.0,
    }


def test_fast_paper_suite_defers_only_fscore() -> None:
    gt = np.ones(EXPECTED_HW, dtype=np.float32)
    pred = np.full(EXPECTED_HW, 1.1, dtype=np.float32)
    result = FastPaperDepthMetricSuite().evaluate(
        DepthPrediction(pred),
        DepthGroundTruth(gt),
    )
    expected = compute_d1_paper_metrics(pred, gt, include_fscore=False)
    assert tuple(result) == FAST_PAPER_METRIC_KEYS
    assert result == expected
    assert "fscore_5cm" not in result


def test_paper_mask_is_strict_and_constant_offset_is_known() -> None:
    gt, pred = _depth_pair(
        {
            (1, 1): (0.3, 100.0),
            (2, 2): (5.0, 100.0),
            (3, 3): (1.0, 1.1),
        }
    )
    result = compute_d1_paper_metrics(pred, gt)
    assert result["absrel"] == pytest.approx(0.1)
    assert result["rmse"] == pytest.approx(0.1)
    assert result["delta1"] == 1.0
    assert result["fscore_5cm"] == 0.0


def test_paper_fscore_has_known_precision_and_recall() -> None:
    gt, pred = _depth_pair(
        {
            (100, 100): (1.0, 1.0),
            (300, 500): (1.0, 2.0),
        }
    )
    assert compute_d1_paper_metrics(pred, gt)["fscore_5cm"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "pred",
    [
        np.full(EXPECTED_HW, np.nan, dtype=np.float32),
        np.full(EXPECTED_HW, -1.0, dtype=np.float32),
    ],
)
def test_paper_metrics_reject_invalid_predictions(pred: np.ndarray) -> None:
    gt = np.ones(EXPECTED_HW, dtype=np.float32)
    with pytest.raises(MetricError, match="non-finite or non-positive"):
        compute_d1_paper_metrics(pred, gt)


def test_paper_suite_rejects_noncanonical_resolution() -> None:
    suite = PaperDepthMetricSuite()
    with pytest.raises(MetricError, match="requires depth shape"):
        suite.evaluate(
            DepthPrediction(np.ones((10, 10), dtype=np.float32)),
            DepthGroundTruth(np.ones((10, 10), dtype=np.float32)),
        )


def test_resume_valid_cache_performs_zero_forwards(tmp_path: Path) -> None:
    adapter = _CountingAdapter()
    model = BatchedDepthBenchmarkModel(
        adapter,
        name="test",
        save_dir=tmp_path,
        resume_predictions=True,
    )
    sample = _sample()
    first = model.predict([sample])
    model.persist_predictions([sample], first)
    assert adapter.frames == 1

    second = model.predict([sample])
    model.persist_predictions([sample], second)
    assert adapter.frames == 1
    assert model.last_cache_hits == [True]
    assert model.resume_stats == {
        "cache_hits": 1,
        "inferred_new": 1,
        "invalid_recomputed": 0,
    }


@pytest.mark.parametrize(
    "failure",
    ["corrupt", "crc", "key", "shape", "dtype", "nan", "negative", "constant"],
)
def test_resume_recomputes_every_invalid_npz(tmp_path: Path, failure: str) -> None:
    adapter = _CountingAdapter()
    model = BatchedDepthBenchmarkModel(
        adapter,
        name="test",
        save_dir=tmp_path,
        resume_predictions=True,
    )
    sample = _sample()
    path = model._prediction_path(sample)
    assert path is not None
    path.parent.mkdir(parents=True)
    good = np.broadcast_to(
        np.linspace(1.0, 2.0, EXPECTED_HW[1], dtype=np.float32), EXPECTED_HW
    ).copy()
    if failure == "corrupt":
        path.write_bytes(b"not an npz")
    elif failure == "crc":
        np.savez_compressed(path, depth=good)
        damaged = bytearray(path.read_bytes())
        damaged[len(damaged) // 2] ^= 0xFF
        path.write_bytes(damaged)
    elif failure == "key":
        np.savez_compressed(path, wrong=good)
    elif failure == "shape":
        np.savez_compressed(path, depth=good[:10])
    elif failure == "dtype":
        np.savez_compressed(path, depth=good.astype(np.float16))
    elif failure == "nan":
        good[0, 0] = np.nan
        np.savez_compressed(path, depth=good)
    elif failure == "negative":
        good[0, 0] = -1.0
        np.savez_compressed(path, depth=good)
    else:
        np.savez_compressed(path, depth=np.ones(EXPECTED_HW, dtype=np.float32))

    predictions = model.predict([sample])
    model.persist_predictions([sample], predictions)
    assert adapter.frames == 1
    assert model.resume_stats["invalid_recomputed"] == 1
    with np.load(path, allow_pickle=False) as payload:
        assert payload.files == ["depth"]
        assert payload["depth"].dtype == np.float32


def test_resume_preserves_mixed_manifest_order(tmp_path: Path) -> None:
    adapter = _CountingAdapter()
    model = BatchedDepthBenchmarkModel(
        adapter,
        name="test",
        save_dir=tmp_path,
        resume_predictions=True,
    )
    samples = [_sample("000001"), _sample("000002"), _sample("000003")]
    initial = model.predict([samples[1]])
    model.persist_predictions([samples[1]], initial)
    adapter.frames = 0
    predictions = model.predict(samples)
    assert adapter.frames == 2
    assert model.last_cache_hits == [False, True, False]
    assert [prediction.depth_map.shape for prediction in predictions] == [EXPECTED_HW] * 3


def test_atomic_failure_keeps_previous_prediction(tmp_path: Path, monkeypatch) -> None:
    adapter = _CountingAdapter()
    model = BatchedDepthBenchmarkModel(adapter, name="test", save_dir=tmp_path)
    sample = _sample()
    path = model._prediction_path(sample)
    assert path is not None
    original = np.broadcast_to(
        np.linspace(1.0, 2.0, EXPECTED_HW[1], dtype=np.float32), EXPECTED_HW
    ).copy()
    model._maybe_save(sample, original)
    before = path.read_bytes()

    def _fail_replace(_source, _destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        model._maybe_save(sample, original + 1.0)
    assert path.read_bytes() == before
    assert list(path.parent.glob("*.part")) == []

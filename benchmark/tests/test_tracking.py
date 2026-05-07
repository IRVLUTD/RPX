"""Tests for object-tracking parity: numpy adapter, metric, pipeline.

Metric golden values are frozen in code (no external fixtures) so
regressions in :func:`rpx_benchmark.evaluators.tracking_metrics`
surface on the next test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rpx_benchmark.adapters import make_numpy_tracking_model
from rpx_benchmark.api import TaskType, Tracklet, TrackletGroundTruth, TrackletPrediction
from rpx_benchmark.evaluators import tracking_metrics
from rpx_benchmark.metrics.registry import compute_metrics

# --------------------------------------------------------------------------- #
# Numpy adapter
# --------------------------------------------------------------------------- #


def test_numpy_tracking_adapter_dict_shape() -> None:
    """`make_numpy_tracking_model` accepts a dict-shape return."""

    def tracker(rgb: np.ndarray):
        return [
            {
                "track_id": "obj_0",
                "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
            }
        ]

    bm = make_numpy_tracking_model(tracker)
    from rpx_benchmark.api import Sample

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    sample = Sample(id="s0", rgb=rgb, ground_truth=TrackletGroundTruth(tracks=[]))
    [pred] = bm.predict([sample])

    assert isinstance(pred, TrackletPrediction)
    assert len(pred.tracks) == 1
    assert pred.tracks[0].track_id == "obj_0"
    assert pred.tracks[0].boxes.shape == (1, 4)


def test_numpy_tracking_adapter_tracklet_list_shape() -> None:
    def tracker(rgb: np.ndarray):
        return [
            Tracklet(
                track_id="a",
                boxes=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            ),
        ]

    bm = make_numpy_tracking_model(tracker)
    from rpx_benchmark.api import Sample

    sample = Sample(
        id="s0",
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        ground_truth=TrackletGroundTruth(tracks=[]),
    )
    [pred] = bm.predict([sample])
    assert [t.track_id for t in pred.tracks] == ["a"]


# --------------------------------------------------------------------------- #
# Golden metrics
# --------------------------------------------------------------------------- #


def _box(x: float, y: float, w: float = 10.0, h: float = 10.0) -> list[float]:
    return [x, y, x + w, y + h]


def test_tracking_metrics_perfect_recall() -> None:
    """Identity track, identical predictions → MOTA=1, IDF1=1, zero errors."""
    gt = [
        Tracklet(
            track_id="a",
            boxes=np.array([_box(0, 0), _box(5, 0), _box(10, 0)], dtype=np.float32),
        )
    ]
    pred = [
        Tracklet(
            track_id="a",
            boxes=np.array([_box(0, 0), _box(5, 0), _box(10, 0)], dtype=np.float32),
        )
    ]
    m = tracking_metrics(pred, gt)
    assert m["mota"] == pytest.approx(1.0)
    assert m["idf1"] == pytest.approx(1.0)
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["idsw"] == 0


def test_tracking_metrics_all_misses() -> None:
    """Empty predictions → all GT counted as FN, MOTA=0."""
    gt = [
        Tracklet(
            track_id="a",
            boxes=np.array([_box(0, 0), _box(5, 0)], dtype=np.float32),
        )
    ]
    m = tracking_metrics([], gt)
    assert m["fn"] == 2
    assert m["fp"] == 0
    # MOTA = 1 - (fn + fp + idsw)/gt = 1 - 2/2 = 0
    assert m["mota"] == pytest.approx(0.0)
    # IDF1 simplified: TP=0 → 0/0 clipped to 0.
    assert m["idf1"] == pytest.approx(0.0)


def test_tracking_metrics_registered_under_task() -> None:
    gt = TrackletGroundTruth(
        tracks=[
            Tracklet(
                track_id="a",
                boxes=np.array([_box(0, 0)], dtype=np.float32),
            )
        ]
    )
    pred = TrackletPrediction(
        tracks=[
            Tracklet(
                track_id="a",
                boxes=np.array([_box(0, 0)], dtype=np.float32),
            )
        ]
    )
    out = compute_metrics(TaskType.OBJECT_TRACKING, pred, gt)
    assert out["mota"] == pytest.approx(1.0)
    assert out["idf1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# End-to-end: task pipeline against a tiny synthetic dataset
# --------------------------------------------------------------------------- #


def _write_tiny_tracking_dataset(root: Path, n_frames: int = 2) -> Path:
    """Minimal JSON manifest + on-disk RGB + tracklets.json for tracking."""
    root.mkdir(parents=True, exist_ok=True)
    rgb_dir = root / "scenes" / "scene_000" / "0" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n_frames):
        Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(rgb_dir / f"{i:05d}.png")
        tracks_path = rgb_dir.parent / f"tracks_{i:05d}.json"
        tracks_path.write_text(
            json.dumps(
                [
                    {"track_id": "a", "boxes": [_box(0, 0)], "scores": [1.0]},
                ]
            )
        )
        samples.append(
            {
                "id": f"scene_000_clutter_{i:05d}",
                "rgb": f"scenes/scene_000/0/rgb/{i:05d}.png",
                "tracks": f"scenes/scene_000/0/tracks_{i:05d}.json",
                "phase": "clutter",
                "difficulty": "easy",
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "object_tracking",
                "root": str(root),
                "samples": samples,
            }
        )
    )
    return manifest


def test_tracking_pipeline_smoke(tmp_path: Path) -> None:
    """Run the OBJECT_TRACKING pipeline end-to-end with a numpy tracker."""
    from rpx_benchmark.loader import RPXDataset
    from rpx_benchmark.runner import BenchmarkRunner

    manifest = _write_tiny_tracking_dataset(tmp_path, n_frames=2)
    ds = RPXDataset.from_manifest(manifest, batch_size=1)

    def perfect_tracker(rgb: np.ndarray):
        return [
            {
                "track_id": "a",
                "boxes": np.array([_box(0, 0)], dtype=np.float32),
                "scores": np.array([1.0], dtype=np.float32),
            }
        ]

    model = make_numpy_tracking_model(perfect_tracker)
    runner = BenchmarkRunner(model=model, dataset=ds)
    result = runner.run()
    # Two samples, perfect prediction ⇒ MOTA / IDF1 average to 1.0.
    assert result.num_samples == 2
    assert result.aggregated["mota"] == pytest.approx(1.0)
    assert result.aggregated["idf1"] == pytest.approx(1.0)

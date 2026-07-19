"""Tests for deferred exact D1-F post-processing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_depth_paper_predictions.py"
    spec = importlib.util.spec_from_file_location("evaluate_depth_paper_predictions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_worker_and_atomic_cache_resume(tmp_path: Path) -> None:
    module = _script_module()
    prediction = np.ones((480, 640), dtype=np.float32)
    prediction_path = tmp_path / "prediction.npz"
    gt_path = tmp_path / "gt.png"
    np.savez_compressed(prediction_path, depth=prediction)
    Image.fromarray(np.full((480, 640), 1000, dtype=np.uint16)).save(gt_path)

    sample_id, value = module._evaluate_one(("sample", str(prediction_path), str(gt_path)))
    assert sample_id == "sample"
    assert value == 1.0

    cache_path = tmp_path / "cache" / "sample.json"
    module._atomic_json(
        cache_path,
        {
            "schema_version": "rpx-d1f-fscore-v1",
            "id": sample_id,
            "fscore_5cm": value,
        },
    )
    assert module._cached_fscore(cache_path, sample_id) == 1.0
    assert module._cached_fscore(cache_path, "different-sample") is None


def test_corrupt_or_out_of_range_fscore_cache_is_rejected(tmp_path: Path) -> None:
    module = _script_module()
    cache_path = tmp_path / "score.json"
    cache_path.write_text("not-json")
    assert module._cached_fscore(cache_path, "sample") is None

    module._atomic_json(
        cache_path,
        {
            "schema_version": "rpx-d1f-fscore-v1",
            "id": "sample",
            "fscore_5cm": 1.1,
        },
    )
    assert module._cached_fscore(cache_path, "sample") is None

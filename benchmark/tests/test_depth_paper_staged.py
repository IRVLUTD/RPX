"""Tests for deferred exact D1-F post-processing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
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


def test_exact_worker_applies_fe2e_log_alignment(tmp_path: Path) -> None:
    module = _script_module()
    gt = np.linspace(0.5, 4.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    scale, shift = 1.8, -0.3
    raw = ((np.log(gt) - shift) / scale).astype(np.float32)
    prediction_path = tmp_path / "prediction.npz"
    gt_path = tmp_path / "gt.png"
    np.savez_compressed(prediction_path, depth=raw)
    Image.fromarray(np.round(gt * 1000).astype(np.uint16)).save(gt_path)

    sample_id, value = module._evaluate_one(
        ("sample", str(prediction_path), str(gt_path), "ls_log", scale, shift)
    )
    assert sample_id == "sample"
    assert value > 0.999


def test_pooled_log_alignment_parameters_match_known_transform(tmp_path: Path) -> None:
    module = _script_module()
    dataset_root = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    samples = []
    scale, shift = 1.7, -0.2
    for frame, offset in enumerate((0.0, 0.2)):
        gt = np.linspace(0.5 + offset, 4.0, 12, dtype=np.float32).reshape(3, 4)
        raw = ((np.log(gt) - shift) / scale).astype(np.float32)
        rgb = f"scenes/scene001/0/rgb/{frame:05d}.webp"
        depth = f"scenes/scene001/0/depth/{frame:05d}.png"
        prediction_path = predictions / "scene001" / "0" / f"{frame:05d}.npz"
        gt_path = dataset_root / depth
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, depth=raw)
        Image.fromarray(np.round(gt * 1000).astype(np.uint16)).save(gt_path)
        samples.append(
            {
                "id": f"scene001__0__{frame:05d}",
                "scene_id": "scene001",
                "phase": 0,
                "rgb": rgb,
                "depth": depth,
            }
        )

    parameters = module._alignment_parameters(
        samples, dataset_root, predictions, "ls_log"
    )
    fitted_scale, fitted_shift, signature = parameters[("scene001", "0")]
    assert fitted_scale == pytest.approx(scale, abs=2e-3)
    assert fitted_shift == pytest.approx(shift, abs=2e-3)
    assert len(signature) == 64


def test_fscore_cache_is_bound_to_alignment_signature(tmp_path: Path) -> None:
    module = _script_module()
    cache_path = tmp_path / "score.json"
    module._atomic_json(
        cache_path,
        {
            "schema_version": "rpx-d1f-fscore-v1",
            "id": "sample",
            "fscore_5cm": 0.9,
            "alignment_signature": "fit-a",
        },
    )
    assert module._cached_fscore(cache_path, "sample", "fit-a") == 0.9
    assert module._cached_fscore(cache_path, "sample", "fit-b") is None

    module._atomic_json(
        cache_path,
        {
            "schema_version": "rpx-d1f-fscore-v1",
            "id": "sample",
            "fscore_5cm": 1.1,
        },
    )
    assert module._cached_fscore(cache_path, "sample") is None

"""Offline checks for the paper-task dummy models and the packaged T5/T6 questions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rpx_benchmark.examples import benchmark_tasks, dummy_models
from rpx_benchmark.hub import DEFAULT_REVISION
from rpx_benchmark.vqa.contract import load_manifest


def test_dummy_models_cover_every_paper_task() -> None:
    assert set(benchmark_tasks.DUMMY_MODELS) == {"T1", "T2", "T3", "T4", "T5", "T6"}


def test_dummy_model_outputs_match_task_contracts() -> None:
    rgb = np.zeros((48, 64, 3), np.uint8)
    assert dummy_models.image_depth(rgb).shape == (48, 64)

    depth = dummy_models.video_depth(np.zeros((3, 48, 64, 3), np.uint8))
    assert depth.shape == (3, 48, 64)
    assert np.ptp(depth) > 0  # the video evaluator rejects constant maps

    (track,) = dummy_models.tracking(rgb)
    assert track["boxes"].shape == (1, 4)

    pose = dummy_models.camera_pose(rgb, rgb)
    assert np.asarray(pose["rotation"]).shape == (3, 3)
    assert np.asarray(pose["translation"]).shape == (3,)

    for fn in (dummy_models.vqa, dummy_models.incontext_vqa):
        answer = json.loads(fn((), "Locate the cup.", 64, "bbox"))
        assert set(answer) == {"label", "bbox"} and len(answer["bbox"]) == 4


@pytest.mark.parametrize("task, in_context", [("t5", False), ("t6", True)])
def test_packaged_vqa_quickstart_is_pinned(task: str, in_context: bool) -> None:
    samples = load_manifest(benchmark_tasks.VQA_QUICKSTART / f"{task}.jsonl")
    assert samples
    assert all(sample.is_in_context == in_context for sample in samples)
    for sample in samples:
        assert sample.image["revision"] == DEFAULT_REVISION
        if sample.reference_image is not None:
            assert sample.reference_image["revision"] == DEFAULT_REVISION


def test_quickstart_api_is_exported() -> None:
    import rpx_benchmark as rpx

    assert {"download", "benchmark", "dummy_models"} <= set(rpx.__all__)


def test_benchmark_rejects_unknown_task(tmp_path) -> None:
    from rpx_benchmark.exceptions import ConfigError
    from rpx_benchmark.quickstart import benchmark

    with pytest.raises(ConfigError):
        benchmark("T9", tmp_path / "manifest.json")


@pytest.mark.parametrize("task", ["T1", "T4"])
def test_benchmark_scores_dummy_model_offline(tmp_path, task: str) -> None:
    from rpx_benchmark.quickstart import benchmark

    manifest, _ = benchmark_tasks.create_smoke(tmp_path / "inputs", task)
    report = benchmark(task, manifest, output=tmp_path / "results")
    assert report["num_samples"] == 1
    assert report["metrics"]

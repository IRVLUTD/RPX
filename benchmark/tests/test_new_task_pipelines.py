"""End-to-end pipeline tests for the six new task runners.

Each task gets a small synthetic dataset fixture plus a fake numpy
model wrapped via the corresponding ``make_numpy_*_model`` factory.
These tests verify that adding a new task is genuinely a one-file
change — the shared ``run_pipeline`` helper carries everything
from config validation through reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

import rpx_benchmark as rpx
from rpx_benchmark.api import TaskType
from rpx_benchmark.evaluators import MetricSuite
from rpx_benchmark.runner import BenchmarkRunner

# --------------------------------------------------------------------------- #
# Helpers — tiny synthetic datasets per task
# --------------------------------------------------------------------------- #


def _write_rgb(path: Path, h: int = 20, w: int = 30, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), value, np.uint8)).save(path)


def _write_pose(path: Path, x: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        position=np.array([x, 0.0, 0.0], dtype=np.float64),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )


def _manifest(tmp_path: Path, task: str, samples: List[Dict[str, Any]]) -> Path:
    mf = tmp_path / "manifest.json"
    mf.write_text(
        json.dumps(
            {
                "task": task,
                "root": str(tmp_path),
                "samples": samples,
            }
        )
    )
    return mf


# --------------------------------------------------------------------------- #
# Object detection
# --------------------------------------------------------------------------- #


def _det_dataset(tmp_path: Path) -> rpx.RPXDataset:
    _write_rgb(tmp_path / "rgb" / "0.png")
    # GT box JSON referenced by the manifest
    boxes_file = tmp_path / "boxes.json"
    boxes_file.write_text(
        json.dumps(
            [
                {"bbox": [5, 5, 15, 15], "label": "cup"},
                {"bbox": [20, 10, 28, 18], "label": "bowl"},
            ]
        )
    )
    manifest = _manifest(
        tmp_path,
        "object_detection",
        [
            {
                "id": "t",
                "rgb": "rgb/0.png",
                "boxes": "boxes.json",
                "phase": "clutter",
                "difficulty": "hard",
            }
        ],
    )
    return rpx.RPXDataset.from_manifest(manifest, batch_size=1)


def test_detection_pipeline_end_to_end(tmp_path):
    def perfect_det(rgb):
        return {
            "boxes": np.array([[5, 5, 15, 15], [20, 10, 28, 18]], dtype=np.float32),
            "scores": np.array([0.99, 0.95], dtype=np.float32),
            "labels": ["cup", "bowl"],
        }

    bm = rpx.make_numpy_detection_model(perfect_det)
    ds = _det_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.OBJECT_DETECTION))
    result, dr = runner.run_with_report(
        primary_metric="f1",
        model_name="perfect_det",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.aggregated["precision"] == 1.0
    assert result.aggregated["recall"] == 1.0
    assert result.aggregated["f1"] == 1.0


def test_detection_tuple_return_shape(tmp_path):
    def tuple_det(rgb):
        return (
            np.array([[5, 5, 15, 15]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            ["cup"],
        )

    bm = rpx.make_numpy_detection_model(tuple_det)
    ds = _det_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.OBJECT_DETECTION))
    result, _ = runner.run_with_report(
        primary_metric="f1",
        model_name="tup",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    # Matched one of the two GT boxes → precision 1, recall 0.5, f1 2/3
    assert result.aggregated["precision"] == 1.0
    assert result.aggregated["recall"] == 0.5


# --------------------------------------------------------------------------- #
# Visual grounding
# --------------------------------------------------------------------------- #


def _grounding_dataset(tmp_path: Path) -> rpx.RPXDataset:
    _write_rgb(tmp_path / "rgb" / "0.png")
    manifest = _manifest(
        tmp_path,
        "visual_grounding",
        [
            {
                "id": "g0",
                "rgb": "rgb/0.png",
                "text": "the red cup",
                "boxes": [[10, 10, 20, 20]],
                "phase": "clutter",
                "difficulty": "hard",
            }
        ],
    )
    return rpx.RPXDataset.from_manifest(manifest, batch_size=1)


def test_grounding_pipeline_end_to_end(tmp_path):
    def perfect(rgb, text):
        assert text == "the red cup"
        return {
            "boxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
        }

    bm = rpx.make_numpy_grounding_model(perfect)
    ds = _grounding_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.VISUAL_GROUNDING))
    result, _ = runner.run_with_report(
        primary_metric="grounding_acc",
        model_name="g",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.aggregated["grounding_iou"] == 1.0
    assert result.aggregated["grounding_acc"] == 1.0


# --------------------------------------------------------------------------- #
# Relative camera pose
# --------------------------------------------------------------------------- #


def _pose_dataset(tmp_path: Path) -> rpx.RPXDataset:
    _write_rgb(tmp_path / "rgb" / "a.png")
    _write_rgb(tmp_path / "rgb" / "b.png")
    _write_pose(tmp_path / "pose" / "a.npz", x=0.0)
    _write_pose(tmp_path / "pose" / "b.npz", x=0.5)
    manifest = _manifest(
        tmp_path,
        "relative_camera_pose",
        [
            {
                "id": "pair",
                "rgb": "rgb/a.png",
                "rgb_b": "rgb/b.png",
                "pose_a": "pose/a.npz",
                "pose_b": "pose/b.npz",
                "phase": "clutter",
                "difficulty": "hard",
            }
        ],
    )
    return rpx.RPXDataset.from_manifest(manifest, batch_size=1)


def test_pose_pipeline_end_to_end(tmp_path):
    def perfect_pose(rgb_a, rgb_b):
        return {"rotation": np.eye(3), "translation": np.array([0.5, 0.0, 0.0])}

    bm = rpx.make_numpy_pose_model(perfect_pose)
    ds = _pose_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.RELATIVE_CAMERA_POSE))
    result, _ = runner.run_with_report(
        primary_metric="rotation_error_deg",
        model_name="p",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.aggregated["rotation_error_deg"] == 0.0
    assert result.aggregated["translation_error_m"] < 1e-6


def test_pose_numpy_adapter_rejects_missing_rgb_b(tmp_path):
    _write_rgb(tmp_path / "rgb" / "a.png")
    _write_pose(tmp_path / "pose" / "a.npz", x=0.0)
    _write_pose(tmp_path / "pose" / "b.npz", x=0.5)
    # Manifest without rgb_b — the loader's NPZ pair still produces a GT,
    # but the numpy adapter should complain.
    manifest = _manifest(
        tmp_path,
        "relative_camera_pose",
        [
            {
                "id": "pair",
                "rgb": "rgb/a.png",
                "pose_a": "pose/a.npz",
                "pose_b": "pose/b.npz",
            }
        ],
    )
    ds = rpx.RPXDataset.from_manifest(manifest, batch_size=1)
    bm = rpx.make_numpy_pose_model(lambda a, b: {"rotation": np.eye(3), "translation": np.zeros(3)})
    from rpx_benchmark.exceptions import AdapterError

    with pytest.raises(AdapterError, match="second RGB frame"):
        for batch in ds:
            bm.predict(batch)


# --------------------------------------------------------------------------- #
# Sparse depth
# --------------------------------------------------------------------------- #


def _sparse_dataset(tmp_path: Path) -> rpx.RPXDataset:
    _write_rgb(tmp_path / "rgb" / "0.png")
    # Inline coordinates + depths so we don't need npy round-trip.
    manifest = _manifest(
        tmp_path,
        "sparse_depth",
        [
            {
                "id": "s",
                "rgb": "rgb/0.png",
                "coordinates": [[5, 5], [10, 10], [15, 15]],
                "depths": [1.0, 2.0, 3.0],
                "phase": "clutter",
                "difficulty": "hard",
            }
        ],
    )
    return rpx.RPXDataset.from_manifest(manifest, batch_size=1)


def test_sparse_depth_pipeline_end_to_end(tmp_path):
    def perfect(rgb, coords):
        # Return the exact GT depths so AbsRel = 0.
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    bm = rpx.make_numpy_sparse_depth_model(perfect)
    ds = _sparse_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.SPARSE_DEPTH))
    result, _ = runner.run_with_report(
        primary_metric="sparse_absrel",
        model_name="s",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.aggregated["sparse_absrel"] == 0.0
    assert result.aggregated["sparse_rmse"] == 0.0


# --------------------------------------------------------------------------- #
# Novel view synthesis
# --------------------------------------------------------------------------- #


def _nvs_dataset(tmp_path: Path) -> rpx.RPXDataset:
    _write_rgb(tmp_path / "rgb" / "src.png", value=64)
    _write_rgb(tmp_path / "rgb" / "tgt.png", value=200)
    _write_pose(tmp_path / "pose" / "tgt.npz", x=0.3)
    manifest = _manifest(
        tmp_path,
        "novel_view_synthesis",
        [
            {
                "id": "n0",
                "rgb": "rgb/src.png",
                "target_rgb": "rgb/tgt.png",
                "target_pose": "pose/tgt.npz",
                "phase": "clutter",
                "difficulty": "hard",
            }
        ],
    )
    return rpx.RPXDataset.from_manifest(manifest, batch_size=1)


def test_nvs_pipeline_end_to_end(tmp_path):
    def perfect_nvs(rgb, target_pose):
        # Return an all-200 image that matches the target.
        return np.full_like(rgb, 200)

    bm = rpx.make_numpy_nvs_model(perfect_nvs)
    ds = _nvs_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.NOVEL_VIEW_SYNTHESIS))
    result, _ = runner.run_with_report(
        primary_metric="psnr",
        model_name="nvs",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    # Identical frames → high PSNR (sentinel value).
    assert result.aggregated["psnr"] >= 99.0


# --------------------------------------------------------------------------- #
# Keypoint matching
# --------------------------------------------------------------------------- #


def _keypoint_dataset(tmp_path: Path) -> rpx.RPXDataset:
    _write_rgb(tmp_path / "rgb" / "a.png")
    _write_rgb(tmp_path / "rgb" / "b.png")
    p0 = np.array([[5, 5], [10, 10], [15, 15]], dtype=np.float32)
    p1 = np.array([[6, 5], [11, 10], [16, 15]], dtype=np.float32)
    vis = np.array([True, True, True])
    (tmp_path / "kp").mkdir()
    np.save(tmp_path / "kp" / "p0.npy", p0)
    np.save(tmp_path / "kp" / "p1.npy", p1)
    np.save(tmp_path / "kp" / "vis.npy", vis)
    manifest = _manifest(
        tmp_path,
        "keypoint_matching",
        [
            {
                "id": "k",
                "rgb": "rgb/a.png",
                "rgb_b": "rgb/b.png",
                "points0": "kp/p0.npy",
                "points1": "kp/p1.npy",
                "visibility": "kp/vis.npy",
                "phase": "clutter",
                "difficulty": "hard",
            }
        ],
    )
    return rpx.RPXDataset.from_manifest(manifest, batch_size=1)


def test_keypoint_pipeline_end_to_end(tmp_path):
    # Return the same points the GT stores; px_threshold is 3 so the
    # 1-pixel shift in p1 (x+1) is well within threshold.
    def perfect_matcher(rgb_a, rgb_b):
        return (
            np.array([[5, 5], [10, 10], [15, 15]], dtype=np.float32),
            np.array([[6, 5], [11, 10], [16, 15]], dtype=np.float32),
        )

    bm = rpx.make_numpy_keypoint_model(perfect_matcher)
    ds = _keypoint_dataset(tmp_path)
    runner = BenchmarkRunner(bm, ds, MetricSuite.for_task(TaskType.KEYPOINT_MATCHING))
    result, _ = runner.run_with_report(
        primary_metric="keypoint_acc",
        model_name="k",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.aggregated["keypoint_acc"] == 1.0
    assert result.aggregated["mean_match_error"] < 1.0


# --------------------------------------------------------------------------- #
# Task registry: every new task is discoverable
# --------------------------------------------------------------------------- #


def test_all_new_tasks_registered():
    from rpx_benchmark.tasks.registry import available_tasks

    registered = {t.value for t in available_tasks()}
    for t in (
        "monocular_depth",
        "object_segmentation",
        "object_detection",
        "open_vocab_detection",
        "visual_grounding",
        "relative_camera_pose",
        "keypoint_matching",
        "sparse_depth",
        "novel_view_synthesis",
    ):
        assert t in registered, f"{t} is not in the task registry"


# CLI subcommand discovery test removed with the CLI deletion in
# v0.3.0 — the task-registry discovery it was exercising is covered
# by ``test_new_tasks_have_specs`` above.

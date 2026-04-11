"""End-to-end segmentation pipeline tests.

Exercises the full chain (synthetic dataset → ``BenchmarkableModel`` →
``BenchmarkRunner`` → ``MetricSuite`` → reports) without touching
HuggingFace or torch. A fake numpy-mask callable is used as the
model under test, so the plugin system carries the task-specific
knobs (primary metric, higher_is_better, ground-truth type).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rpx_benchmark as rpx
from rpx_benchmark.api import SegmentationPrediction, TaskType
from rpx_benchmark.evaluators import MetricSuite
from rpx_benchmark.runner import BenchmarkRunner
from rpx_benchmark.tasks.registry import get_task_spec


# --------------------------------------------------------------------------- #
# Fixture: 6-frame synthetic segmentation dataset
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_seg_dataset(tmp_path: Path) -> rpx.RPXDataset:
    samples = []
    for phase_idx, phase_name in (("0", "clutter"),
                                   ("1", "interaction"),
                                   ("2", "clean")):
        for difficulty in ("easy", "hard"):
            pdir = tmp_path / "scenes" / "scene_000" / phase_idx
            (pdir / "rgb").mkdir(parents=True, exist_ok=True)
            (pdir / "mask").mkdir(parents=True, exist_ok=True)
            frame = f"0{difficulty[0]}"
            # RGB: uniform gray
            Image.fromarray(np.full((20, 30, 3), 128, np.uint8)).save(
                pdir / "rgb" / f"{frame}.png"
            )
            # GT mask: two instances (1 = upper-left square, 2 = lower-right square)
            mask = np.zeros((20, 30), dtype=np.int32)
            mask[2:10, 2:10] = 1
            mask[12:18, 18:26] = 2
            Image.fromarray(mask.astype(np.int32), mode="I").save(
                pdir / "mask" / f"{frame}.png"
            )
            samples.append({
                "id": f"scene_000_{phase_name}_{frame}",
                "rgb":  f"scenes/scene_000/{phase_idx}/rgb/{frame}.png",
                "mask": f"scenes/scene_000/{phase_idx}/mask/{frame}.png",
                "phase": phase_name,
                "difficulty": difficulty,
            })
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "task": "object_segmentation",
        "root": str(tmp_path),
        "samples": samples,
    }))
    return rpx.RPXDataset.from_manifest(manifest_path, batch_size=1)


# --------------------------------------------------------------------------- #
# make_numpy_mask_model contract
# --------------------------------------------------------------------------- #

def test_numpy_mask_model_returns_segmentation_prediction():
    def fn(rgb):
        return np.ones(rgb.shape[:2], dtype=np.int32)

    bm = rpx.make_numpy_mask_model(fn, name="unit")
    assert bm.task is TaskType.OBJECT_SEGMENTATION

    from rpx_benchmark.api import Sample
    sample = Sample(
        id="t",
        rgb=np.full((20, 30, 3), 128, np.uint8),
        ground_truth=None,
    )
    preds = bm.predict([sample])
    assert len(preds) == 1
    assert isinstance(preds[0], SegmentationPrediction)
    assert preds[0].mask.shape == (20, 30)
    assert preds[0].mask.dtype == np.int32


def test_numpy_mask_model_nearest_resizes_output():
    def fn(rgb):
        # Return at half resolution
        return np.ones((10, 15), dtype=np.int32)

    bm = rpx.make_numpy_mask_model(fn)
    from rpx_benchmark.api import Sample
    sample = Sample(
        id="t",
        rgb=np.full((20, 30, 3), 128, np.uint8),
        ground_truth=None,
    )
    out = bm.predict([sample])[0]
    assert out.mask.shape == (20, 30)
    assert out.mask.dtype == np.int32
    # Nearest upsample preserves integer values (all 1s)
    assert (out.mask == 1).all()


def test_numpy_mask_model_rejects_non_2d_output():
    def fn(rgb):
        return np.zeros((2, 4, 5), dtype=np.int32)

    bm = rpx.make_numpy_mask_model(fn)
    from rpx_benchmark.api import Sample
    from rpx_benchmark.exceptions import AdapterError
    sample = Sample(
        id="t",
        rgb=np.full((10, 10, 3), 128, np.uint8),
        ground_truth=None,
    )
    with pytest.raises(AdapterError, match="2-D array"):
        bm.predict([sample])


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #

def _perfect_seg(rgb: np.ndarray) -> np.ndarray:
    """Exactly matches the synthetic GT mask from the fixture."""
    mask = np.zeros(rgb.shape[:2], dtype=np.int32)
    mask[2:10, 2:10] = 1
    mask[12:18, 18:26] = 2
    return mask


def _wrong_seg(rgb: np.ndarray) -> np.ndarray:
    """Deliberately wrong: all-background."""
    return np.zeros(rgb.shape[:2], dtype=np.int32)


def test_segmentation_perfect_prediction_yields_miou_one(synthetic_seg_dataset):
    bm = rpx.make_numpy_mask_model(_perfect_seg, name="perfect")
    runner = BenchmarkRunner(
        bm, synthetic_seg_dataset,
        MetricSuite.for_task(TaskType.OBJECT_SEGMENTATION),
    )
    result, dr = runner.run_with_deployment_readiness(
        primary_metric="miou",
        model_name="perfect",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.num_samples == 6
    assert result.aggregated["miou"] == 1.0
    wps = dr.weighted_phase_score
    assert wps is not None
    assert wps.s_overall == 1.0


def test_segmentation_runner_attaches_metadata(synthetic_seg_dataset):
    bm = rpx.make_numpy_mask_model(_perfect_seg)
    runner = BenchmarkRunner(
        bm, synthetic_seg_dataset,
        MetricSuite.for_task(TaskType.OBJECT_SEGMENTATION),
    )
    result, _ = runner.run_with_deployment_readiness(
        primary_metric="miou",
        model_name="unit",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert {"miou", "id", "phase", "difficulty"} <= result.per_sample[0].keys()
    assert {"id", "phase", "difficulty"}.isdisjoint(result.aggregated.keys())


def test_segmentation_wrong_prediction_has_nonzero_error(synthetic_seg_dataset):
    bm = rpx.make_numpy_mask_model(_wrong_seg)
    runner = BenchmarkRunner(
        bm, synthetic_seg_dataset,
        MetricSuite.for_task(TaskType.OBJECT_SEGMENTATION),
    )
    result, _ = runner.run_with_deployment_readiness(
        primary_metric="miou",
        model_name="allbg",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    # Everything-background gets class 0 right and classes 1,2 wrong.
    assert result.aggregated["miou"] < 0.5


# --------------------------------------------------------------------------- #
# Task registration
# --------------------------------------------------------------------------- #

def test_segmentation_task_is_registered():
    spec = get_task_spec(TaskType.OBJECT_SEGMENTATION)
    assert spec.primary_metric == "miou"
    assert spec.higher_is_better is True
    assert "rgb" in spec.required_modalities
    assert "mask" in spec.required_modalities
    assert callable(spec.build_config)
    assert callable(spec.run)
    assert callable(spec.add_cli_arguments)


def test_segmentation_subcommand_in_cli(capsys):
    from rpx_benchmark import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["bench", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "object_segmentation" in out

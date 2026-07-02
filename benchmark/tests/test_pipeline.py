"""End-to-end pipeline test against the synthetic dataset fixture.

These tests hit the full chain (loader → BenchmarkableModel →
BenchmarkRunner → MetricSuite → DeploymentReadinessReport → reports)
without touching HuggingFace, torch, or real checkpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

import rpx_benchmark as rpx
from rpx_benchmark.api import TaskType
from rpx_benchmark.evaluators import MetricSuite
from rpx_benchmark.reports import format_markdown_summary, write_json
from rpx_benchmark.runner import BenchmarkRunner
from rpx_benchmark.tasks._pipeline import resolve_device


def _perfect_depth(rgb: np.ndarray) -> np.ndarray:
    """Return 2 m everywhere — matches the 2000 mm synthetic GT exactly."""
    return np.full(rgb.shape[:2], 2.0, dtype=np.float32)


def test_gpu_only_resolution_rejects_cpu_request():
    from rpx_benchmark.exceptions import ConfigError

    with pytest.raises(ConfigError, match="GPU-only"):
        resolve_device("cpu", require_cuda=True)


def test_runner_attaches_per_sample_metadata(synthetic_depth_dataset):
    bm = rpx.make_numpy_depth_model(_perfect_depth, name="unit")
    runner = BenchmarkRunner(
        bm,
        synthetic_depth_dataset,
        MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
    )
    result, _ = runner.run_with_report(
        primary_metric="absrel",
        model_name="unit",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.num_samples == 6
    # Every per_sample entry has metric + metadata fields
    for entry in result.per_sample:
        assert {"absrel", "rmse", "delta1", "id", "phase", "difficulty"} <= entry.keys()
    # Metadata does not leak into aggregated means
    assert {"id", "phase", "difficulty"}.isdisjoint(result.aggregated.keys())
    assert {"absrel", "rmse"} <= result.aggregated.keys()


def test_perfect_prediction_yields_zero_error(synthetic_depth_dataset):
    bm = rpx.make_numpy_depth_model(_perfect_depth)
    runner = BenchmarkRunner(
        bm,
        synthetic_depth_dataset,
        MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
    )
    result, dr = runner.run_with_report(
        primary_metric="absrel",
        model_name="perfect",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert result.aggregated["absrel"] == 0.0
    assert result.aggregated["rmse"] == 0.0
    assert dr.weighted_phase_score is not None
    wps = dr.weighted_phase_score
    assert wps.s_overall == 0.0
    assert wps.delta_int == 0.0


def test_latency_skips_warmup_and_is_reasonable(synthetic_depth_dataset):
    """Per-sample latency should reflect non-warmup timing roughly."""

    def slow_depth(rgb: np.ndarray) -> np.ndarray:
        time.sleep(0.01)  # 10 ms per sample
        return np.full(rgb.shape[:2], 2.0, dtype=np.float32)

    bm = rpx.make_numpy_depth_model(slow_depth)
    runner = BenchmarkRunner(
        bm,
        synthetic_depth_dataset,
        MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
    )
    _, dr = runner.run_with_report(
        primary_metric="absrel",
        model_name="slow",
        compute_ts=False,
        compute_sgc_flag=False,
    )
    assert dr.latency_ms_per_sample is not None
    # Median of per-sample wall clock should be within the 10 ms order of
    # magnitude. Allow generous slack for CI noise.
    assert 5.0 <= dr.latency_ms_per_sample <= 200.0


def test_reports_write_json_and_markdown(synthetic_depth_dataset, tmp_path: Path):
    bm = rpx.make_numpy_depth_model(_perfect_depth)
    runner = BenchmarkRunner(
        bm,
        synthetic_depth_dataset,
        MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
    )
    result, dr = runner.run_with_report(
        primary_metric="absrel",
        model_name="perfect",
        compute_ts=False,
        compute_sgc_flag=False,
    )

    json_path = write_json(
        tmp_path / "out.json",
        task="monocular_depth",
        model_name="perfect",
        split="all",
        repo_id="synthetic",
        result=result,
        dr_report=dr,
    )
    md = format_markdown_summary(
        task="monocular_depth",
        model_name="perfect",
        split="all",
        repo_id="synthetic",
        result=result,
        dr_report=dr,
    )

    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["num_samples"] == 6
    assert payload["aggregated"]["absrel"] == 0.0
    # Metadata keys are present per-sample
    assert "id" in payload["per_sample"][0]
    assert "phase" in payload["per_sample"][0]

    assert "RPX benchmark" in md
    assert "Weighted Phase Score" in md
    assert "0.0000" in md


def test_progress_callback_fires_once_per_sample(synthetic_depth_dataset):
    events = []

    def cb(done, total, stage):
        events.append((done, total, stage))

    bm = rpx.make_numpy_depth_model(_perfect_depth)
    runner = BenchmarkRunner(
        bm,
        synthetic_depth_dataset,
        MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
    )
    runner.run_with_report(
        primary_metric="absrel",
        model_name="cb_test",
        compute_ts=False,
        compute_sgc_flag=False,
        progress=cb,
    )
    # 1 setup + 1 predict-start + 6 per-sample updates
    stages = {e[2] for e in events}
    assert "setup" in stages
    assert "predict" in stages
    done_counts = [e[0] for e in events if e[2] == "predict"]
    assert done_counts[-1] == 6


def test_relative_depth_alignment_is_pooled_per_scene_phase(
    synthetic_depth_dataset,
    monkeypatch,
):
    """The evaluator must solve three phase cells, not six frames."""
    from rpx_benchmark import runner as runner_module

    # The shared fixture predates scene metadata; add the canonical field
    # that official/local RPX manifests carry.
    for entry in synthetic_depth_dataset.samples:
        entry["metadata"] = {"scene_id": "scene_000"}

    real_align = runner_module.align_pred_to_gt_pooled
    calls = []

    def recording_align(pred_seq, gt_seq, mode, valid_seq=None):
        calls.append((pred_seq.shape, mode))
        return real_align(pred_seq, gt_seq, mode, valid_seq)

    monkeypatch.setattr(runner_module, "align_pred_to_gt_pooled", recording_align)
    bm = rpx.make_numpy_depth_model(
        lambda rgb: np.full(rgb.shape[:2], 7.0, dtype=np.float32),
        depth_output_kind="relative",
        native_alignment="ls_affine",
    )
    result, _ = BenchmarkRunner(
        bm,
        synthetic_depth_dataset,
        MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
    ).run_with_report(
        primary_metric="absrel",
        compute_ts=False,
        compute_sgc_flag=False,
    )

    assert len(calls) == 3
    assert all(shape[0] == 2 for shape, _mode in calls)
    assert {mode for _shape, mode in calls} == {"ls_affine"}
    assert result.aggregated["absrel"] < 1e-6

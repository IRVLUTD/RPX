"""Tests for the Embodied Readiness Score and per-sample latency.

Covers:

- Formula arithmetic on controlled inputs (perfect / terrible / partial).
- Missing-component handling: ``None`` fields drop out and weights
  re-normalise.
- Direction: ``higher_is_better=False`` (e.g. AbsRel) maps 0 → 1.
- Runner attaches per-sample ``latency_ms`` to every metric row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rpx_benchmark as rpx
from rpx_benchmark.deployment import (
    DEFAULT_ERS_BUDGETS,
    DEFAULT_ERS_WEIGHTS,
    DeploymentReadinessReport,
    EmbodiedReadinessScore,
    ESDResult,
    StateTransitionRobustnessResult,
    TemporalStabilityResult,
    WeightedPhaseScore,
    compute_embodied_readiness,
)
from rpx_benchmark.exceptions import MetricError


def _wps_at(value: float, metric_key: str = "x") -> WeightedPhaseScore:
    """Build a WeightedPhaseScore whose every phase score equals ``value``."""
    def phase() -> ESDResult:
        return ESDResult(easy=value, medium=value, hard=value, metric_key=metric_key)
    return WeightedPhaseScore(clutter=phase(), interaction=phase(), clean=phase())


def _full_report(
    wps_overall: float = 1.0,
    ts: float | None = 1.0,
    str_drop: float = 0.0,
    flops_g: float | None = 50.0,
    latency_ms: float | None = 10.0,
    peak_mem_mb: float | None = 500.0,
) -> DeploymentReadinessReport:
    ts_res = (
        TemporalStabilityResult(ts_score=ts, num_pairs=10) if ts is not None else None
    )
    str_res = StateTransitionRobustnessResult(
        str_c_to_i=str_drop, str_i_to_l=0.0,
        metric_clutter=0.0, metric_interaction=0.0, metric_clean=0.0,
    )
    return DeploymentReadinessReport(
        task="monocular_depth", model_name="test",
        weighted_phase_score=_wps_at(wps_overall),
        temporal_stability=ts_res,
        state_transition=str_res,
        flops_g=flops_g,
        latency_ms_per_sample=latency_ms,
        peak_memory_mb=peak_mem_mb,
    )


# --------------------------------------------------------------------------- #
# Formula corners
# --------------------------------------------------------------------------- #

def test_perfect_report_gives_max_ers() -> None:
    report = _full_report(
        wps_overall=1.0, ts=1.0, str_drop=0.0,
        flops_g=0.0, latency_ms=0.0, peak_mem_mb=0.0,
    )
    ers = compute_embodied_readiness(report, higher_is_better=True)
    assert ers.score == pytest.approx(1.0)
    assert ers.accuracy == pytest.approx(1.0)
    assert ers.robustness == pytest.approx(1.0)
    assert ers.latency == pytest.approx(1.0)
    assert ers.memory == pytest.approx(1.0)
    assert ers.compute == pytest.approx(1.0)


def test_terrible_report_gives_zero_ers() -> None:
    report = _full_report(
        wps_overall=0.0, ts=0.0, str_drop=1.0,
        flops_g=DEFAULT_ERS_BUDGETS["flops_g"] * 10,
        latency_ms=DEFAULT_ERS_BUDGETS["latency_ms"] * 10,
        peak_mem_mb=DEFAULT_ERS_BUDGETS["memory_mb"] * 10,
    )
    ers = compute_embodied_readiness(report, higher_is_better=True)
    assert ers.score == pytest.approx(0.0)


def test_ers_drops_missing_components_and_renormalises() -> None:
    """Report without TS / STR / mem / flops still produces a score."""
    report = DeploymentReadinessReport(
        task="monocular_depth", model_name="t",
        weighted_phase_score=_wps_at(0.5),
        latency_ms_per_sample=50.0,   # half of latency budget
    )
    ers = compute_embodied_readiness(report, higher_is_better=True)
    w_acc = DEFAULT_ERS_WEIGHTS["accuracy"]
    w_lat = DEFAULT_ERS_WEIGHTS["latency"]
    expected = (w_acc * 0.5 + w_lat * 0.5) / (w_acc + w_lat)
    assert ers.score == pytest.approx(expected)
    assert ers.robustness is None
    assert ers.memory is None
    assert ers.compute is None


def test_lower_is_better_accuracy_decay() -> None:
    """WPS = 0 → accuracy 1; WPS = 1 → accuracy ≈ exp(-1)."""
    ers_zero = compute_embodied_readiness(
        DeploymentReadinessReport(
            task="t", model_name="m", weighted_phase_score=_wps_at(0.0),
        ),
        higher_is_better=False,
    )
    assert ers_zero.accuracy == pytest.approx(1.0)

    ers_one = compute_embodied_readiness(
        DeploymentReadinessReport(
            task="t", model_name="m", weighted_phase_score=_wps_at(1.0),
        ),
        higher_is_better=False,
    )
    assert ers_one.accuracy == pytest.approx(np.exp(-1.0), rel=1e-4)


def test_empty_report_still_scores_zero_accuracy() -> None:
    """With no WPS the accuracy component defaults to 0, so the ERS is
    well-defined (= 0). No MetricError is raised since accuracy is
    always derivable unless the caller deliberately nulls it out."""
    empty = DeploymentReadinessReport(task="t", model_name="m")
    ers = compute_embodied_readiness(empty, higher_is_better=True)
    assert ers.score == pytest.approx(0.0)
    assert ers.accuracy == pytest.approx(0.0)
    assert ers.robustness is None and ers.latency is None
    assert ers.memory is None and ers.compute is None


def test_explicit_accuracy_wins() -> None:
    """Passing ``accuracy`` overrides the wps-based derivation."""
    report = DeploymentReadinessReport(
        task="t", model_name="m",
        weighted_phase_score=_wps_at(0.0),
        latency_ms_per_sample=0.0,
    )
    ers = compute_embodied_readiness(report, higher_is_better=True, accuracy=0.8)
    assert ers.accuracy == pytest.approx(0.8)


def test_custom_weights_applied() -> None:
    report = _full_report(latency_ms=50.0)
    ers = compute_embodied_readiness(
        report,
        weights={"accuracy": 0.0, "robustness": 0.0, "latency": 1.0,
                 "memory": 0.0, "compute": 0.0},
    )
    # With only latency weighted, composite equals the latency component.
    assert ers.score == pytest.approx(ers.latency)


def test_ers_exposed_on_top_level() -> None:
    assert rpx.EmbodiedReadinessScore is EmbodiedReadinessScore
    assert rpx.compute_embodied_readiness is compute_embodied_readiness


# --------------------------------------------------------------------------- #
# Runner integration: per-sample latency lands in metric rows
# --------------------------------------------------------------------------- #

def _tiny_depth_manifest(root: Path, n: int = 3) -> Path:
    rgb = root / "rgb"
    depth = root / "depth"
    rgb.mkdir(parents=True, exist_ok=True)
    depth.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        Image.fromarray(np.full((16, 16, 3), 100, np.uint8)).save(rgb / f"{i}.png")
        Image.fromarray(np.full((16, 16), 2000, np.uint16)).save(depth / f"{i}.png")
        samples.append({
            "id": f"s{i}", "rgb": f"rgb/{i}.png", "depth": f"depth/{i}.png",
            "phase": "clutter", "difficulty": "easy",
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "task": "monocular_depth", "root": str(root), "samples": samples,
    }))
    return manifest


def test_runner_attaches_per_sample_latency(tmp_path: Path) -> None:
    from rpx_benchmark.loader import RPXDataset
    from rpx_benchmark.runner import BenchmarkRunner

    manifest = _tiny_depth_manifest(tmp_path, n=4)
    ds = RPXDataset.from_manifest(manifest, batch_size=2)

    def constant_depth(rgb):
        return np.full(rgb.shape[:2], 2.0, dtype=np.float32)

    model = rpx.make_numpy_depth_model(constant_depth)
    runner = BenchmarkRunner(model=model, dataset=ds)
    result, report = runner.run_with_deployment_readiness(
        primary_metric="absrel", model_name="unit",
    )

    assert all("latency_ms" in row for row in result.per_sample)
    assert all(row["latency_ms"] >= 0 for row in result.per_sample)
    assert report.embodied_readiness is not None
    assert 0.0 <= report.embodied_readiness.score <= 1.0

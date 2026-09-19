"""Synthetic fixtures for RPX D1-F paper statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.exceptions import DatasetError
from rpx_benchmark.metrics.depth_paper import PAPER_METRIC_KEYS
from rpx_benchmark.paper_depth_analysis import (
    _holm_adjust,
    _hotelling_transition,
    analyze_depth_cells,
    write_depth_analysis,
)

PHASES = ("clutter", "interaction", "clean")


def _rows(*, phase_effect: float = 0.0, interaction: bool = False):
    rng = np.random.default_rng(41)
    rows = []
    for scene_index in range(100):
        difficulty = ("easy", "medium", "hard")[min(scene_index // 33, 2)]
        base = rng.normal(size=6) + np.arange(6) * scene_index / 100.0
        scene_phase_noise = (
            rng.normal(scale=0.025, size=(3, 6)) if phase_effect else np.zeros((3, 6))
        )
        difficulty_sign = {"easy": -1.0, "medium": 0.0, "hard": 1.0}[difficulty]
        for phase_index, phase in enumerate(PHASES):
            shift = phase_effect * phase_index
            if interaction:
                shift *= difficulty_sign
            values = base + shift + scene_phase_noise[phase_index]
            row = {
                "model_name": "DA-V2 Large",
                "task": "monocular_depth",
                "scene_id": f"scene_{scene_index:03d}",
                "phase": phase,
                "difficulty": difficulty,
                "n_samples": 250,
                "frame_budget": 0,
            }
            row.update(
                {f"metric:{key}": float(values[i]) for i, key in enumerate(PAPER_METRIC_KEYS)}
            )
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _three_logs(tmp_path: Path, rows):
    groups = {"easy": [], "medium": [], "hard": []}
    for row in rows:
        groups[row["difficulty"]].append(row)
    return [_write_jsonl(tmp_path / f"{tier}.jsonl", groups[tier]) for tier in groups]


def test_phase_invariant_cube_has_phi_one(tmp_path: Path) -> None:
    analysis, cells = analyze_depth_cells(_three_logs(tmp_path, _rows()))
    assert len(cells) == 300
    assert analysis["overall_phase_manova"]["phi_wilks"] == 1.0
    assert analysis["overall_phase_manova"]["p_value"] == 1.0
    assert all(item["phi"] == 1.0 for item in analysis["transitions"])
    assert analysis["jedi"] == {"status": "bounds_missing"}


def test_known_phase_shift_is_significant_and_holm_is_monotone(tmp_path: Path) -> None:
    analysis, _ = analyze_depth_cells(_three_logs(tmp_path, _rows(phase_effect=0.35)))
    overall = analysis["overall_phase_manova"]
    assert 0.0 <= overall["phi_wilks"] < 1.0
    assert overall["p_value"] < 0.05
    assert overall["phi_wilks"] <= overall["phi_pillai"] + 1e-12
    for transition in analysis["transitions"]:
        assert transition["p_value_holm"] >= transition["p_value_raw"]


def test_hotelling_and_holm_match_independent_fixture() -> None:
    from scipy.stats import f as f_distribution

    rng = np.random.default_rng(3)
    first = rng.normal(size=(20, 6))
    second = first + rng.normal(loc=0.2, scale=0.5, size=(20, 6))
    result = _hotelling_transition(first, second)
    differences = second - first
    mean = differences.mean(axis=0)
    covariance = np.cov(differences, rowvar=False, ddof=1)
    expected_t2 = 20.0 * float(mean @ np.linalg.inv(covariance) @ mean)
    expected_f = (20 - 6) * expected_t2 / (6 * (20 - 1))
    assert result["hotelling_t_squared"] == pytest.approx(expected_t2)
    assert result["f_value"] == pytest.approx(expected_f)
    assert result["p_value_raw"] == pytest.approx(f_distribution.sf(expected_f, 6, 14))
    assert _holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_phase_by_difficulty_interaction_fixture(tmp_path: Path) -> None:
    no_interaction, _ = analyze_depth_cells(
        _three_logs(tmp_path / "plain", _rows(phase_effect=0.25))
    )
    strong_interaction, _ = analyze_depth_cells(
        _three_logs(tmp_path / "interaction", _rows(phase_effect=0.6, interaction=True))
    )
    assert no_interaction["phase_by_difficulty"]["p_value"] > 0.01
    assert strong_interaction["phase_by_difficulty"]["p_value"] < 0.01


def test_analysis_writes_all_canonical_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    analysis, cells = analyze_depth_cells(_three_logs(tmp_path, _rows(phase_effect=0.2)))
    outputs = write_depth_analysis(analysis, cells, tmp_path / "out")
    assert set(outputs) == {
        "combined_cells",
        "analysis_json",
        "analysis_markdown",
        "paper_table",
    }
    assert all(path.is_file() for path in outputs.values())


def test_versioned_jedi_bounds_compute_without_inference_inputs(tmp_path: Path) -> None:
    bounds = {
        "version": "fixture-v1",
        "provenance": "unit-test frozen bounds",
        "epsilon": 1e-6,
        "metrics": {
            key: {
                "direction": "higher" if key in {"delta1", "fscore_5cm"} else "lower",
                "best": 10.0 if key in {"delta1", "fscore_5cm"} else -10.0,
                "worst": -10.0 if key in {"delta1", "fscore_5cm"} else 10.0,
            }
            for key in PAPER_METRIC_KEYS
        },
    }
    bounds_path = tmp_path / "bounds.json"
    bounds_path.write_text(json.dumps(bounds))
    analysis, _ = analyze_depth_cells(
        _three_logs(tmp_path / "logs", _rows(phase_effect=0.2)),
        jedi_bounds_path=bounds_path,
    )
    assert analysis["jedi_status"] == "computed"
    assert 0.0 < analysis["jedi"]["overall"]["score"] <= 1.0
    assert len(analysis["jedi"]["bounds_sha256"]) == 64


def test_duplicate_missing_and_constant_metrics_fail(tmp_path: Path) -> None:
    rows = _rows(phase_effect=0.2)
    duplicate = rows + [dict(rows[0])]
    with pytest.raises(DatasetError, match="Duplicate"):
        analyze_depth_cells(_three_logs(tmp_path / "duplicate", duplicate))

    with pytest.raises(DatasetError, match="Expected 300 cells"):
        analyze_depth_cells(_three_logs(tmp_path / "missing", rows[:-1]))

    for row in rows:
        row["metric:absrel"] = 1.0
    with pytest.raises(DatasetError, match="non-constant"):
        analyze_depth_cells(_three_logs(tmp_path / "constant", rows))

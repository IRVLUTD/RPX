"""Paper-level statistical analysis for RPX D3 tracking cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .exceptions import ConfigError, DatasetError
from .metrics.tracking_paper import PAPER_TRACKING_METRICS
from .paper_depth_analysis import (
    DIFFICULTIES,
    PHASES,
    _holm_adjust,
    _hotelling_transition,
    _interaction_test,
    _nested_effect,
    _phase_contrasts,
)

TRACKING_DIRECTIONS = {
    "mota": "higher",
    "idf1": "higher",
    "hota": "higher",
    "idsw": "lower",
}
_PHASE_NAMES = {0: "clutter", 1: "interaction", 2: "clean"}


def _phase_test(cube: np.ndarray) -> dict[str, float]:
    n_scenes, n_phases, n_metrics = cube.shape
    if n_phases != 3 or n_metrics != len(PAPER_TRACKING_METRICS):
        raise ConfigError(f"Unexpected D3 MANOVA cube shape: {cube.shape}")
    responses = cube.reshape(n_scenes * n_phases, n_metrics)
    subjects = np.repeat(np.eye(n_scenes), n_phases, axis=0)
    phase = np.tile(_phase_contrasts(), (n_scenes, 1))
    return _nested_effect(
        responses,
        subjects,
        np.column_stack((subjects, phase)),
        df_h=2,
    )


def _load_bounds(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    raw = path.read_bytes()
    value = json.loads(raw)
    if not value.get("version") or not value.get("provenance"):
        raise ConfigError("D3 JEDI bounds require version and provenance.")
    epsilon = value.get("epsilon")
    if not isinstance(epsilon, (int, float)) or not 0 < float(epsilon) <= 1:
        raise ConfigError("D3 JEDI epsilon must be in (0, 1].")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(PAPER_TRACKING_METRICS):
        raise ConfigError(
            f"D3 JEDI bounds must define exactly {list(PAPER_TRACKING_METRICS)}."
        )
    for metric, direction in TRACKING_DIRECTIONS.items():
        spec = metrics[metric]
        if spec.get("direction") != direction:
            raise ConfigError(f"D3 JEDI direction mismatch for {metric}.")
        best, worst = spec.get("best"), spec.get("worst")
        if not isinstance(best, (int, float)) or not isinstance(worst, (int, float)):
            raise ConfigError(f"D3 JEDI bounds for {metric} must be numeric.")
        if not np.isfinite([best, worst]).all():
            raise ConfigError(f"D3 JEDI bounds for {metric} must be finite.")
        if (direction == "higher" and best <= worst) or (
            direction == "lower" and best >= worst
        ):
            raise ConfigError(f"D3 JEDI bounds contradict direction for {metric}.")
    return value, hashlib.sha256(raw).hexdigest()


def _jedi(values: Mapping[str, float], bounds: Mapping[str, Any]) -> dict[str, Any]:
    desirabilities: dict[str, float] = {}
    for metric in PAPER_TRACKING_METRICS:
        spec = bounds["metrics"][metric]
        desirability = (values[metric] - spec["worst"]) / (spec["best"] - spec["worst"])
        desirabilities[metric] = float(np.clip(desirability, 0, 1))
    epsilon = float(bounds["epsilon"])
    score = float(
        np.exp(np.mean(np.log([max(value, epsilon) for value in desirabilities.values()])))
    )
    return {"score": score, "desirabilities": desirabilities}


def analyze_tracking_cells(
    paths: Sequence[str | Path],
    *,
    jedi_bounds_path: str | Path | None = None,
) -> tuple[dict[str, Any], "Any"]:
    import pandas as pd

    frames = [pd.read_parquet(path) for path in paths]
    cells = pd.concat(frames, ignore_index=True)
    required = {
        "model",
        "task",
        "split",
        "scene",
        "phase",
        *(f"metric:{metric}" for metric in PAPER_TRACKING_METRICS),
    }
    missing = required - set(cells.columns)
    if missing:
        raise DatasetError(f"D3 cells are missing columns: {sorted(missing)}")
    cells = cells[cells["task"] == "object_tracking"].copy()
    if cells.empty:
        raise DatasetError("No object_tracking cells found.")
    models = set(cells["model"].astype(str))
    if len(models) != 1:
        raise DatasetError(f"D3 analysis requires one model, got {sorted(models)}.")
    if cells.duplicated(["scene", "phase"]).any():
        raise DatasetError("D3 cells contain duplicate scene-phase rows.")
    if len(cells) != 300 or cells["scene"].nunique() != 100:
        raise DatasetError(
            f"D3 analysis requires 300 cells/100 scenes; got {len(cells)}/"
            f"{cells['scene'].nunique()}."
        )

    scenes = sorted(cells["scene"].astype(str).unique())
    raw = np.empty((100, 3, len(PAPER_TRACKING_METRICS)), dtype=np.float64)
    difficulties: list[str] = []
    for scene_index, scene in enumerate(scenes):
        scene_rows = cells[cells["scene"].astype(str) == scene]
        tiers = set(scene_rows["split"].astype(str))
        if len(tiers) != 1 or next(iter(tiers)) not in DIFFICULTIES:
            raise DatasetError(f"Invalid/changing difficulty for {scene}: {tiers}")
        difficulties.append(next(iter(tiers)))
        for phase in range(3):
            phase_rows = scene_rows[scene_rows["phase"].astype(int) == phase]
            if len(phase_rows) != 1:
                raise DatasetError(f"Missing D3 cell for {scene}/phase {phase}.")
            row = phase_rows.iloc[0]
            for metric_index, metric in enumerate(PAPER_TRACKING_METRICS):
                value = float(row[f"metric:{metric}"])
                if not np.isfinite(value):
                    raise DatasetError(f"Non-finite {metric} for {scene}/phase {phase}.")
                raw[scene_index, phase, metric_index] = value

    oriented = raw.copy()
    for index, metric in enumerate(PAPER_TRACKING_METRICS):
        if TRACKING_DIRECTIONS[metric] == "lower":
            oriented[:, :, index] *= -1
    flat = oriented.reshape(-1, len(PAPER_TRACKING_METRICS))
    standard_deviation = flat.std(axis=0, ddof=0)
    if np.any(standard_deviation <= 0):
        raise DatasetError("D3 analysis requires four non-constant metrics.")
    standardized = ((flat - flat.mean(axis=0)) / standard_deviation).reshape(oriented.shape)

    transitions: list[dict[str, Any]] = []
    for label, first, second in (
        ("clutter_to_interaction", 0, 1),
        ("interaction_to_clean", 1, 2),
        ("clutter_to_clean", 0, 2),
    ):
        transitions.append(
            {
                "transition": label,
                **_hotelling_transition(standardized[:, first], standardized[:, second]),
            }
        )
    adjusted = _holm_adjust([value["p_value_raw"] for value in transitions])
    for value, adjusted_p in zip(transitions, adjusted, strict=True):
        value["p_value_holm"] = adjusted_p

    tier_array = np.asarray(difficulties)
    per_tier = {
        tier: {
            "n_scenes": int(np.sum(tier_array == tier)),
            **_phase_test(standardized[tier_array == tier]),
        }
        for tier in DIFFICULTIES
    }
    phase_means = {
        _PHASE_NAMES[phase]: {
            metric: float(raw[:, phase, index].mean())
            for index, metric in enumerate(PAPER_TRACKING_METRICS)
        }
        for phase in range(3)
    }
    overall_means = {
        metric: float(raw[:, :, index].mean())
        for index, metric in enumerate(PAPER_TRACKING_METRICS)
    }
    bounds, bounds_sha = _load_bounds(Path(jedi_bounds_path) if jedi_bounds_path else None)
    if bounds is None:
        jedi: dict[str, Any] = {"status": "bounds_missing"}
    else:
        jedi = {
            "status": "computed",
            "bounds_version": bounds["version"],
            "bounds_sha256": bounds_sha,
            "overall": _jedi(overall_means, bounds),
            "per_phase": {
                phase: _jedi(values, bounds) for phase, values in phase_means.items()
            },
        }

    analysis = {
        "schema_version": "rpx-d3-paper-analysis-v1",
        "model": next(iter(models)),
        "metrics": list(PAPER_TRACKING_METRICS),
        "directions": TRACKING_DIRECTIONS,
        "n_scenes": 100,
        "n_cells": 300,
        "overall_phase_manova": _phase_test(standardized),
        "transitions": transitions,
        "per_tier": per_tier,
        "phase_by_difficulty": _interaction_test(standardized, difficulties),
        "metric_means": {"overall": overall_means, "per_phase": phase_means},
        "jedi": jedi,
        "jedi_status": jedi["status"],
    }
    return analysis, cells


def write_tracking_analysis(
    analysis: Mapping[str, Any],
    cells: Any,
    output_dir: str | Path,
) -> dict[str, Path]:
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined_cells": output / "combined_cells.parquet",
        "json": output / "paper_analysis.json",
        "markdown": output / "paper_analysis.md",
        "table": output / "paper_table.csv",
    }
    cells.to_parquet(paths["combined_cells"], index=False)
    paths["json"].write_text(
        json.dumps(dict(analysis), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overall = analysis["overall_phase_manova"]
    table = {
        "model": analysis["model"],
        "phi": overall["phi_wilks"],
        "wilks_lambda": overall["wilks_lambda"],
        "p_value": overall["p_value"],
        "partial_eta_squared": overall["partial_eta_squared"],
        "jedi_status": analysis["jedi_status"],
    }
    pd.DataFrame([table]).to_csv(paths["table"], index=False)
    lines = [
        "# RPX D3 paper analysis",
        "",
        f"- Model: `{analysis['model']}`",
        f"- Cells: {analysis['n_cells']} (100 scenes × 3 phases)",
        f"- Φ: {overall['phi_wilks']:.6g}",
        f"- Wilks’ Λ: {overall['wilks_lambda']:.6g}",
        f"- Rao p: {overall['p_value']:.6g}",
        f"- Partial η²: {overall['partial_eta_squared']:.6g}",
        f"- JEDI: `{analysis['jedi_status']}`",
        "",
        "## Per-phase means",
        "",
        "| Phase | MOTA | IDF1 | HOTA | ID switches |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        values = analysis["metric_means"]["per_phase"][phase]
        lines.append(
            f"| {phase} | {values['mota']:.6g} | {values['idf1']:.6g} | "
            f"{values['hota']:.6g} | {values['idsw']:.6g} |"
        )
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths

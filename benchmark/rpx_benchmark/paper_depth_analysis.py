"""Paper-level statistical analysis for the six-metric RPX D1-F cell log."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cell_log import merge_cells, write_cells
from .exceptions import ConfigError, DatasetError
from .metrics.depth_paper import D1_CALIBRATION, PAPER_METRIC_KEYS

PHASES = ("clutter", "interaction", "clean")
DIFFICULTIES = ("easy", "medium", "hard")
METRIC_DIRECTIONS = {
    "absrel": "lower",
    "rmse": "lower",
    "silog": "lower",
    "delta1": "higher",
    "irmse": "lower",
    "fscore_5cm": "higher",
}
_PHASE_ALIASES = {
    "0": "clutter",
    "clu": "clutter",
    "clutter": "clutter",
    "1": "interaction",
    "int": "interaction",
    "interaction": "interaction",
    "2": "clean",
    "cln": "clean",
    "clean": "clean",
}


def _phase_contrasts() -> np.ndarray:
    return np.array(
        [
            [-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(6.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(6.0)],
            [0.0, 2.0 / math.sqrt(6.0)],
        ]
    )


def _residual_ssp(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, int]:
    rank = int(np.linalg.matrix_rank(design))
    df = len(y) - rank
    if df <= 0:
        raise ConfigError("MANOVA design has no residual degrees of freedom.")
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return residual.T @ residual, df


def _effect_summary(h: np.ndarray, e: np.ndarray, *, df_h: int, df_e: int) -> dict[str, float]:
    from scipy.stats import f as f_distribution

    h = (h + h.T) / 2.0
    e = (e + e.T) / 2.0
    response_dim = e.shape[0]
    effect_s = min(response_dim, df_h)

    # An exactly phase-invariant cube has H=0 and E=0 after subject effects.
    # This is a well-defined null effect (Phi=1), even though E has no inverse.
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(e))) * 1_000
    if float(np.linalg.norm(h)) <= tolerance:
        p = float(response_dim)
        q = float(df_h)
        numerator = p * p * q * q - 4.0
        denominator = p * p + q * q - 5.0
        rao_t = math.sqrt(numerator / denominator) if numerator > 0 and denominator > 0 else 1.0
        df1 = response_dim * df_h
        df2 = rao_t * (df_e - (response_dim - df_h + 1.0) / 2.0) - (df1 - 2.0) / 2.0
        if df2 <= 0:
            raise ConfigError("Rao F approximation produced non-positive denominator df.")
        return {
            "wilks_lambda": 1.0,
            "rao_f": 0.0,
            "df1": float(df1),
            "df2": float(df2),
            "p_value": 1.0,
            "partial_eta_squared": 0.0,
            "phi_wilks": 1.0,
            "pillai_trace": 0.0,
            "phi_pillai": 1.0,
            "pillai_minus_wilks_phi": 0.0,
        }
    try:
        chol = np.linalg.cholesky(e)
    except np.linalg.LinAlgError as exc:
        raise ConfigError(
            "MANOVA residual SSP matrix is singular.",
            hint="Check for constant/redundant metrics or incomplete scene-phase cells.",
        ) from exc
    transformed = np.linalg.solve(chol, h)
    transformed = np.linalg.solve(chol, transformed.T).T
    eigenvalues = np.linalg.eigvalsh((transformed + transformed.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    wilks = float(np.prod(1.0 / (1.0 + eigenvalues)))
    pillai = float(np.sum(eigenvalues / (1.0 + eigenvalues)))
    phi_wilks = float(wilks ** (1.0 / effect_s))
    eta_wilks = float(1.0 - phi_wilks)
    phi_pillai = float(1.0 - pillai / effect_s)

    # Rao's F approximation for Wilks' lambda.
    p = float(response_dim)
    q = float(df_h)
    numerator = p * p * q * q - 4.0
    denominator = p * p + q * q - 5.0
    rao_t = math.sqrt(numerator / denominator) if numerator > 0 and denominator > 0 else 1.0
    df1 = int(response_dim * df_h)
    df2 = rao_t * (df_e - (response_dim - df_h + 1.0) / 2.0) - (df1 - 2.0) / 2.0
    if df2 <= 0:
        raise ConfigError("Rao F approximation produced non-positive denominator df.")
    lambda_root = wilks ** (1.0 / rao_t)
    f_value = 0.0 if lambda_root >= 1.0 else ((1.0 - lambda_root) / lambda_root) * (df2 / df1)
    p_value = float(f_distribution.sf(f_value, df1, df2))
    return {
        "wilks_lambda": wilks,
        "rao_f": float(f_value),
        "df1": float(df1),
        "df2": float(df2),
        "p_value": p_value,
        "partial_eta_squared": eta_wilks,
        "phi_wilks": phi_wilks,
        "pillai_trace": pillai,
        "phi_pillai": phi_pillai,
        "pillai_minus_wilks_phi": float(phi_pillai - phi_wilks),
    }


def _nested_effect(
    y: np.ndarray, reduced: np.ndarray, full: np.ndarray, *, df_h: int
) -> dict[str, float]:
    e_reduced, _ = _residual_ssp(y, reduced)
    e_full, df_e = _residual_ssp(y, full)
    h = e_reduced - e_full
    return _effect_summary(h, e_full, df_h=df_h, df_e=df_e)


def _repeated_phase_test(cube: np.ndarray) -> dict[str, float]:
    n_scenes, n_phases, n_metrics = cube.shape
    if n_phases != 3 or n_metrics != len(PAPER_METRIC_KEYS):
        raise ConfigError(f"Unexpected MANOVA cube shape: {cube.shape}")
    y = cube.reshape(n_scenes * n_phases, n_metrics)
    subjects = np.repeat(np.eye(n_scenes), n_phases, axis=0)
    phase = np.tile(_phase_contrasts(), (n_scenes, 1))
    return _nested_effect(y, subjects, np.column_stack((subjects, phase)), df_h=2)


def _hotelling_transition(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    from scipy.stats import f as f_distribution

    diff = b - a
    n, k = diff.shape
    if n <= k:
        raise ConfigError(f"Hotelling T2 requires scenes > metrics; got {n} <= {k}.")
    mean = np.mean(diff, axis=0)
    if np.allclose(diff, 0.0, rtol=0.0, atol=1e-12):
        return {
            "hotelling_t_squared": 0.0,
            "f_value": 0.0,
            "df1": float(k),
            "df2": float(n - k),
            "p_value_raw": 1.0,
            "wilks_lambda": 1.0,
            "phi": 1.0,
        }
    covariance = np.cov(diff, rowvar=False, ddof=1)
    try:
        solved = np.linalg.solve(covariance, mean)
    except np.linalg.LinAlgError as exc:
        raise ConfigError("Hotelling covariance matrix is singular.") from exc
    t_squared = float(n * mean @ solved)
    f_value = float((n - k) * t_squared / (k * (n - 1)))
    p_value = float(f_distribution.sf(f_value, k, n - k))
    wilks = float(1.0 / (1.0 + t_squared / (n - 1)))
    return {
        "hotelling_t_squared": t_squared,
        "f_value": f_value,
        "df1": float(k),
        "df2": float(n - k),
        "p_value_raw": p_value,
        "wilks_lambda": wilks,
        "phi": wilks,
    }


def _holm_adjust(values: Sequence[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _interaction_test(cube: np.ndarray, difficulties: Sequence[str]) -> dict[str, float]:
    n_scenes, n_phases, n_metrics = cube.shape
    y = cube.reshape(n_scenes * n_phases, n_metrics)
    subjects = np.repeat(np.eye(n_scenes), n_phases, axis=0)
    phase = np.tile(_phase_contrasts(), (n_scenes, 1))
    difficulty_index = np.array([DIFFICULTIES.index(value) for value in difficulties])
    difficulty = _phase_contrasts()[difficulty_index]
    difficulty_long = np.repeat(difficulty, n_phases, axis=0)
    interaction = (phase[:, :, None] * difficulty_long[:, None, :]).reshape(len(y), 4)
    reduced = np.column_stack((subjects, phase))
    full = np.column_stack((reduced, interaction))
    return _nested_effect(y, reduced, full, df_h=4)


def _load_jedi_bounds(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not payload.get("version") or not payload.get("provenance"):
        raise ConfigError("JEDI bounds require non-empty version and provenance.")
    epsilon = payload.get("epsilon")
    if not isinstance(epsilon, (int, float)) or not 0.0 < float(epsilon) <= 1.0:
        raise ConfigError("JEDI epsilon must be numeric in (0, 1].")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(PAPER_METRIC_KEYS):
        raise ConfigError(f"JEDI bounds must define exactly: {list(PAPER_METRIC_KEYS)}")
    for key in PAPER_METRIC_KEYS:
        spec = metrics[key]
        if spec.get("direction") != METRIC_DIRECTIONS[key]:
            raise ConfigError(f"JEDI direction mismatch for {key}.")
        if not all(isinstance(spec.get(field), (int, float)) for field in ("best", "worst")):
            raise ConfigError(f"JEDI {key} requires numeric best and worst bounds.")
        best = float(spec["best"])
        worst = float(spec["worst"])
        if not np.isfinite([best, worst]).all():
            raise ConfigError(f"JEDI {key} bounds must be finite.")
        if (spec["direction"] == "lower" and best >= worst) or (
            spec["direction"] == "higher" and best <= worst
        ):
            raise ConfigError(f"JEDI {key} best/worst bounds contradict its direction.")
    return payload, hashlib.sha256(raw).hexdigest()


def _jedi_score(values: Mapping[str, float], bounds: Mapping[str, Any]) -> dict[str, Any]:
    epsilon = float(bounds["epsilon"])
    desirabilities: dict[str, float] = {}
    for key in PAPER_METRIC_KEYS:
        spec = bounds["metrics"][key]
        value = float(values[key])
        desirability = (value - float(spec["worst"])) / (float(spec["best"]) - float(spec["worst"]))
        desirabilities[key] = float(np.clip(desirability, 0.0, 1.0))
    score = float(
        np.exp(np.mean(np.log([max(value, epsilon) for value in desirabilities.values()])))
    )
    return {"score": score, "desirabilities": desirabilities}


def analyze_depth_cells(
    paths: Sequence[str | Path],
    *,
    jedi_bounds_path: str | Path | None = None,
    expected_scenes: int = 100,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate, merge and analyze three RPX D1-F split cell logs."""
    rows = merge_cells(Path(path) for path in paths)
    rows = [
        row
        for row in rows
        if row.get("task") == "monocular_depth" and int(row.get("frame_budget") or 0) == 0
    ]
    if not rows:
        raise DatasetError("No monocular_depth cells found in the supplied logs.")
    model_names = {str(row.get("model_name")) for row in rows}
    if len(model_names) != 1 or model_names == {"None"} or model_names == {""}:
        raise DatasetError(f"Paper analysis requires exactly one model; got {sorted(model_names)}")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        scene = str(row.get("scene_id") or "")
        phase_raw = str(row.get("phase") or "").lower()
        phase = _PHASE_ALIASES.get(phase_raw)
        if not scene or phase is None:
            raise DatasetError(f"Invalid scene/phase cell: {scene!r}/{phase_raw!r}")
        key = (scene, phase)
        if key in by_key:
            raise DatasetError(f"Duplicate scene-phase cell: {key}")
        normalized = dict(row)
        normalized["phase"] = phase
        by_key[key] = normalized

    scenes = sorted({scene for scene, _phase in by_key})
    if len(scenes) != expected_scenes:
        raise DatasetError(f"Expected {expected_scenes} scenes; found {len(scenes)}.")
    if len(by_key) != expected_scenes * 3:
        raise DatasetError(f"Expected {expected_scenes * 3} cells; found {len(by_key)}.")

    raw_cube = np.empty((len(scenes), 3, len(PAPER_METRIC_KEYS)), dtype=np.float64)
    difficulties: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        scene_difficulties = set()
        for phase_index, phase in enumerate(PHASES):
            if (scene, phase) not in by_key:
                raise DatasetError(f"Missing scene-phase cell: {scene}/{phase}")
            row = by_key[(scene, phase)]
            difficulty = str(row.get("difficulty") or "").lower()
            if difficulty not in DIFFICULTIES:
                raise DatasetError(f"Invalid difficulty for {scene}/{phase}: {difficulty!r}")
            scene_difficulties.add(difficulty)
            for metric_index, metric in enumerate(PAPER_METRIC_KEYS):
                value = row.get(f"metric:{metric}")
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise DatasetError(f"Missing/non-finite {metric} for {scene}/{phase}.")
                raw_cube[scene_index, phase_index, metric_index] = float(value)
            normalized_rows.append(row)
        if len(scene_difficulties) != 1:
            raise DatasetError(f"Difficulty changes across phases for scene {scene}.")
        difficulties.append(scene_difficulties.pop())

    oriented = raw_cube.copy()
    for index, metric in enumerate(PAPER_METRIC_KEYS):
        if METRIC_DIRECTIONS[metric] == "lower":
            oriented[:, :, index] *= -1.0
    flat = oriented.reshape(-1, len(PAPER_METRIC_KEYS))
    std = np.std(flat, axis=0, ddof=0)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise DatasetError("Paper analysis requires six non-constant finite metrics.")
    standardized = ((flat - np.mean(flat, axis=0)) / std).reshape(oriented.shape)

    overall = _repeated_phase_test(standardized)
    transitions: list[dict[str, Any]] = []
    for label, first, second in (
        ("clutter_to_interaction", 0, 1),
        ("interaction_to_clean", 1, 2),
        ("clutter_to_clean", 0, 2),
    ):
        transitions.append(
            {
                "transition": label,
                **_hotelling_transition(standardized[:, first, :], standardized[:, second, :]),
            }
        )
    adjusted = _holm_adjust([item["p_value_raw"] for item in transitions])
    for item, value in zip(transitions, adjusted, strict=True):
        item["p_value_holm"] = value

    per_tier: dict[str, Any] = {}
    difficulty_array = np.asarray(difficulties)
    for tier in DIFFICULTIES:
        subset = standardized[difficulty_array == tier]
        per_tier[tier] = {"n_scenes": int(len(subset)), **_repeated_phase_test(subset)}

    phase_means = {
        phase: {
            metric: float(np.mean(raw_cube[:, phase_index, metric_index]))
            for metric_index, metric in enumerate(PAPER_METRIC_KEYS)
        }
        for phase_index, phase in enumerate(PHASES)
    }
    overall_means = {
        metric: float(np.mean(raw_cube[:, :, metric_index]))
        for metric_index, metric in enumerate(PAPER_METRIC_KEYS)
    }

    bounds_path = Path(jedi_bounds_path) if jedi_bounds_path else None
    bounds, bounds_sha = _load_jedi_bounds(bounds_path)
    if bounds is None:
        jedi: dict[str, Any] = {"status": "bounds_missing"}
    else:
        jedi = {
            "status": "computed",
            "bounds_version": bounds["version"],
            "bounds_sha256": bounds_sha,
            "overall": _jedi_score(overall_means, bounds),
            "per_phase": {
                phase: _jedi_score(values, bounds) for phase, values in phase_means.items()
            },
        }

    analysis = {
        "schema_version": "rpx-d1f-paper-analysis-v1",
        "model_name": next(iter(model_names)),
        "metrics": list(PAPER_METRIC_KEYS),
        "metric_directions": METRIC_DIRECTIONS,
        "n_scenes": len(scenes),
        "n_cells": len(by_key),
        "calibration": D1_CALIBRATION.to_dict(),
        "overall_phase_manova": overall,
        "transitions": transitions,
        "per_tier": per_tier,
        "phase_by_difficulty": _interaction_test(standardized, difficulties),
        "metric_means": {"overall": overall_means, "per_phase": phase_means},
        "jedi": jedi,
        "jedi_status": jedi["status"],
        "provenance": dict(provenance or {}),
    }
    return analysis, normalized_rows


def write_depth_analysis(
    analysis: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined = out / "combined_cells.parquet"
    write_cells(cells, combined, format="parquet")
    json_path = out / "paper_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    table_path = out / "paper_table.csv"
    flat: dict[str, Any] = {
        "model_name": analysis["model_name"],
        "phi": analysis["overall_phase_manova"]["phi_wilks"],
        "wilks_lambda": analysis["overall_phase_manova"]["wilks_lambda"],
        "rao_f": analysis["overall_phase_manova"]["rao_f"],
        "p_value": analysis["overall_phase_manova"]["p_value"],
        "partial_eta_squared": analysis["overall_phase_manova"]["partial_eta_squared"],
        "phi_pillai": analysis["overall_phase_manova"]["phi_pillai"],
        "wilks_pillai_gap": analysis["overall_phase_manova"]["pillai_minus_wilks_phi"],
        "jedi_status": analysis["jedi"]["status"],
    }
    if analysis["jedi"]["status"] == "computed":
        flat["jedi"] = analysis["jedi"]["overall"]["score"]
    for phase, values in analysis["metric_means"]["per_phase"].items():
        for metric, value in values.items():
            flat[f"{metric}_{phase}"] = value
    for item in analysis["transitions"]:
        flat[f"phi_{item['transition']}"] = item["phi"]
        flat[f"hotelling_t2_{item['transition']}"] = item["hotelling_t_squared"]
        flat[f"p_raw_{item['transition']}"] = item["p_value_raw"]
        flat[f"p_holm_{item['transition']}"] = item["p_value_holm"]
    for tier, values in analysis["per_tier"].items():
        for key in ("n_scenes", "wilks_lambda", "phi_wilks", "p_value", "partial_eta_squared"):
            flat[f"{key}_{tier}"] = values[key]
    for key in ("wilks_lambda", "rao_f", "p_value", "partial_eta_squared"):
        flat[f"phase_by_difficulty_{key}"] = analysis["phase_by_difficulty"][key]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

    markdown_path = out / "paper_analysis.md"
    overall = analysis["overall_phase_manova"]
    lines = [
        "# RPX D1-F Paper Analysis",
        "",
        f"- Model: `{analysis['model_name']}`",
        f"- Complete cells: {analysis['n_cells']} ({analysis['n_scenes']} scenes x 3 phases)",
        f"- Wilks Lambda: {overall['wilks_lambda']:.6g}",
        f"- Phi: {overall['phi_wilks']:.6g}",
        f"- Rao F({overall['df1']:.0f}, {overall['df2']:.3f}): "
        f"{overall['rao_f']:.6g}; p={overall['p_value']:.6g}",
        f"- Partial eta squared: {overall['partial_eta_squared']:.6g}",
        f"- Pillai Phi: {overall['phi_pillai']:.6g}; gap={overall['pillai_minus_wilks_phi']:.6g}",
        f"- JEDI: `{analysis['jedi']['status']}`",
        "",
        "## Phase transitions",
        "",
        "| Transition | Hotelling T2 | Wilks Phi | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in analysis["transitions"]:
        lines.append(
            f"| {item['transition']} | {item['hotelling_t_squared']:.6g} | "
            f"{item['phi']:.6g} | {item['p_value_raw']:.6g} | "
            f"{item['p_value_holm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Difficulty tiers",
            "",
            "| Tier | Scenes | Wilks Lambda | Phi | p | Partial eta squared |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tier in DIFFICULTIES:
        values = analysis["per_tier"][tier]
        lines.append(
            f"| {tier} | {values['n_scenes']} | {values['wilks_lambda']:.6g} | "
            f"{values['phi_wilks']:.6g} | {values['p_value']:.6g} | "
            f"{values['partial_eta_squared']:.6g} |"
        )
    interaction = analysis["phase_by_difficulty"]
    lines.extend(
        [
            "",
            "## Phase by difficulty",
            "",
            f"Wilks Lambda={interaction['wilks_lambda']:.6g}; "
            f"Rao F({interaction['df1']:.0f}, {interaction['df2']:.3f})="
            f"{interaction['rao_f']:.6g}; p={interaction['p_value']:.6g}; "
            f"partial eta squared={interaction['partial_eta_squared']:.6g}.",
            "",
            "## Phase means",
            "",
            "| Phase | " + " | ".join(PAPER_METRIC_KEYS) + " |",
            "|---|" + "---:|" * len(PAPER_METRIC_KEYS),
        ]
    )
    for phase in PHASES:
        values = analysis["metric_means"]["per_phase"][phase]
        lines.append(
            "| "
            + phase
            + " | "
            + " | ".join(f"{values[key]:.6g}" for key in PAPER_METRIC_KEYS)
            + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    return {
        "combined_cells": combined,
        "analysis_json": json_path,
        "analysis_markdown": markdown_path,
        "paper_table": table_path,
    }

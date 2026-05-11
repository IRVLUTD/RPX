#!/usr/bin/env python3
"""Stage 2 of the RPX splits pipeline: turn the raw 27-feature table into
Easy / Medium / Hard tertile assignments per (scene, phase).

Reads the CSV produced by ``build_esd_splits.py`` (Stage 1), computes five
candidate scoring methods side-by-side, and writes the splits manifest plus
flat ``easy.txt`` / ``medium.txt`` / ``hard.txt`` tier lists for downstream
task runners.

Two weighting tiers (paper §3.2):

- ``uniform_v1`` — wᵢ = 1/F. Always reproducible from data alone. Default.
- ``mi_v1``     — MI-derived weights from a calibration model set. Activated
  by ``--weights-mi <path>``. The artifact contract is documented in
  :func:`rpx_benchmark.data.esd_scoring.load_mi_weights`.

All five methods are computed regardless of the chosen primary so a future
researcher can pivot without re-running. The primary method (default
``mean_pn``, override via ``--primary``) determines which method's tertiles
populate ``easy.txt`` / ``medium.txt`` / ``hard.txt``.

Usage
-----

::

    # Default: uniform weights, mean_pn primary
    python build_difficulty_splits.py \\
        --features benchmark/data/splits/phase_esd_splits.csv

    # MI weights once calibration models are available
    python build_difficulty_splits.py \\
        --features benchmark/data/splits/phase_esd_splits.csv \\
        --weights-mi paper-submission/.../mi_weights_v1.json \\
        --primary mean_pn
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402
from rpx_benchmark.data.esd_scoring import (  # noqa: E402
    CONFIDENCE_LABELS,
    DEFAULT_EFFORT_ALPHA,
    DEFAULT_PRIMARY,
    FEATURE_CATEGORIES,
    SCORING_METHODS,
    TERTILE_LABELS,
    ScoringProvenance,
    aggregate_to_scene_splits,
    all_methods,
    compute_confidence,
    compute_consensus_tier,
    compute_per_category_scores,
    effort_stratified_weights,
    load_mi_weights,
    percentile_normalize,
    percentile_rank,
    sha256_of_file,
    tertile_cut,
    uniform_weights,
)
from rpx_benchmark.logging_utils import configure_logging, get_logger  # noqa: E402

log = get_logger("build_difficulty_splits")

SCHEMA_VERSION = 2  # v2 adds consensus_tier, confidence, per_category to every row
FEATURE_SET_VERSION = "v1.0"  # bump when FEATURE_NAMES changes

# Phase index → human-readable phase name (matches loader.py).
_PHASE_NAMES = {0: "clutter", 1: "interaction", 2: "clean"}


# --------------------------------------------------------------------------- #
# CSV reader (skips the schema-version comment header from Stage 1)
# --------------------------------------------------------------------------- #

def _read_features_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open() as f:
        # Stage 1 prepends a "# schema_version=..." comment line.
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.startswith("#"):
                f.seek(pos)
                break
        rows.extend(csv.DictReader(f))
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Core build
# --------------------------------------------------------------------------- #

def build_splits(
    csv_path: Path,
    out_dir: Path,
    *,
    primary: str = DEFAULT_PRIMARY,
    weights_mi_path: Optional[Path] = None,
    seed: int = 0,
    confidence_n_perturb: int = 1000,
    weighting: str = "effort_stratified",
    effort_alpha: float = DEFAULT_EFFORT_ALPHA,
) -> Dict[str, object]:
    """Run Stage 2 end-to-end. Returns the JSON payload (also written to disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_features_csv(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    # Feature matrix in canonical FEATURE_NAMES order — guards against the CSV
    # being shuffled or missing columns.
    missing = [n for n in FEATURE_NAMES if n not in rows[0]]
    if missing:
        raise SystemExit(f"feature columns missing from CSV: {missing}")

    feature_matrix = np.asarray(
        [[float(r[name]) for name in FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )
    log.info("loaded %d rows × %d features from %s",
             feature_matrix.shape[0], feature_matrix.shape[1], csv_path)

    # ── Drop any rows with all-zero features (failed extractions) ──────────
    nonzero = feature_matrix.sum(axis=1) > 0
    if (~nonzero).any():
        n_drop = int((~nonzero).sum())
        log.warning("dropping %d rows with all-zero features (failed extractions)", n_drop)
        feature_matrix = feature_matrix[nonzero]
        rows = [r for r, ok in zip(rows, nonzero) if ok]

    # ── Weights ───────────────────────────────────────────────────────────
    if weights_mi_path:
        weights_dict, mi_metadata = load_mi_weights(weights_mi_path, list(FEATURE_NAMES))
        scoring_version = "mi_v1"
        log.info("loaded MI weights from %s (calibration_model_set=%s)",
                 weights_mi_path, mi_metadata.get("calibration_model_set", "unknown"))
    elif weighting == "uniform":
        weights_dict = uniform_weights(list(FEATURE_NAMES))
        mi_metadata = {}
        scoring_version = "uniform_v1"
        log.info("using uniform weights (1/%d per feature)", len(FEATURE_NAMES))
    elif weighting == "effort_stratified":
        weights_dict = effort_stratified_weights(list(FEATURE_NAMES), alpha=effort_alpha)
        mi_metadata = {"effort_alpha": effort_alpha}
        scoring_version = "effort_stratified_v1"
        log.info("using effort-stratified weights (alpha=%.3f) — "
                 "iter features = %.4f each, others = %.4f each",
                 effort_alpha,
                 weights_dict["iter_mean"],
                 weights_dict["depth_invalid"])
    else:
        raise SystemExit(f"unknown --weighting choice: {weighting!r}; "
                         f"expected 'effort_stratified', 'uniform', or use --weights-mi")

    weights_vec = np.asarray([weights_dict[name] for name in FEATURE_NAMES], dtype=np.float64)

    # ── Percentile-normalise ──────────────────────────────────────────────
    pn = percentile_normalize(feature_matrix)

    # ── Per-feature contributions for the primary method ──────────────────
    # Only meaningful for the weighted-mean methods; informational for others.
    contributions = pn * weights_vec[None, :]

    # ── All methods, side-by-side ─────────────────────────────────────────
    if primary not in SCORING_METHODS:
        raise SystemExit(f"--primary must be one of {SCORING_METHODS}, got {primary!r}")
    method_scores = all_methods(pn, weights_vec, seed=seed)
    available_methods = list(method_scores)
    if primary not in available_methods:
        raise SystemExit(
            f"primary method {primary!r} unavailable in this run "
            f"(scikit-learn missing?). Available: {available_methods}"
        )

    method_percentiles = {m: percentile_rank(s) for m, s in method_scores.items()}
    method_tiers = {m: tertile_cut(s) for m, s in method_scores.items()}
    primary_tier = method_tiers[primary]

    # ── Structured-metric extras (data story §S4 + §S5 + consensus) ───────
    # `consensus_tier`: majority vote across all available methods.
    consensus = compute_consensus_tier(method_tiers)
    # `confidence`: per-row weight-perturbation flip rate → high/medium/low.
    log.info("computing per-row confidence from %d Dirichlet weight samples...",
             confidence_n_perturb)
    flip_rate, confidence_label = compute_confidence(
        pn, n_perturb=confidence_n_perturb, seed=seed,
    )
    # `per_category`: 8-d sub-score vector per row.
    per_cat = compute_per_category_scores(pn, list(FEATURE_NAMES))

    # ── Summary stats ─────────────────────────────────────────────────────
    tier_counts = dict(Counter(primary_tier))
    consensus_counts = dict(Counter(consensus))
    confidence_counts = dict(Counter(confidence_label))
    primary_vs_consensus_match = int(np.sum(primary_tier == consensus))
    phase_composition: Dict[str, Dict[str, int]] = {t: {p: 0 for p in _PHASE_NAMES.values()}
                                                     for t in TERTILE_LABELS}
    for r, t in zip(rows, primary_tier):
        phase_name = _PHASE_NAMES[int(r["phase"])]
        phase_composition[t][phase_name] += 1

    # ── Provenance ────────────────────────────────────────────────────────
    prov = ScoringProvenance(
        input_sha256=sha256_of_file(csv_path),
        feature_set_version=FEATURE_SET_VERSION,
        scoring_version=scoring_version,
        primary_method=primary,
        weights=weights_dict,
        feature_names=list(FEATURE_NAMES),
        n_entries=len(rows),
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script_version=_git_commit(),
        extra={"mi_metadata": mi_metadata} if mi_metadata else {},
    )

    # ── Build payload ─────────────────────────────────────────────────────
    payload: Dict[str, object] = {
        "schema_version":   SCHEMA_VERSION,
        "primary_method":   primary,
        "scoring_version":  scoring_version,
        "available_methods": available_methods,
        "tertile_labels":   list(TERTILE_LABELS),
        "confidence_labels": list(CONFIDENCE_LABELS),
        "feature_categories": {k: list(v) for k, v in FEATURE_CATEGORIES.items()},
        "provenance":       prov.to_dict(),
        "summary": {
            "n_entries":                  len(rows),
            "tertile_counts":             {t: tier_counts.get(t, 0) for t in TERTILE_LABELS},
            "consensus_tier_counts":      {t: consensus_counts.get(t, 0) for t in TERTILE_LABELS},
            "confidence_counts":          {c: confidence_counts.get(c, 0)
                                            for c in CONFIDENCE_LABELS},
            "primary_vs_consensus_match": primary_vs_consensus_match,
            "primary_vs_consensus_match_pct": round(
                100.0 * primary_vs_consensus_match / max(len(rows), 1), 2,
            ),
            "phase_composition":          phase_composition,
        },
        "phases": {},
    }

    for i, r in enumerate(rows):
        key = f"{r['scene_id']}.phase{r['phase']}"
        payload["phases"][key] = {  # type: ignore[index]
            "scene_id":      r["scene_id"],
            "phase":         int(r["phase"]),
            "n_frames_used": int(r["n_frames_used"]),
            "raw_features":  {name: float(r[name]) for name in FEATURE_NAMES},
            "pn_features":   {name: float(pn[i, j]) for j, name in enumerate(FEATURE_NAMES)},
            "contributions": {name: float(contributions[i, j]) for j, name in enumerate(FEATURE_NAMES)},
            "scores":        {m: float(method_scores[m][i]) for m in available_methods},
            "percentiles":   {m: float(method_percentiles[m][i]) for m in available_methods},
            "tiers":         {m: str(method_tiers[m][i]) for m in available_methods},
            # ── Structured-metric fields (the v2 deliverable) ────────────
            "tier":           str(primary_tier[i]),
            "score":          float(method_scores[primary][i]),
            "percentile":     float(method_percentiles[primary][i]),
            "consensus_tier": str(consensus[i]),
            "confidence":     str(confidence_label[i]),
            "weight_flip_rate": float(flip_rate[i]),
            "per_category":   {cat: float(per_cat[cat][i]) for cat in per_cat},
        }

    # ── Write outputs ─────────────────────────────────────────────────────
    json_path = out_dir / "phase_difficulty.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    log.info("wrote %d entries → %s", len(rows), json_path)

    csv_path_out = out_dir / "phase_difficulty.csv"
    _write_csv(csv_path_out, rows, method_scores, method_percentiles, method_tiers,
               primary, available_methods,
               consensus=consensus, confidence_label=confidence_label,
               flip_rate=flip_rate, per_cat=per_cat)
    log.info("wrote flat CSV → %s", csv_path_out)

    for tier in TERTILE_LABELS:
        tier_path = out_dir / f"{tier}.txt"
        keys = [f"{r['scene_id']}.phase{r['phase']}"
                for r, t in zip(rows, primary_tier) if t == tier]
        tier_path.write_text("\n".join(keys) + ("\n" if keys else ""))
        log.info("wrote %d %s entries → %s", len(keys), tier, tier_path)

    prov_path = out_dir / "splits_provenance.json"
    prov_path.write_text(json.dumps(prov.to_dict(), indent=2))
    log.info("wrote provenance → %s", prov_path)

    # ── Per-scene rollup: the benchmark-consumer deliverable ──────────────
    # All 3 phases of a scene stay together (STR requirement). We aggregate
    # the per-(scene, phase) primary scores to a per-scene mean and tertile
    # the resulting N scenes (Easy and Medium = ⌊N/3⌋ each, Hard = remainder).
    # Per-phase tier + score info is preserved per scene so downstream
    # analysts can ask "which phase is hardest for this scene?" without
    # cross-referencing phase_difficulty.json.
    scene_id_per_row      = np.asarray([r["scene_id"] for r in rows])
    phase_per_row         = np.asarray([int(r["phase"]) for r in rows])
    primary_score_per_row = method_scores[primary]
    primary_tier_per_row  = method_tiers[primary]
    scene_rollup = aggregate_to_scene_splits(
        scene_id_per_row,
        primary_score_per_row,
        phases=phase_per_row,
        phase_tiers=primary_tier_per_row,
    )
    scene_splits_payload = {
        "schema_version":  SCHEMA_VERSION,
        "primary_method":  primary,
        "scoring_version": scoring_version,
        "n_scenes":        sum(len(v) for v in scene_rollup["splits"].values()),
        "tier_counts":     {t: len(v) for t, v in scene_rollup["splits"].items()},
        "aggregation":     "mean of per-phase primary scores",
        "splits":          scene_rollup["splits"],
        "scene_detail":    scene_rollup["scene_detail"],
        "provenance":      prov.to_dict(),
    }
    scene_splits_path = out_dir / "scene_splits.json"
    scene_splits_path.write_text(json.dumps(scene_splits_payload, indent=2, sort_keys=False))
    log.info("wrote scene splits → %s  (counts: %s)",
             scene_splits_path, scene_splits_payload["tier_counts"])

    return payload


def _write_csv(
    path: Path, rows: List[Dict[str, str]],
    method_scores: Dict[str, np.ndarray],
    method_percentiles: Dict[str, np.ndarray],
    method_tiers: Dict[str, np.ndarray],
    primary: str, available_methods: List[str],
    *,
    consensus: np.ndarray,
    confidence_label: np.ndarray,
    flip_rate: np.ndarray,
    per_cat: Dict[str, np.ndarray],
) -> None:
    cat_names = list(per_cat)
    fieldnames = ["scene_id", "phase", "n_frames_used"]
    fieldnames += [f"score_{m}" for m in available_methods]
    fieldnames += [f"percentile_{m}" for m in available_methods]
    fieldnames += [f"tier_{m}" for m in available_methods]
    fieldnames += ["score", "percentile", "tier"]
    # Structured-metric columns.
    fieldnames += ["consensus_tier", "confidence", "weight_flip_rate"]
    fieldnames += [f"cat_{c}" for c in cat_names]

    with path.open("w", newline="") as f:
        f.write(f"# schema_version={SCHEMA_VERSION} primary={primary} "
                f"feature_set={FEATURE_SET_VERSION} "
                f"generator=rpx_benchmark.data.esd_scoring\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows):
            row = {
                "scene_id":      r["scene_id"],
                "phase":         r["phase"],
                "n_frames_used": r["n_frames_used"],
                **{f"score_{m}":      float(method_scores[m][i])      for m in available_methods},
                **{f"percentile_{m}": float(method_percentiles[m][i]) for m in available_methods},
                **{f"tier_{m}":       str(method_tiers[m][i])         for m in available_methods},
                "score":      float(method_scores[primary][i]),
                "percentile": float(method_percentiles[primary][i]),
                "tier":       str(method_tiers[primary][i]),
                "consensus_tier":   str(consensus[i]),
                "confidence":       str(confidence_label[i]),
                "weight_flip_rate": float(flip_rate[i]),
                **{f"cat_{c}":      float(per_cat[c][i])              for c in cat_names},
            }
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _attach_file_handler(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logging.getLogger("rpx_benchmark").addHandler(fh)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True,
                        help="path to phase_esd_splits.csv (Stage 1 output)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory (default: parent of --features)",
    )
    parser.add_argument("--primary", choices=SCORING_METHODS, default=DEFAULT_PRIMARY,
                        help=f"method whose tiers populate the txt files (default: {DEFAULT_PRIMARY})")
    parser.add_argument("--weighting", choices=["effort_stratified", "uniform"],
                        default="effort_stratified",
                        help="default scoring tier (default: effort_stratified)")
    parser.add_argument("--effort-alpha", type=float, default=DEFAULT_EFFORT_ALPHA,
                        help=f"convex weight on the effort block in effort_stratified "
                             f"scoring (default: {DEFAULT_EFFORT_ALPHA}; iter features "
                             f"each get alpha/2)")
    parser.add_argument("--weights-mi", type=Path, default=None,
                        help="optional MI weights JSON (activates mi_v1 scoring; "
                             "overrides --weighting)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for k-means / GMM / weight perturbation (default: 0)")
    parser.add_argument("--confidence-n-perturb", type=int, default=1000,
                        help="Dirichlet weight samples for per-row confidence (default: 1000)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="also mirror logs to this file (default: <out-dir>/build_difficulty_splits.log)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(level="DEBUG" if args.verbose else "INFO")

    from rpx_benchmark import cli_ux
    cli_ux.banner(
        "ESD splits — Stage 2 (tier assignment)",
        f"features: {args.features}",
    )

    if not args.features.is_file():
        cli_ux.error(f"--features not found: {args.features}")
        return 2

    out_dir = args.out_dir or args.features.parent
    log_file = args.log_file or (out_dir / "build_difficulty_splits.log")
    _attach_file_handler(log_file)

    cli_ux.config(
        {
            "features":            args.features,
            "out-dir":             out_dir,
            "primary":             args.primary,
            "weighting":           args.weighting,
            "effort-alpha":        args.effort_alpha,
            "weights-mi":          args.weights_mi or "(uniform fallback)",
            "seed":                args.seed,
            "confidence samples":  args.confidence_n_perturb,
            "log-file":            log_file,
        }
    )

    cli_ux.section("Building splits")
    t0 = time.perf_counter()
    try:
        with cli_ux.working("scoring + tertile cut + confidence sampling"):
            payload = build_splits(
                csv_path=args.features,
                out_dir=out_dir,
                primary=args.primary,
                weights_mi_path=args.weights_mi,
                seed=args.seed,
                confidence_n_perturb=args.confidence_n_perturb,
                weighting=args.weighting,
                effort_alpha=args.effort_alpha,
            )
    except SystemExit:
        raise
    except Exception:
        log.exception("fatal error")
        cli_ux.error("fatal error; see logfile for traceback")
        return 3

    summary = payload["summary"]  # type: ignore[index]
    duration = time.perf_counter() - t0
    cli_ux.summary(
        {
            "entries":         summary["n_entries"],
            "tertile counts":  summary["tertile_counts"],
            "elapsed":         cli_ux.fmt_duration(duration),
            "out-dir":         out_dir,
            "logfile":         log_file,
        },
        title="DONE",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

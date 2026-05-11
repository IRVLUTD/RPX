#!/usr/bin/env python3
"""Build the per-(scene, phase) ESD feature table for the RPX dataset.

Walks ``--data-root`` for ``<scene_dir>/<phase_idx>/`` directories,
computes the 18 ESD features defined in
``paper-submission/neurips-2026/overleaf/text/12_appendix.tex``,
and dumps a single JSON keyed by ``<scene_dir>.phase<phase_idx>``.

The output deliberately preserves ``scene_id`` per row so downstream
splitting (train/val/test) can group by scene_id and keep all three
phases of a scene in the same split — required for the
state-transition robustness (STR) metric.

Per-phase failures do not abort the run: the offending (scene, phase)
is logged with a full traceback, recorded in the output JSON's
``summary.failures`` list, and the script exits non-zero so the
caller knows partial output was produced.

Usage
-----

::

    python build_esd_splits.py \\
        --data-root /path/to/test_dataset_aggregated \\
        --output    experiments/splits/phase_esd_splits.json \\
        --workers   8 \\
        --log-file  build_esd_splits.log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allow running this script without installing the package by adding the
# repo's ``benchmark`` directory to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import (  # noqa: E402
    FEATURE_NAMES,
    PhaseFeatures,
    extract_phase_features,
    iter_phase_dirs,
)
from rpx_benchmark.logging_utils import configure_logging, get_logger  # noqa: E402

log = get_logger("build_esd_splits")


@dataclass
class GatherResult:
    rows: Dict[str, dict] = field(default_factory=dict)
    failures: List[Dict[str, str]] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return len(self.rows)

    @property
    def n_failed(self) -> int:
        return len(self.failures)


def _key_for(features: PhaseFeatures) -> str:
    return f"{features.scene_id}.phase{features.phase}"


def _process_one(phase_dir_str: str) -> Dict[str, object]:
    """Worker entry. Always returns a dict; on error it carries a traceback string."""
    phase_dir = Path(phase_dir_str)
    try:
        pf = extract_phase_features(phase_dir)
        return {
            "ok": True,
            "phase_dir": phase_dir_str,
            "row": {
                "key":            _key_for(pf),
                "scene_id":       pf.scene_id,
                "phase":          pf.phase,
                "n_frames_total": pf.n_frames_total,
                "n_frames_used":  pf.n_frames_used,
                "features":       pf.features,
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "phase_dir": phase_dir_str,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _record(result: Dict[str, object], gather: GatherResult,
            i: int, total: int, elapsed: Optional[float]) -> None:
    """Log + accumulate a single per-phase result. Centralised so the
    single-worker and multi-worker paths report identically."""
    phase_dir = str(result["phase_dir"])
    short = "/".join(phase_dir.rstrip("/").split("/")[-2:])
    if result["ok"]:
        row = dict(result["row"])  # type: ignore[arg-type]
        key = row.pop("key")
        gather.rows[key] = row
        suffix = f" in {elapsed:.2f}s" if elapsed is not None else ""
        log.info("[%d/%d] OK   %s — %d frames%s",
                 i, total, short, row["n_frames_used"], suffix)
    else:
        gather.failures.append({
            "phase_dir": phase_dir,
            "error":     str(result["error"]),
            "traceback": str(result["traceback"]),
        })
        log.error("[%d/%d] FAIL %s — %s\n%s",
                  i, total, short, result["error"], result["traceback"])


def _gather(data_root: Path, workers: int) -> GatherResult:
    phase_dirs: List[Path] = list(iter_phase_dirs(data_root))
    if not phase_dirs:
        raise SystemExit(f"no phase directories found under {data_root}")

    log.info("found %d (scene, phase) directories under %s", len(phase_dirs), data_root)
    gather = GatherResult()

    if workers <= 1:
        for i, pd in enumerate(phase_dirs, 1):
            t0 = time.perf_counter()
            result = _process_one(str(pd))
            _record(result, gather, i, len(phase_dirs), time.perf_counter() - t0)
        return gather

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_one, str(pd)): pd for pd in phase_dirs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                result = fut.result()
            except Exception as e:  # safety net: worker crash before _process_one returns
                pd = futures[fut]
                result = {
                    "ok": False,
                    "phase_dir": str(pd),
                    "error": f"worker crash: {type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            _record(result, gather, done, len(phase_dirs), None)
    return gather


def _compute_feature_stats(gather: GatherResult) -> Dict[str, Dict[str, float]]:
    """Per-feature descriptive stats over all successfully-extracted rows.

    Returns ``{feature_name: {n, mean, std, var, min, p25, p50, p75, max,
    cv, ci95_low, ci95_high, prop_zero}}``. Pure numpy; no extra deps.
    """
    if not gather.rows:
        return {}

    matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES]
         for row in gather.rows.values()],
        dtype=np.float64,
    )

    stats: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = matrix[:, j]
        n = int(col.size)
        mean = float(col.mean())
        # Sample std (ddof=1) is the standard convention for descriptive stats.
        std  = float(col.std(ddof=1)) if n > 1 else 0.0
        var  = std * std
        # 95% CI of the mean using a normal approximation; for small n this is
        # an approximation, but consistent with the per-feature reporting we
        # need for the paper. Tests can refine with bootstrap later.
        se   = std / math.sqrt(n) if n > 0 else 0.0
        z    = 1.959964  # two-sided 95% normal quantile
        ci_l = mean - z * se
        ci_h = mean + z * se
        cv   = (std / mean) if mean != 0 else float("nan")
        stats[name] = {
            "n":          n,
            "mean":       mean,
            "std":        std,
            "var":        var,
            "min":        float(col.min()),
            "p25":        float(np.percentile(col, 25)),
            "p50":        float(np.percentile(col, 50)),
            "p75":        float(np.percentile(col, 75)),
            "max":        float(col.max()),
            "cv":         cv,
            "ci95_low":   float(ci_l),
            "ci95_high":  float(ci_h),
            "prop_zero":  float((col == 0).mean()),
        }
    return stats


def _write_feature_stats_csv(
    stats: Dict[str, Dict[str, float]], csv_path: Path,
) -> None:
    if not stats:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["feature", "n", "mean", "std", "var", "min", "p25", "p50", "p75",
            "max", "cv", "ci95_low", "ci95_high", "prop_zero"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for name, s in stats.items():
            writer.writerow({"feature": name, **s})


def _write_output(gather: GatherResult, output: Path,
                  data_root: Path, started_at: str, duration_s: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_stats = _compute_feature_stats(gather)
    payload = {
        "schema_version": 2,  # bumped: added feature_stats block
        "feature_names":  list(FEATURE_NAMES),
        "summary": {
            "data_root":      str(data_root),
            "started_at_utc": started_at,
            "duration_s":     round(duration_s, 3),
            "n_entries":      gather.n_ok,
            "n_failed":       gather.n_failed,
            "ok":             gather.n_failed == 0,
            "failures":       gather.failures,
        },
        "feature_stats":  feature_stats,
        # Sort by scene_id then phase so the file diffs cleanly across runs.
        "phases": dict(sorted(
            gather.rows.items(),
            key=lambda kv: (kv[1]["scene_id"], kv[1]["phase"]),
        )),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=False))
    log.info("wrote %d entries (%d failures) → %s",
             gather.n_ok, gather.n_failed, output)

    # Flat CSV alongside the JSON: pandas/sklearn-friendly. JSON keeps the
    # provenance + failure log; CSV is the analysis table.
    csv_path = output.with_suffix(".csv")
    _write_csv(gather, csv_path)
    log.info("wrote %d rows × %d feature columns → %s",
             gather.n_ok, len(FEATURE_NAMES), csv_path)

    stats_csv = output.parent / (output.stem + "_feature_stats.csv")
    _write_feature_stats_csv(feature_stats, stats_csv)
    log.info("wrote per-feature descriptive stats → %s", stats_csv)


_CSV_SCHEMA_VERSION = 2  # bump in lockstep with the JSON schema_version


def _write_csv(gather: GatherResult, csv_path: Path) -> None:
    """One row per (scene, phase); raw feature values, no normalization.

    The first line is a comment marker carrying the schema version and
    feature count — readable with ``pd.read_csv(path, comment='#')`` or
    skippable with any standard CSV reader. This keeps the CSV traceable
    to the extractor version without requiring the JSON sibling.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "phase", "n_frames_total", "n_frames_used",
                  *FEATURE_NAMES]
    with csv_path.open("w", newline="") as f:
        f.write(
            f"# schema_version={_CSV_SCHEMA_VERSION} "
            f"n_features={len(FEATURE_NAMES)} "
            f"generator=rpx_benchmark.data.esd\n"
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Same sort order as the JSON, for diff stability.
        for _, row in sorted(
            gather.rows.items(),
            key=lambda kv: (kv[1]["scene_id"], kv[1]["phase"]),
        ):
            writer.writerow({
                "scene_id":       row["scene_id"],
                "phase":          row["phase"],
                "n_frames_total": row["n_frames_total"],
                "n_frames_used":  row["n_frames_used"],
                **{name: row["features"][name] for name in FEATURE_NAMES},
            })


def _attach_file_handler(log_file: Path) -> None:
    """Mirror all log records to a file in addition to the console handler."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    # Attach to the "rpx_benchmark" logger — configure_logging sets
    # propagate=False there, so attaching to Python's root captures nothing.
    logging.getLogger("rpx_benchmark").addHandler(fh)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="root containing <scene_dir>/<phase_idx>/...")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "splits" / "phase_esd_splits.json",
        help="output JSON path (default: experiments/splits/phase_esd_splits.json)",
    )
    # Workers default to all available cores: the streaming extractor
    # holds ~tens of MB per worker, so memory is not the binding constraint.
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)),
                        help="parallel workers (default: cpu_count)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="also write all log records to this file (default: alongside --output)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="DEBUG-level logging")
    args = parser.parse_args(argv)

    configure_logging(level="DEBUG" if args.verbose else "INFO")

    log_file = args.log_file or args.output.with_suffix(".log")
    _attach_file_handler(log_file)

    from rpx_benchmark import cli_ux
    cli_ux.banner(
        "ESD splits — Stage 1 (feature extraction)",
        f"data-root: {args.data_root}",
    )
    cli_ux.config(
        {
            "data-root":  args.data_root,
            "output":     args.output,
            "workers":    args.workers,
            "log-file":   log_file,
            "verbose":    args.verbose,
        }
    )

    if not args.data_root.is_dir():
        cli_ux.error(f"--data-root not a directory: {args.data_root}")
        return 2

    started = time.gmtime()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", started)
    t0 = time.perf_counter()

    cli_ux.section("Extracting per-(scene, phase) features")
    try:
        with cli_ux.working(f"scanning + extracting features ({args.workers} workers)"):
            gather = _gather(args.data_root, args.workers)
    except SystemExit:
        raise
    except Exception:
        log.exception("fatal error during gather")
        cli_ux.error("fatal error during gather; see logfile for traceback")
        return 3

    duration = time.perf_counter() - t0

    cli_ux.section("Writing output")
    try:
        _write_output(gather, args.output, args.data_root, started_iso, duration)
    except OSError:
        log.exception("failed to write output JSON")
        cli_ux.error("failed to write output JSON; see logfile for traceback")
        return 4
    cli_ux.step(f"wrote {args.output}")
    cli_ux.step(f"wrote {args.output.with_suffix('.csv')}")
    cli_ux.step(f"wrote {args.output.with_name(args.output.stem + '_feature_stats.csv')}")

    cli_ux.summary(
        {
            "entries written":  gather.n_ok,
            "failures":         gather.n_failed,
            "elapsed":          cli_ux.fmt_duration(duration),
            "rate":             cli_ux.fmt_rate(gather.n_ok, duration),
            "logfile":          log_file,
        },
        title=("DONE OK" if gather.n_failed == 0 else "DONE WITH ERRORS"),
    )
    return 1 if gather.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Emit LaTeX rows for Tables 3–8 from per-model JSON reports.

Accepts two input shapes:

1. **Runner report JSON** — per-model :class:`DeploymentReadinessReport`
   from ``BenchmarkRunner.run_with_report``. Nests ``phi_jedi`` under
   the top-level keys.
2. **Analysis summary JSON** — output of ``scripts/analyze_experiment.py``
   (the post-hoc path). Same nesting; ``phi_jedi`` already populated.

Both shapes work with the dotted-path lookups below. For the post-hoc
workflow, run ``analyze_experiment.py`` first to materialise summaries
from cell logs, then point this script at the summary files.

The script reads N files for a given task, extracts JEDI 𝒥, per-phase
primary metric, Φ (conservative), p, Φ_{0→1}, params, and latency, and
emits the ``& ... &`` separated rows that fill the corresponding tabular
block in ``paper-submission/overleaf/text/neurips_v2/05_experiments.tex``.

Usage
-----

  # Workflow A: from runner report.json files
  PYTHONPATH=. python scripts/fill_paper_table.py \\
      --task depth \\
      --reports rpx_results/*/report.json \\
      --output table_d1_rows.tex

  # Workflow B: from analyze_experiment.py summaries
  PYTHONPATH=. python scripts/fill_paper_table.py \\
      --task depth \\
      --reports rpx_results/analysis/monocular_depth/*.summary.json \\
      --output table_d1_rows.tex

Then paste the generated rows between ``\\midrule`` and ``\\bottomrule``
in the relevant table.

Per-task column layouts (must match 05_experiments.tex)
-------------------------------------------------------

  depth      Model & J & AbsRel_clu & AbsRel_int & AbsRel_cln & Φ & p & Φ_{0→1} & Params & Lat.
  detection  Model & Prompt & J & AP_clu & AP_int & AP_cln & Φ & p & Params & Lat.
  tracking   Model & Init & J & MOTA_clu & MOTA_int & MOTA_cln & IDF1_int & Φ & p & Params & Lat.
  pose       Model & J & AUC@10° & M-AUC & Cross-Δ & Φ & Params & Lat.

QA (D4/D5) is dual-task and is not emitted by this script — fill manually
or extend with a dedicated layout.

Missing values become ``---``. The script never invents numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Layouts: per-task column specifications
# --------------------------------------------------------------------------- #


@dataclass
class ColumnSpec:
    """One column in a task's row layout.

    ``field`` is a dotted path into the report dict; ``fmt`` is the
    ``"{:...}"`` format string applied when the value is finite.
    """

    field: str
    fmt: str
    fallback: str = "---"


def _layout_depth(primary: str) -> List[ColumnSpec]:
    """K=5 metrics; per-phase primary shown alongside JEDI / Φ."""
    return [
        ColumnSpec("phi_jedi.jedi.jedi", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.clu", "{:.3f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.int", "{:.3f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.cln", "{:.3f}"),
        ColumnSpec("phi_jedi.phi.phi_conservative", "{:.2f}"),
        ColumnSpec("phi_jedi.phi.p_value", "{:.1e}"),
        ColumnSpec("phi_jedi.phi_per_transition.clu->int.phi", "{:.2f}"),
        ColumnSpec("params_m", "{:.0f}M"),
        ColumnSpec("latency_ms_per_sample", "{:.0f}ms"),
    ]


def _layout_detection(primary: str = "ap50") -> List[ColumnSpec]:
    return [
        ColumnSpec("prompt", "{}"),
        ColumnSpec("phi_jedi.jedi.jedi", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.clu", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.int", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.cln", "{:.2f}"),
        ColumnSpec("phi_jedi.phi.phi_conservative", "{:.2f}"),
        ColumnSpec("phi_jedi.phi.p_value", "{:.1e}"),
        ColumnSpec("params_m", "{:.0f}M"),
        ColumnSpec("latency_ms_per_sample", "{:.0f}ms"),
    ]


def _layout_tracking(primary: str = "mota") -> List[ColumnSpec]:
    return [
        ColumnSpec("init", "{}"),
        ColumnSpec("phi_jedi.jedi.jedi", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.clu", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.int", "{:.2f}"),
        ColumnSpec(f"phi_jedi._per_cell_mean.{primary}.cln", "{:.2f}"),
        ColumnSpec("phi_jedi._per_cell_mean.idf1.int", "{:.2f}"),
        ColumnSpec("phi_jedi.phi.phi_conservative", "{:.2f}"),
        ColumnSpec("phi_jedi.phi.p_value", "{:.1e}"),
        ColumnSpec("params_m", "{:.0f}M"),
        ColumnSpec("latency_ms_per_sample", "{:.0f}ms"),
    ]


def _layout_pose() -> List[ColumnSpec]:
    return [
        ColumnSpec("phi_jedi.jedi.jedi", "{:.2f}"),
        ColumnSpec("phi_jedi._per_cell_mean.auc_10deg.clu", "{:.2f}"),
        ColumnSpec("phi_jedi._per_cell_mean.metric_auc.clu", "{:.2f}"),
        ColumnSpec("phi_jedi._cross_phase_delta", "{:.3f}"),
        ColumnSpec("phi_jedi.phi.phi_conservative", "{:.2f}"),
        ColumnSpec("params_m", "{:.0f}M"),
        ColumnSpec("latency_ms_per_sample", "{:.0f}ms"),
    ]




LAYOUTS: Dict[str, Any] = {
    "depth": _layout_depth,
    "detection": _layout_detection,
    "tracking": _layout_tracking,
    "pose": _layout_pose,
}


# --------------------------------------------------------------------------- #
# Report parsing
# --------------------------------------------------------------------------- #


def _lookup(d: Dict[str, Any], dotted: str) -> Optional[Any]:
    """Walk a dotted path into a nested dict. Returns None if any hop misses."""
    cur: Any = d
    for piece in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(piece)
        else:
            cur = getattr(cur, piece, None)
    return cur


def _format_cell(value: Any, fmt: str, fallback: str) -> str:
    """Format a numeric value, returning the fallback when missing/non-finite."""
    if value is None:
        return fallback
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v or v == float("inf") or v == float("-inf"):  # NaN/inf check
        return fallback
    try:
        return fmt.format(v)
    except (ValueError, KeyError):
        return fallback


def _row_from_report(
    report: Dict[str, Any],
    columns: Sequence[ColumnSpec],
    model_label: str,
) -> str:
    """Build a single LaTeX row from a report dict."""
    cells = [model_label]
    for col in columns:
        raw = _lookup(report, col.field)
        if isinstance(raw, str) and col.fmt == "{}":
            cells.append(raw)
        else:
            cells.append(_format_cell(raw, col.fmt, col.fallback))
    return " & ".join(cells) + r" \\"


def _derive_per_cell_means(report: Dict[str, Any]) -> None:
    """Compute per-phase per-metric means from the report's raw cell data.

    The runner ships ``phi_jedi.metrics_used`` (the metric keys) but does
    not denormalise per-phase per-metric means into the JSON. The layouts
    above reference ``phi_jedi._per_cell_mean.<metric>.<phase>``; here we
    construct that nested dict in-place by averaging the JEDI per-cell
    individual desirabilities back to raw scale via the spec bounds.

    Note: this is best-effort. When the underlying per-sample metrics are
    not present in the report JSON, the field stays empty and the
    corresponding cell renders as ``---``.
    """
    phi_jedi = report.get("phi_jedi")
    if not isinstance(phi_jedi, dict):
        return
    cell_means: Dict[str, Dict[str, float]] = {}
    # The runner currently stores aggregated JEDI per phase but not the raw
    # per-metric per-phase mean. If the runner is extended to ship them
    # (e.g. via ``phi_jedi.per_phase_metric_means``), we read here.
    raw = phi_jedi.get("per_phase_metric_means")
    if isinstance(raw, dict):
        for metric, by_phase in raw.items():
            if isinstance(by_phase, dict):
                cell_means[metric] = {p: float(v) for p, v in by_phase.items()}
    phi_jedi["_per_cell_mean"] = cell_means


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--task",
        required=True,
        choices=sorted(LAYOUTS.keys()),
        help="Which task's row layout to emit (depth/detection/tracking/pose).",
    )
    p.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Glob(s) of per-model JSON report files (one per model run).",
    )
    p.add_argument(
        "--primary",
        default=None,
        help="Primary per-phase metric key (e.g. absrel, ap50, mota). Defaults vary by task.",
    )
    p.add_argument(
        "--output",
        default="-",
        help="Output .tex path. Use '-' (default) to print to stdout.",
    )
    p.add_argument(
        "--label-from",
        default="model_name",
        help="Top-level report key whose value becomes the row's model label.",
    )
    args = p.parse_args(argv)

    # Resolve layout (some need a primary metric key).
    layout_fn = LAYOUTS[args.task]
    if args.task in {"depth", "detection", "tracking"}:
        primary = args.primary or {
            "depth": "absrel",
            "detection": "ap50",
            "tracking": "mota",
        }[args.task]
        columns = layout_fn(primary)
    else:
        columns = layout_fn()

    # Resolve glob(s).
    paths: List[Path] = []
    for pat in args.reports:
        matches = sorted(Path(m) for m in glob.glob(pat))
        if not matches:
            print(f"warning: no files matched {pat!r}", file=sys.stderr)
        paths.extend(matches)

    if not paths:
        print("error: no report files found", file=sys.stderr)
        return 2

    rows: List[str] = []
    for path in paths:
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"warning: skip {path} (invalid JSON: {e})", file=sys.stderr)
            continue
        _derive_per_cell_means(report)
        label = str(report.get(args.label_from, path.stem))
        rows.append(_row_from_report(report, columns, label))

    body = "\n".join(rows) + "\n"
    if args.output == "-":
        sys.stdout.write(body)
    else:
        out = Path(args.output)
        out.write_text(body)
        print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc analysis over per-(scene, phase) cell logs.

Reads one or more cell logs (written by the runner via
``--cell-log-path``), groups rows by ``(model_name, task)``, and runs
:func:`rpx_benchmark.phi_jedi_summary.summarize_phi_jedi` per group.
Writes a ``summary.json`` per (model, task) plus an aggregated index.

This is the decoupled-analysis half of the experiment workflow:

    1. Model runs produce cell logs (cheap to write, expensive to
       reproduce).
    2. This script reads the cell logs and computes Φ + JEDI.
    3. ``scripts/fill_paper_table.py`` reads the summaries and emits
       LaTeX rows for Tables 3–8.

Sensitivity sweeps (vary ε, swap Wilks↔Pillai, change the metric set)
are re-runs of step 2 on the same cell logs — no model re-evaluation
needed.

Usage
-----

  PYTHONPATH=. python scripts/analyze_experiment.py \\
      --cell-logs rpx_results/monocular_depth/*/cells.parquet \\
      --output-dir rpx_results/analysis/monocular_depth/ \\
      --epsilon 0.0

The script never invents numbers: when a group has too few scenes for
the MANOVA (K ≥ N_eff), the summary records the diagnostic and the
corresponding Φ fields stay ``None``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Make the toolkit importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rpx_benchmark.cell_log import METRIC_PREFIX, merge_cells, read_cells  # noqa: E402
from rpx_benchmark.phi_jedi_summary import summarize_phi_jedi  # noqa: E402


def _cell_to_runner_row(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a cell-log row into the format ``summarize_phi_jedi`` expects.

    The summary expects a per-sample-like dict with ``scene``, ``phase``,
    ``difficulty`` keys plus bare metric keys (no ``metric:`` prefix).
    Each cell is treated as a single "sample" because it already encodes
    the per-(scene, phase) mean across frames.
    """
    out: Dict[str, Any] = {
        "scene": cell.get("scene_id"),
        "phase": cell.get("phase"),
        "difficulty": cell.get("difficulty"),
        "id": f"{cell.get('scene_id')}_{cell.get('phase')}",
    }
    for k, v in cell.items():
        if k.startswith(METRIC_PREFIX):
            out[k[len(METRIC_PREFIX) :]] = v
    return out


def _resolve_globs(patterns: Sequence[str]) -> List[Path]:
    """Expand glob patterns into a flat list of paths."""
    paths: List[Path] = []
    for pat in patterns:
        matches = sorted(Path(m) for m in glob.glob(pat))
        if not matches:
            print(f"warning: no files matched {pat!r}", file=sys.stderr)
        paths.extend(matches)
    return paths


def _group_by_model_task(
    cells: List[Dict[str, Any]],
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Bucket cells by (model_name, task)."""
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        key = (c.get("model_name"), c.get("task"))
        buckets[key].append(c)
    return dict(buckets)


def _safe_filename(s: str) -> str:
    """Make ``s`` safe to use as a directory or file name."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def _paper_metric_keys(task: str) -> Optional[Sequence[str]]:
    """Return a locked paper K-vector instead of auto-selecting diagnostics."""
    if task == "video_depth":
        from rpx_benchmark.tasks.video_depth import D1V_MANOVA_METRICS

        return D1V_MANOVA_METRICS
    if task == "monocular_depth":
        from rpx_benchmark.tasks.video_depth import D1F_MANOVA_METRICS

        return D1F_MANOVA_METRICS
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--cell-logs",
        nargs="+",
        required=True,
        help="One or more glob patterns pointing at cell logs (.parquet or .jsonl).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory under which per-(model, task) summaries are written.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help=(
            "JEDI ε floor for individual desirabilities. Default 0.0 (strict "
            "geometric-mean dominance, paper §3.3)."
        ),
    )
    p.add_argument(
        "--no-holm",
        action="store_true",
        help="Disable Holm-Bonferroni adjustment on the per-transition p-values.",
    )
    p.add_argument(
        "--index-name",
        default="index.json",
        help="Filename for the top-level index summarising every (model, task).",
    )
    p.add_argument(
        "--write-adjacent",
        action="store_true",
        help=(
            "Also drop a copy of each summary as ``summary.json`` next to "
            "its source cell log. This puts the summary inside the per-run "
            "output directory so the toolkit's Box upload (upload_run_dir) "
            "picks it up automatically — no separate sync step needed."
        ),
    )
    args = p.parse_args(argv)

    paths = _resolve_globs(args.cell_logs)
    if not paths:
        print("error: no cell-log files found", file=sys.stderr)
        return 2

    cells = merge_cells(paths)
    if not cells:
        print(f"error: cell logs at {paths} contain no rows", file=sys.stderr)
        return 2

    # Track which source path each (model, task) came from so we can
    # drop the adjacent summary back next to the right cell log when
    # --write-adjacent is set. When a single cell log holds rows for
    # multiple (model, task) keys, the first source path wins.
    source_by_key: Dict[tuple, Path] = {}
    for path in paths:
        for row in read_cells(path):
            key = (row.get("model_name"), row.get("task"))
            if key not in source_by_key:
                source_by_key[key] = path

    buckets = _group_by_model_task(cells)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, Any]] = []
    for (model_name, task), group_cells in sorted(buckets.items()):
        # Translate the cell rows into the summarize_phi_jedi input format.
        rows = [_cell_to_runner_row(c) for c in group_cells]
        summary = summarize_phi_jedi(
            rows,
            metric_keys=_paper_metric_keys(str(task)),
            jedi_epsilon=args.epsilon,
            apply_holm=not args.no_holm,
        )

        out_payload = {
            "model_name": model_name,
            "task": task,
            "phi_jedi": summary.to_dict(),
            "input_cells_path": [str(p) for p in paths],
            "n_input_cells": len(group_cells),
            "epsilon": args.epsilon,
            "apply_holm": not args.no_holm,
        }

        # Per-(model, task) summary file in the central analysis dir.
        sub_dir = args.output_dir / _safe_filename(str(task))
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / f"{_safe_filename(str(model_name))}.summary.json"
        out_path.write_text(json.dumps(out_payload, indent=2, default=str))

        # Also drop a copy alongside the source cell log so the toolkit's
        # Box upload (upload_run_dir) picks it up without a separate sync.
        if args.write_adjacent:
            src = source_by_key.get((model_name, task))
            if src is not None:
                adj = src.parent / "summary.json"
                adj.write_text(json.dumps(out_payload, indent=2, default=str))

        # Index entry with the most-useful summary scalars surfaced.
        idx_row: Dict[str, Any] = {
            "model_name": model_name,
            "task": task,
            "summary_path": str(out_path),
            "n_cells": len(group_cells),
            "n_eff": summary.n_eff,
            "n_dropped": summary.n_dropped,
            "metrics_used": summary.metrics_used,
        }
        if summary.phi is not None:
            idx_row["phi_wilks"] = summary.phi.phi_wilks
            idx_row["phi_pillai"] = summary.phi.phi_pillai
            idx_row["phi_conservative"] = summary.phi.phi_conservative
            idx_row["phi_p_value"] = summary.phi.p_value
            idx_row["phi_interpretation"] = summary.phi_interpretation
        if summary.jedi is not None:
            idx_row["jedi"] = summary.jedi.jedi
            idx_row["jedi_n_oob"] = summary.jedi.n_out_of_bounds
        if summary.notes:
            idx_row["notes"] = summary.notes
        index.append(idx_row)

        print(
            f"{task:20s}  {model_name:30s}  "
            f"N_eff={summary.n_eff:3d}  "
            f"Φ={(summary.phi.phi_conservative if summary.phi else float('nan')):.3f}  "
            f"J={(summary.jedi.jedi if summary.jedi else float('nan')):.3f}",
            file=sys.stderr,
        )

    index_path = args.output_dir / args.index_name
    index_path.write_text(json.dumps(index, indent=2, default=str))
    print(f"\nWrote {len(buckets)} summaries + index → {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate per-model `result.json` files into the sweep-level DRS table.

The runner emits an :class:`OperatingPoint` per (model, split) under
``result.json["deployment_readiness"]["operating_point"]``. The DRS
itself is **defined sweep-wide** — it needs the median FLOPs across
every model in the sweep as the efficiency anchor — so it can't be
computed during a single-model run. This script does the post-sweep
aggregation:

1. Walk ``rpx_results/<model_name>/<split>/result.json``.
2. Pull each ``operating_point`` (one or more per model — typically one
   per precision: fp32, fp16).
3. Call :func:`rpx_benchmark.compute_sweep_drs` to get the DRS + TP + R
   + E components per model.
4. (Optional) Call :func:`rpx_benchmark.drs_sensitivity.run_sensitivity`
   to fill in the paper-appendix exponent / E-function / anchor
   sensitivity tables.
5. Write the DRS table (CSV + JSON) to ``rpx_results/_sweep/drs.csv`` /
   ``drs.json``; sensitivity to ``rpx_results/_sweep/sensitivity.json``.

Usage
-----
::

    # After running run_depth.py for every model in the registry:
    PYTHONPATH=. python scripts/run_drs_sweep.py --split easy
    PYTHONPATH=. python scripts/run_drs_sweep.py --split easy --sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_operating_points(
    results_root: Path,
    split: str,
) -> Dict[str, List]:
    """Walk ``results_root/<model>/<split>/result.json`` and pull each
    ``operating_point``. Returns ``{model_name: [OperatingPoint, ...]}``.
    """
    from rpx_benchmark import OperatingPoint  # noqa: PLC0415

    models: Dict[str, List[OperatingPoint]] = {}
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        result_path = model_dir / split / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text())
        op_dict = (payload.get("deployment_readiness") or {}).get("operating_point")
        if not op_dict:
            print(f"  [skip] {model_dir.name}: result.json has no operating_point")
            continue
        op = OperatingPoint(
            precision=op_dict["precision"],
            task_metric=op_dict["task_metric"],
            task_metric_name=op_dict["task_metric_name"],
            higher_is_better=op_dict["higher_is_better"],
            str_score=op_dict["str_score"],
            flops_g=op_dict["flops_g"],
            params_m=op_dict["params_m"],
            memory_traffic_gb=op_dict.get("memory_traffic_gb"),
        )
        models.setdefault(model_dir.name, []).append(op)
    return models


def _write_drs_table(
    results: Dict,
    out_csv: Path,
    out_json: Path,
) -> None:
    """Write the DRS table as CSV (paper-friendly) + JSON (analytics-friendly)."""
    rows = []
    for name, res in sorted(results.items(), key=lambda kv: -kv[1].drs):
        op = res.best_op
        rows.append(
            {
                "model": name,
                "drs": round(res.drs, 4),
                "tp": round(res.tp, 4),
                "robustness": round(res.r, 4),
                "efficiency": round(res.e, 4),
                "best_precision": op.precision if op else "?",
                "task_metric": round(op.task_metric, 4) if op else None,
                "task_metric_name": op.task_metric_name if op else "?",
                "str_score": round(op.str_score, 4) if op else None,
                "flops_g": round(op.flops_g, 1) if op else None,
                "params_m": round(op.params_m, 1) if op else None,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {out_csv}")
    print(f"  wrote {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("./rpx_results"),
        help="dir containing <model>/<split>/result.json",
    )
    ap.add_argument("--split", default="easy", help="which split to aggregate")
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="output dir (default: <results-root>/_sweep)"
    )
    ap.add_argument(
        "--sensitivity",
        action="store_true",
        help="also run the DRS sensitivity analysis (Kendall's τ "
        "across exponent / E-function / anchor perturbations) "
        "and emit it to sensitivity.json",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or args.results_root / "_sweep"
    print(f"[drs-sweep] split={args.split} reading from {args.results_root}")
    models = _load_operating_points(args.results_root, args.split)
    if not models:
        from rpx_benchmark import DatasetError

        raise DatasetError(
            f"no models with operating_point found under "
            f"{args.results_root}/<model>/{args.split}/result.json",
            hint="Run scripts/run_depth.py for each registered model first; "
            "the runner persists OperatingPoint into result.json's "
            "deployment_readiness.operating_point.",
        )
    print(
        f"  loaded {sum(len(ops) for ops in models.values())} operating "
        f"points across {len(models)} models"
    )

    from rpx_benchmark import compute_sweep_drs

    results = compute_sweep_drs(models)
    print(f"  computed DRS for {len(results)} models (F_median = median across all OPs)")

    _write_drs_table(
        results,
        out_csv=out_dir / f"drs_{args.split}.csv",
        out_json=out_dir / f"drs_{args.split}.json",
    )

    if args.sensitivity:
        try:
            from rpx_benchmark.drs_sensitivity import run_sensitivity
        except ImportError:
            print("  [skip] sensitivity: rpx_benchmark.drs_sensitivity not present")
        else:
            print(f"[drs-sweep] running sensitivity analysis...")
            sens = run_sensitivity(models)
            out = out_dir / f"sensitivity_{args.split}.json"
            from dataclasses import asdict, is_dataclass

            payload = asdict(sens) if is_dataclass(sens) else sens
            out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            print(f"  wrote {out}")

    # Headline print: top-3 DRS for the human reading the terminal.
    print()
    print(f"=== Top-3 by DRS (split={args.split}) ===")
    for name, res in sorted(results.items(), key=lambda kv: -kv[1].drs)[:3]:
        print(f"  {name:>32}  DRS={res.drs:.4f}  (TP={res.tp:.3f}  R={res.r:.3f}  E={res.e:.3f})")


if __name__ == "__main__":
    main()

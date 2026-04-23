#!/usr/bin/env python3
"""Generate LaTeX table fragments from experiment result JSONs.

Usage:
    python generate_latex_tables.py --results-dir ../results --output-dir ../tables

Reads all <task>/<model>.json files, aggregates metrics, and outputs:
  - main_results.tex   (Table 5 in paper: all models × all metrics)
  - phase_results.tex  (Table 6: per-phase breakdown + STR)
  - todo_summary.txt   (list of remaining \todo{X} values to fill)

Each JSON must follow the schema in EXPERIMENT_PLAYBOOK.md.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

def load_results(results_dir: Path) -> dict:
    """Load all result JSONs into {task: {model: data}}."""
    results = defaultdict(dict)
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for json_file in sorted(task_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            model_name = data.get("model", json_file.stem)
            results[task_dir.name][model_name] = data
    return dict(results)


def format_val(v, fmt=".1f"):
    """Format a numeric value or return \\todo{X} if None/missing."""
    if v is None or v == "":
        return r"\todo{X}"
    try:
        return f"{float(v):{fmt}}"
    except (ValueError, TypeError):
        return str(v)


def generate_main_table(results: dict, output_path: Path):
    """Generate the main results table (Table 5)."""
    # Task display config: task_key -> (display_name, primary_metric, metric_fmt)
    task_config = {
        "object_detection": ("Open-Vocab Det. (AP$_{50}$, \\%)", "ap50", ".1f"),
        "segmentation": ("Instance Seg. (mIoU, \\%)", "miou", ".1f"),
        "object_tracking": ("Tracking (MOTA, \\%)", "mota", ".1f"),
        "visual_grounding": ("Grounding (Acc@0.5, \\%)", "acc_at_05", ".1f"),
        "relative_pose": ("Rel. Pose (Rot. err, $^\\circ$)", "rot_err", ".2f"),
        "monocular_depth": ("Depth (AbsRel, \\%)", "abs_rel", ".3f"),
        "novel_view_synthesis": ("NVS (PSNR, dB)", "psnr", ".2f"),
        "keypoint_matching": ("Keypoints (Acc@3px, \\%)", "acc_3px", ".1f"),
    }

    lines = []
    for task_key, (display, metric, fmt) in task_config.items():
        if task_key not in results:
            lines.append(f"% WARNING: no results for {task_key}")
            continue
        models = results[task_key]
        n = len(models)
        lines.append(f"\\midrule")
        for i, (model_name, data) in enumerate(models.items()):
            agg = data.get("aggregate", {})
            overall = agg.get("overall", {})
            by_phase = agg.get("by_phase", {})

            s_overall = format_val(overall.get(metric), fmt)
            # Compute STR if phase data exists
            s_c = by_phase.get("clutter", {}).get(metric)
            s_i = by_phase.get("interaction", {}).get(metric)
            s_l = by_phase.get("clean", {}).get(metric)

            delta_int = format_val(
                (float(s_i) - float(s_c)) if s_i and s_c else None, "+.2f"
            ) if s_i and s_c else r"\todo{X}"
            delta_rec = format_val(
                (float(s_l) - float(s_i)) if s_l and s_i else None, "+.2f"
            ) if s_l and s_i else r"\todo{X}"

            ts = format_val(data.get("ts_score"), ".3f")
            sgc = format_val(data.get("sgc_score"), ".3f")
            params = format_val(data.get("params_M"), ".0f")
            flops = format_val(data.get("flops_G"), ".0f")

            prefix = f"\\multirow{{{n}}}{{*}}{{\\shortstack[l]{{{display}}}}}" if i == 0 else ""
            lines.append(
                f" {prefix} & {model_name} & local & {s_overall} & {delta_int} & {delta_rec} & {ts} & {sgc} & {params}\\,M / {flops}\\,G \\\\"
            )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


def generate_todo_summary(results: dict, output_path: Path):
    """List what's filled vs what's still missing."""
    filled = 0
    missing = 0
    details = []
    for task, models in results.items():
        for model, data in models.items():
            agg = data.get("aggregate", {}).get("overall", {})
            if agg:
                filled += 1
                details.append(f"  ✅ {task}/{model}: {agg}")
            else:
                missing += 1
                details.append(f"  ⬜ {task}/{model}: no aggregate results")

    summary = [
        f"Experiment results summary",
        f"  Filled: {filled}",
        f"  Missing: {missing}",
        f"  Total: {filled + missing}",
        "",
    ] + details
    output_path.write_text("\n".join(summary) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results", type=Path)
    parser.add_argument("--output-dir", default="../tables", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}. Waiting for experiment JSONs.")
        sys.exit(0)

    generate_main_table(results, args.output_dir / "main_results.tex")
    generate_todo_summary(results, args.output_dir / "todo_summary.txt")
    print("Done. Paste table fragments into the paper.")

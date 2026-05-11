#!/usr/bin/env python3
"""Quick-check script to run as soon as first results land.

Looks for the four candidate surprising findings:
  1. Robustness Inversion: standard accuracy vs RPX deployment score
  2. ESD Reversal: Easy ranking vs Hard ranking
  3. Coherence Collapse: SGC by phase
  4. Flickering Giants: accuracy vs TS

Usage:
    python check_surprising_findings.py --results-dir ../results

Run this after Day 3 of experiments (first ~12 model results).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_all_results(results_dir: Path):
    """Load all result JSONs."""
    results = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for json_file in sorted(task_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            data["_task"] = task_dir.name
            data["_file"] = str(json_file)
            results.append(data)
    return results


def check_str_vs_accuracy(results):
    """Finding 1: Does standard accuracy predict STR?"""
    print("\n" + "=" * 60)
    print("FINDING 1: STR vs Accuracy (Robustness Inversion)")
    print("=" * 60)

    pairs = []
    for r in results:
        agg = r.get("aggregate", {})
        overall = agg.get("overall", {})
        by_phase = agg.get("by_phase", {})

        # Get primary metric value
        for metric in ["abs_rel", "miou", "ap50", "acc_3px", "psnr"]:
            if metric in overall:
                accuracy = overall[metric]
                break
        else:
            continue

        clutter = by_phase.get("clutter", {})
        interaction = by_phase.get("interaction", {})
        if not clutter or not interaction:
            continue

        for metric in ["abs_rel", "miou", "ap50", "acc_3px", "psnr"]:
            if metric in clutter and metric in interaction:
                str_ci = interaction[metric] - clutter[metric]
                pairs.append((r["model"], r["_task"], accuracy, str_ci))
                break

    if len(pairs) < 3:
        print(f"  Only {len(pairs)} models with phase data. Need more results.")
        return

    accuracies = [p[2] for p in pairs]
    strs = [p[3] for p in pairs]

    from scipy.stats import spearmanr
    rho, pval = spearmanr(accuracies, strs)
    print(f"  Spearman ρ(accuracy, STR_C→I) = {rho:.3f} (p={pval:.4f})")
    print(f"  Models: {len(pairs)}")
    if abs(rho) < 0.3:
        print("  >>> FINDING DETECTED: Weak correlation → Robustness Inversion!")
    elif rho < -0.3:
        print("  >>> FINDING DETECTED: Negative correlation → STRONG Robustness Inversion!")
    else:
        print("  >>> Moderate/strong positive correlation. Standard accuracy does predict STR.")

    for model, task, acc, str_ci in sorted(pairs, key=lambda x: x[3]):
        print(f"    {task}/{model}: accuracy={acc:.3f}, STR={str_ci:+.3f}")


def check_esd_reversal(results):
    """Finding 2: Do Easy and Hard rankings agree?"""
    print("\n" + "=" * 60)
    print("FINDING 2: ESD Ranking Reversal")
    print("=" * 60)

    easy_scores = {}
    hard_scores = {}

    for r in results:
        agg = r.get("aggregate", {})
        by_diff = agg.get("by_difficulty", {})
        if not by_diff:
            continue

        key = f"{r['_task']}/{r['model']}"
        for metric in ["abs_rel", "miou", "ap50", "acc_3px", "psnr"]:
            if metric in by_diff.get("easy", {}):
                easy_scores[key] = by_diff["easy"][metric]
                hard_scores[key] = by_diff.get("hard", {}).get(metric)
                break

    shared = [k for k in easy_scores if hard_scores.get(k) is not None]
    if len(shared) < 3:
        print(f"  Only {len(shared)} models with ESD data. Need more results.")
        return

    from scipy.stats import kendalltau
    e = [easy_scores[k] for k in shared]
    h = [hard_scores[k] for k in shared]
    tau, pval = kendalltau(e, h)
    print(f"  Kendall τ(Easy rank, Hard rank) = {tau:.3f} (p={pval:.4f})")
    if tau < 0.5:
        print("  >>> FINDING DETECTED: Rankings diverge between Easy and Hard!")
    else:
        print("  >>> Rankings are consistent across difficulty levels.")


def check_sgc_by_phase(results):
    """Finding 3: Does SGC drop during interaction?"""
    print("\n" + "=" * 60)
    print("FINDING 3: SGC by Phase (Coherence Collapse)")
    print("=" * 60)

    sgc_by_phase = defaultdict(list)
    for r in results:
        if "sgc_by_phase" in r:
            for phase, val in r["sgc_by_phase"].items():
                sgc_by_phase[phase].append(val)

    if not sgc_by_phase:
        print("  No SGC data found. Need segmentation + depth results with SGC computation.")
        return

    for phase in ["clutter", "interaction", "clean"]:
        vals = sgc_by_phase.get(phase, [])
        if vals:
            print(f"  SGC_{phase}: mean={np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")

    if "clutter" in sgc_by_phase and "interaction" in sgc_by_phase:
        drop = np.mean(sgc_by_phase["interaction"]) - np.mean(sgc_by_phase["clutter"])
        print(f"  SGC drop (interaction - clutter): {drop:+.3f}")
        if drop < -0.05:
            print("  >>> FINDING DETECTED: SGC drops during interaction!")


def check_ts_vs_accuracy(results):
    """Finding 4: Flickering Giants — high accuracy, low TS."""
    print("\n" + "=" * 60)
    print("FINDING 4: TS vs Accuracy (Flickering Giants)")
    print("=" * 60)

    pairs = []
    for r in results:
        ts = r.get("ts_score")
        if ts is None:
            continue
        agg = r.get("aggregate", {}).get("overall", {})
        for metric in ["abs_rel", "miou"]:
            if metric in agg:
                pairs.append((r["model"], r["_task"], agg[metric], ts))
                break

    if len(pairs) < 3:
        print(f"  Only {len(pairs)} models with TS data. Need more results.")
        return

    from scipy.stats import pearsonr
    accuracies = [p[2] for p in pairs]
    tss = [p[3] for p in pairs]
    r_val, pval = pearsonr(accuracies, tss)
    print(f"  Pearson r(accuracy, TS) = {r_val:.3f} (p={pval:.4f})")
    if abs(r_val) < 0.4:
        print("  >>> FINDING DETECTED: TS and accuracy are weakly correlated!")

    for model, task, acc, ts in sorted(pairs, key=lambda x: -x[2]):
        flag = " ← FLICKERING GIANT" if acc > np.median(accuracies) and ts < np.median(tss) else ""
        print(f"    {task}/{model}: accuracy={acc:.3f}, TS={ts:.3f}{flag}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results", type=Path)
    args = parser.parse_args()

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "check_surprising_findings — first-look results triage",
        "fast sanity-check on the day the first experiments come back",
    )
    cli_ux.config(vars(args))

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}.")
        print("Run this after experiments produce JSON outputs.")
        sys.exit(0)

    print(f"Loaded {len(results)} result files.")

    try:
        check_str_vs_accuracy(results)
    except ImportError:
        print("  (Install scipy: pip install scipy)")
    try:
        check_esd_reversal(results)
    except ImportError:
        print("  (Install scipy: pip install scipy)")
    check_sgc_by_phase(results)
    try:
        check_ts_vs_accuracy(results)
    except ImportError:
        print("  (Install scipy: pip install scipy)")

    print("\n" + "=" * 60)
    print("NEXT: Use whichever finding is strongest as the paper's headline.")
    print("Name it. A named finding is 10x more memorable than a p-value.")
    print("=" * 60)

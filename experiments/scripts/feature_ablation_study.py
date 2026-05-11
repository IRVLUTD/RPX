#!/usr/bin/env python3
"""Feature ablation study: rank the 27 ESD features by load-bearing-ness.

For each feature in turn, we drop it from the matrix, recompute the uniform
RPX-DS tier, and quantify how much the assignment changes:

- **Kendall τ** between full-feature and ablated continuous scores
  (close to 1.0 = the feature was redundant; close to 0.0 = removing it
  scrambles the ranking).
- **Tier-flip rate**: fraction of (scene, phase) pairs whose Easy/Med/Hard
  tier changes when the feature is removed.
- **Tier-flip count**: absolute number of pairs flipped.

The output ranks features from "most load-bearing" (high flip rate, low τ)
to "least informative" (low flip rate, high τ). The latter are candidates
for removal in v2 of the feature set; flagging them now is the answer to
the open question "should we drop fisheye_dark / fisheye_bright?"

Usage
-----

::

    python feature_ablation_study.py \\
        --features benchmark/data/splits/phase_esd_splits.csv \\
        --out-dir  benchmark/data/splits/methodology_study/ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau
except ImportError as e:
    raise SystemExit(
        "feature_ablation_study requires extra deps. Install with:\n"
        "  pip install 'rpx-benchmark[analysis]'\n"
        f"missing: {e}"
    ) from e

_REPO_ROOT     = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402
from rpx_benchmark.data.esd_scoring import (  # noqa: E402
    percentile_normalize,
    sha256_of_file,
    tertile_cut,
)


def run_ablation(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    """For each feature in FEATURE_NAMES, drop it and measure tier impact."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, comment="#")
    missing = [n for n in FEATURE_NAMES if n not in df.columns]
    if missing:
        raise SystemExit(f"feature columns missing from CSV: {missing}")

    feat = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    nonzero = feat.sum(axis=1) > 0
    if (~nonzero).any():
        df   = df[nonzero].reset_index(drop=True)
        feat = feat[nonzero]

    pn_full    = percentile_normalize(feat)
    score_full = pn_full.mean(axis=1)
    tier_full  = tertile_cut(score_full)
    n          = len(df)
    F          = pn_full.shape[1]

    rows: List[dict] = []
    for j, name in enumerate(FEATURE_NAMES):
        # Drop column j and recompute.
        keep = np.delete(np.arange(F), j)
        # Re-percentile-normalise on the reduced feature subset (the
        # remaining columns' ranks are unchanged, but the ablation matters
        # at the score-aggregation level).
        pn_abl    = pn_full[:, keep]
        score_abl = pn_abl.mean(axis=1)
        tier_abl  = tertile_cut(score_abl)

        tau, _ = kendalltau(score_full, score_abl)
        flip_mask  = tier_abl != tier_full
        flip_rate  = float(flip_mask.mean())
        flip_count = int(flip_mask.sum())

        # Per-tier flip breakdown.
        tier_flip_breakdown: dict = {}
        for tier in ("easy", "medium", "hard"):
            mask_tier = tier_full == tier
            if mask_tier.any():
                tier_flip_breakdown[f"flip_in_{tier}"] = int(
                    (flip_mask & mask_tier).sum()
                )
            else:
                tier_flip_breakdown[f"flip_in_{tier}"] = 0

        rows.append({
            "feature":        name,
            "kendall_tau":    float(tau),
            "tier_flip_rate": flip_rate,
            "tier_flip_n":    flip_count,
            **tier_flip_breakdown,
        })

    out_df = pd.DataFrame(rows).sort_values("tier_flip_rate", ascending=False).reset_index(drop=True)
    out_df["rank_load_bearing"] = np.arange(1, len(out_df) + 1)

    out_df.to_csv(out_dir / "feature_ablation.csv", index=False)

    # Provenance.
    (out_dir / "ablation_provenance.json").write_text(json.dumps({
        "input_sha256":     sha256_of_file(csv_path),
        "n_entries":        int(n),
        "n_features":       int(F),
        "feature_names":    list(FEATURE_NAMES),
        "method":           "drop-one + uniform mean_pn + tertile_cut",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    return out_df


def make_plot(ablation_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: features ranked by tier-flip rate, with Kendall τ overlay."""
    fig, ax1 = plt.subplots(figsize=(13, 6))
    n_feat = len(ablation_df)
    xs = np.arange(n_feat)
    bars = ax1.bar(xs, ablation_df["tier_flip_rate"], color="steelblue",
                    label="tier-flip rate")
    ax1.set_xlabel("feature (ranked from most → least load-bearing)")
    ax1.set_ylabel("tier-flip rate when feature is dropped", color="steelblue")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(ablation_df["feature"], rotation=90, fontsize=8)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0, max(0.05, ablation_df["tier_flip_rate"].max() * 1.15))

    ax2 = ax1.twinx()
    ax2.plot(xs, ablation_df["kendall_tau"], "o-", color="firebrick",
              label="Kendall τ vs full ranking")
    ax2.set_ylabel("Kendall τ between ablated and full RPX-DS", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax2.set_ylim(0, 1.0)

    fig.suptitle("Feature ablation — drop one, recompute, measure tier impact")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "splits"
                / "methodology_study" / "ablation",
    )
    args = parser.parse_args(argv)

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "feature_ablation_study — which features are load-bearing",
        "rank the 27 ESD features by their contribution to the final tier assignment",
    )
    cli_ux.config(vars(args))

    if not args.features.is_file():
        raise SystemExit(f"--features not found: {args.features}")

    out_df = run_ablation(args.features, args.out_dir)
    make_plot(out_df, args.out_dir / "feature_ablation.png")

    print(f"\n[done] wrote {args.out_dir / 'feature_ablation.csv'}")
    print(f"[done] wrote {args.out_dir / 'feature_ablation.png'}")
    print("\nTop-5 most load-bearing features (drop → most tier change):")
    print(out_df.head(5)[["rank_load_bearing", "feature", "tier_flip_rate",
                           "tier_flip_n", "kendall_tau"]].to_string(index=False))
    print("\nBottom-5 least informative features (drop → least change):")
    print(out_df.tail(5)[["rank_load_bearing", "feature", "tier_flip_rate",
                           "tier_flip_n", "kendall_tau"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

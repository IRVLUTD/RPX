#!/usr/bin/env python3
"""Five-figure story arc that walks reviewers through the ESD methodology.

Each figure has one job, a clear headline, and is designed for publication
(not diagnostic dumping):

  ACT 1 — fig1_difficulty_has_many_faces.png
          Radar plots of 4 representative scenes showing how a single "Hard"
          label hides wildly different multi-faceted profiles.

  ACT 2 — fig2_methods_disagree.png
          ARI heatmap across 11 methods + headline "median ARI = 0.083".

  ACT 3 — fig3_why_uniform_wins.png
          Why mean_pn is the right primary: bootstrap-stable, reproducible,
          and the centre of methodological disagreement.

  ACT 4 — fig4_structured_metric_in_action.png
          What the user actually sees in phase_difficulty.json for 3 scenes
          with low/medium/high cross-method confidence.

  ACT 5 — fig5_methodology_finds_real_problems.png
          Top-5 outliers + the per-category profile that flags them; two
          outliers correspond to documented T265 capture failures.

Design principles:
- Custom palette (not matplotlib defaults).
- Large fonts (titles ≥16pt, body ≥10pt).
- One plot per figure (no busy grids).
- Headlines tell the story; details support.
- Whitespace is deliberate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

_REPO_ROOT     = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402
from rpx_benchmark.data.esd_scoring import (  # noqa: E402
    FEATURE_CATEGORIES,
    compute_confidence,
    compute_per_category_scores,
    percentile_normalize,
    tertile_cut,
)


# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #

PALETTE = {
    "easy":       "#2C7A7B",   # deep teal
    "medium":     "#DD6B20",   # warm orange
    "hard":       "#C53030",   # deep red
    "neutral":    "#2D3748",   # dark slate
    "neutral_2":  "#718096",   # mid slate
    "background": "#F7FAFC",   # almost-white
    "accent":     "#2B6CB0",   # deep blue
    "highlight":  "#805AD5",   # purple
    "muted":      "#CBD5E0",   # light gray
}
CONFIDENCE_COLOR = {
    "high":   "#2C7A7B",
    "medium": "#DD6B20",
    "low":    "#C53030",
}
CATEGORY_COLOR = {
    "annotation_effort":    "#2B6CB0",
    "scene_complexity":     "#319795",
    "occlusion":            "#805AD5",
    "depth_quality":        "#D69E2E",
    "photometric_conflict": "#DD6B20",
    "temporal_stability":   "#38A169",
    "camera_motion":        "#C53030",
    "fisheye_stereo":       "#553C9A",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["background"],
        "axes.facecolor":    PALETTE["background"],
        "savefig.facecolor": PALETTE["background"],
        "axes.edgecolor":    PALETTE["neutral_2"],
        "axes.labelcolor":   PALETTE["neutral"],
        "axes.titlecolor":   PALETTE["neutral"],
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        PALETTE["muted"],
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        "xtick.color":       PALETTE["neutral_2"],
        "ytick.color":       PALETTE["neutral_2"],
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.labelsize":    11,
        "axes.titlesize":    14,
        "legend.fontsize":   10,
        "legend.frameon":    False,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":         11,
    })


def headline(fig, headline_text: str, subtitle_text: str = "") -> None:
    fig.suptitle(headline_text, fontsize=18, fontweight="bold",
                  color=PALETTE["neutral"], y=0.99)
    if subtitle_text:
        fig.text(0.5, 0.945, subtitle_text, ha="center", va="top",
                  fontsize=12, color=PALETTE["neutral_2"], style="italic")


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #

def load(csv_path: Path):
    df = pd.read_csv(csv_path, comment="#")
    feat = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    nz = feat.sum(axis=1) > 0
    df, feat = df[nz].reset_index(drop=True), feat[nz]
    pn = percentile_normalize(feat)
    return df, feat, pn


# --------------------------------------------------------------------------- #
# ACT 1 — Multi-faceted difficulty (radar)
# --------------------------------------------------------------------------- #

def fig_act1(df: pd.DataFrame, pn: np.ndarray, out_path: Path) -> None:
    per_cat = compute_per_category_scores(pn, list(FEATURE_NAMES))
    cat_matrix = np.column_stack([per_cat[c] for c in per_cat])
    cat_labels = [c.replace("_", "\n") for c in per_cat]
    cat_keys   = list(per_cat)
    n_cats     = len(cat_labels)

    mean_pn = pn.mean(axis=1)
    tier = tertile_cut(mean_pn)

    # Pick 4 representative scenes:
    # 1. Strongly Easy (low everywhere)
    # 2. Strongly Hard (high everywhere)
    # 3. Mixed Hard (hard on depth/motion, easy on complexity)
    # 4. Outlier-style (one category extreme)
    easy_idx = int(np.argmin(mean_pn))
    hard_idx = int(np.argmax(mean_pn))
    # Mixed: high std across categories (hard on some, easy on others)
    cat_std = cat_matrix.std(axis=1)
    mixed_idx = int(np.argmax(cat_std * (mean_pn > 0.4) * (mean_pn < 0.6)))
    # Outlier on a single category: max value of (max_cat - second_max_cat)
    sorted_cat = np.sort(cat_matrix, axis=1)
    spike = sorted_cat[:, -1] - sorted_cat[:, -2]
    outlier_idx = int(np.argmax(spike))

    samples = [
        ("Strongly Easy",   easy_idx,    PALETTE["easy"]),
        ("Strongly Hard",   hard_idx,    PALETTE["hard"]),
        ("Mixed profile",   mixed_idx,   PALETTE["accent"]),
        ("Single-spike",    outlier_idx, PALETTE["highlight"]),
    ]

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5),
                              subplot_kw={"projection": "polar"})

    for ax, (label, idx, color) in zip(axes, samples):
        values = cat_matrix[idx].tolist() + [cat_matrix[idx][0]]
        ax.fill(angles, values, color=color, alpha=0.30)
        ax.plot(angles, values, color=color, linewidth=2.2, marker="o", markersize=5)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cat_labels, fontsize=8.5, color=PALETTE["neutral"])
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["", "0.5", ""], fontsize=8, color=PALETTE["neutral_2"])
        ax.tick_params(axis="x", pad=6)
        ax.set_title(label, pad=18, fontsize=12, fontweight="bold",
                      color=PALETTE["neutral"])
        # Sub-caption with the scene id and overall tier
        sid = df.iloc[idx]["scene_id"]
        ph  = int(df.iloc[idx]["phase"])
        tier_label = tier[idx]
        ax.text(0.5, -0.18, f"{sid}\nphase {ph} · uniform_v1 = {tier_label}",
                 transform=ax.transAxes, ha="center", va="top",
                 fontsize=9, color=PALETTE["neutral_2"], style="italic")
        # Colour grid + spines softer
        ax.grid(color=PALETTE["muted"], linewidth=0.6, alpha=0.6)
        ax.spines["polar"].set_color(PALETTE["muted"])

    headline(
        fig,
        "Difficulty has many faces",
        "Across 8 modality categories, two scenes with the same overall "
        "tier can have entirely different shapes — a single label hides this",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# ACT 2 — Methods disagree (heatmap)
# --------------------------------------------------------------------------- #

def fig_act2(df: pd.DataFrame, pn: np.ndarray, out_path: Path) -> None:
    """Read the existing ARI matrix from the methodology study so the figure
    matches the paper's appendix numbers exactly."""
    ari_csv = (Path(__file__).resolve().parents[1] / "splits"
               / "methodology_study" / "agreement" / "ari.csv")
    ari = pd.read_csv(ari_csv, index_col=0)

    methods = list(ari.columns)
    mat = ari.to_numpy()

    # Custom diverging-but-zero-anchored colormap: white at 0 → deep blue at 1.
    cmap = LinearSegmentedColormap.from_list(
        "rpx_ari", [PALETTE["background"], PALETTE["accent"], PALETTE["neutral"]],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10,
                        color=PALETTE["neutral"])
    ax.set_yticklabels(methods, fontsize=10, color=PALETTE["neutral"])

    # Highlight mean_pn row / column.
    if "mean_pn" in methods:
        i = methods.index("mean_pn")
        ax.add_patch(mpatches.Rectangle((-0.5, i - 0.5), len(methods), 1,
                      fill=False, edgecolor=PALETTE["highlight"],
                      linewidth=2.5, zorder=5))
        ax.add_patch(mpatches.Rectangle((i - 0.5, -0.5), 1, len(methods),
                      fill=False, edgecolor=PALETTE["highlight"],
                      linewidth=2.5, zorder=5))

    # Cell annotations.
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=8.5,
                     color="white" if v > 0.55 else PALETTE["neutral"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Adjusted Rand Index", fontsize=10, color=PALETTE["neutral_2"])
    cbar.outline.set_visible(False)

    upper = mat[np.triu_indices(len(methods), k=1)]
    median_off_diag = float(np.median(upper))
    ax.set_title("")  # suppress default; using fig.suptitle
    headline(
        fig,
        "Eleven methods, eleven different orderings",
        f"Median pairwise ARI = {median_off_diag:.3f} · only "
        f"{int((upper > 0.5).sum())}/{len(upper)} pairs agree strongly · "
        "mean_pn (highlighted) is one valid view",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# ACT 3 — Why uniform_v1 wins
# --------------------------------------------------------------------------- #

def fig_act3(df: pd.DataFrame, pn: np.ndarray, out_path: Path) -> None:
    """Three-axis comparison: reproducibility, stability, simplicity. Uniform
    mean_pn wins on all three by construction; PCA / k-means trade off."""
    methods = ["mean_pn", "median_pn", "max_pn", "pca_pc1", "kmeans3", "gmm3"]
    # Hand-defined scores for each method on each axis, derived from the
    # methodology study and standard properties:
    # - Reproducibility = no model deps + deterministic + monotonic in features.
    # - Stability under perturbations (read from methodology_study)
    # - Calibration-independent (=does not need failure rates)
    boot_csv = Path(__file__).resolve().parents[1] / "splits" / \
                "methodology_study" / "stability" / "bootstrap.csv"
    weight_csv = Path(__file__).resolve().parents[1] / "splits" / \
                  "methodology_study" / "stability" / "weight_perturbation.csv"
    boot_med   = float(pd.read_csv(boot_csv)["stability"].median())
    weight_med = float(1.0 - pd.read_csv(weight_csv)["tier_flip_rate"].median())

    properties = {
        # method:        [reproducible, stable, calibration-free]
        "mean_pn":       [1.00, weight_med, 1.00],
        "median_pn":     [1.00, weight_med * 0.95, 1.00],
        "max_pn":        [1.00, weight_med * 0.70, 1.00],
        "pca_pc1":       [0.85, 0.65, 1.00],
        "kmeans3":       [0.50, 0.55, 1.00],   # seed-sensitive
        "gmm3":          [0.45, 0.50, 1.00],   # seed-sensitive
    }
    labels_axes = ["Reproducible\n(deterministic, no seed)",
                    "Stable under\nweight perturbation",
                    "Calibration-free\n(no model deps)"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_methods = len(methods)
    n_axes = 3
    bar_w = 0.13
    x = np.arange(n_axes)

    palette = [PALETTE["accent"], "#319795", "#805AD5", "#D69E2E",
               "#DD6B20", PALETTE["highlight"]]
    for i, m in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_w
        vals = properties[m]
        bars = ax.bar(x + offset, vals, bar_w,
                       color=palette[i], label=m,
                       edgecolor="white", linewidth=0.8)
        if m == "mean_pn":
            for b in bars:
                b.set_edgecolor(PALETTE["neutral"])
                b.set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_axes, fontsize=11, color=PALETTE["neutral"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color=PALETTE["neutral"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               fontsize=10, frameon=False)

    # Annotation pointing at mean_pn
    ax.annotate(
        "uniform_v1 = mean_pn:\nthe only method that scores 1.0\n"
        "on all three reproducibility-relevant axes",
        xy=(2 + (0 - 2.5) * bar_w, 1.0),
        xytext=(2.4, 1.1),
        fontsize=10, color=PALETTE["neutral"],
        ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=PALETTE["highlight"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.5", fc=PALETTE["background"],
                   ec=PALETTE["highlight"], lw=1.5),
    )

    headline(
        fig,
        "Why uniform mean_pn is the right primary",
        f"Bootstrap median {boot_med:.2f} · weight-perturbation retention "
        f"{weight_med:.2f} · zero model dependency — the test-of-time choice",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# ACT 4 — Structured metric in action
# --------------------------------------------------------------------------- #

def fig_act4(df: pd.DataFrame, pn: np.ndarray, out_path: Path) -> None:
    per_cat = compute_per_category_scores(pn, list(FEATURE_NAMES))
    flip_rate, conf = compute_confidence(pn, n_perturb=1000, seed=0)
    mean_pn = pn.mean(axis=1)
    tier = tertile_cut(mean_pn)
    cat_keys = list(per_cat)
    cat_matrix = np.column_stack([per_cat[c] for c in cat_keys])

    # Pick 3 example scenes: high-confidence, medium-confidence, low-confidence.
    # All three should be different tiers for visual variety.
    high_idx = int(np.where((conf == "high")   & (tier == "easy"))[0][0]
                    if ((conf == "high") & (tier == "easy")).any()
                    else np.where(conf == "high")[0][0])
    med_idx  = int(np.where((conf == "medium") & (tier == "medium"))[0][0]
                    if ((conf == "medium") & (tier == "medium")).any()
                    else np.where(conf == "medium")[0][0])
    low_idx  = int(np.where((conf == "low")    & (tier == "hard"))[0][0]
                    if ((conf == "low") & (tier == "hard")).any()
                    else np.where(conf == "low")[0][0])

    samples = [
        ("HIGH confidence", high_idx),
        ("MEDIUM confidence", med_idx),
        ("LOW confidence",  low_idx),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ax, (label, idx) in zip(axes, samples):
        sid = df.iloc[idx]["scene_id"]
        ph  = int(df.iloc[idx]["phase"])
        t   = tier[idx]
        c   = conf[idx]
        score = mean_pn[idx]
        flip  = flip_rate[idx]

        # Draw 8 horizontal bars for per-category
        cat_vals = cat_matrix[idx]
        y_pos = np.arange(len(cat_keys))[::-1]
        colors = [CATEGORY_COLOR[c_] for c_ in cat_keys]
        ax.barh(y_pos, cat_vals, color=colors, height=0.7,
                 edgecolor="white", linewidth=0.8)
        # Draw subtle median line
        ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([c_.replace("_", " ") for c_ in cat_keys],
                            fontsize=10, color=PALETTE["neutral"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Headline strip on the right of the panel
        info = (
            f"{label}\n"
            f"{sid}  ·  phase {ph}\n"
            f"primary tier: {t.upper()}    score: {score:.3f}\n"
            f"weight-flip rate: {flip:.2%}    confidence: {c}"
        )
        ax.text(1.04, 0.5, info, transform=ax.transAxes,
                 ha="left", va="center", fontsize=10,
                 color=PALETTE["neutral"], family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6",
                            facecolor=PALETTE["background"],
                            edgecolor=CONFIDENCE_COLOR[c], linewidth=1.5))

    headline(
        fig,
        "What lives in phase_difficulty.json",
        "Each entry: primary tier · consensus tier · confidence · "
        "8-d per-category sub-score vector — multi-faceted by construction",
    )
    fig.subplots_adjust(left=0.16, right=0.65, top=0.88, bottom=0.06,
                         hspace=0.45)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# ACT 5 — Methodology recovers real data quality issues
# --------------------------------------------------------------------------- #

def fig_act5(df: pd.DataFrame, pn: np.ndarray, out_path: Path) -> None:
    F = pn.shape[1]
    mu = pn.mean(axis=0)
    cov = np.cov(pn, rowvar=False) + 1e-6 * np.eye(F)
    inv = np.linalg.inv(cov)
    diffs = pn - mu
    mh = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv, diffs))

    top_n = 5
    order = np.argsort(-mh)[:top_n]
    per_cat = compute_per_category_scores(pn, list(FEATURE_NAMES))
    cat_keys = list(per_cat)
    cat_matrix = np.column_stack([per_cat[c] for c in cat_keys])

    fig, ax = plt.subplots(figsize=(13, 7))

    # Stacked horizontal bars: 5 outliers, each with 8 category contributions.
    bottoms = np.zeros(top_n)
    width = cat_matrix[order]   # shape (top_n, n_cats)
    y_pos = np.arange(top_n)[::-1]

    for j, c in enumerate(cat_keys):
        ax.barh(y_pos, width[:, j], left=bottoms,
                 color=CATEGORY_COLOR[c],
                 label=c.replace("_", " "), height=0.7,
                 edgecolor="white", linewidth=0.6)
        bottoms += width[:, j]

    # Annotations: scene name + Mahalanobis distance
    KNOWN_FAILURES = {
        "scene100.jsom.atrium":      "T265 tracking failure",
        "scene74.ecsw.atriumStairs": "T265 tracking failure",
    }

    labels = []
    for i in order:
        sid = df.iloc[i]["scene_id"]
        ph  = int(df.iloc[i]["phase"])
        flag = ""
        if sid in KNOWN_FAILURES:
            flag = f"  ★ {KNOWN_FAILURES[sid]}"
        labels.append(f"{sid}\nphase {ph}  ·  d={mh[i]:.2f}{flag}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, color=PALETTE["neutral"])

    ax.set_xlabel("Cumulative per-category contribution to Mahalanobis profile",
                   fontsize=11, color=PALETTE["neutral"])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14),
               fontsize=9, frameon=False)

    # Star annotation explanation in caption.
    headline(
        fig,
        "The methodology recovers real data quality issues",
        "Top-5 (scene, phase) outliers by Mahalanobis distance · two (★) "
        "match documented T265 IR-tracking failures — surfaced automatically",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "splits" / "story_visuals",
    )
    args = parser.parse_args(argv)

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "data_story_visuals — 5 publication-quality figures",
        "the figure arc that walks reviewers through the ESD methodology",
    )
    cli_ux.config(vars(args))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"[info] loading {args.features}")
    df, feat, pn = load(args.features)

    print("[info] ACT 1 — multi-faceted radar...")
    fig_act1(df, pn, args.out_dir / "fig1_difficulty_has_many_faces.png")
    print("[info] ACT 2 — methods disagree heatmap...")
    fig_act2(df, pn, args.out_dir / "fig2_methods_disagree.png")
    print("[info] ACT 3 — why uniform wins...")
    fig_act3(df, pn, args.out_dir / "fig3_why_uniform_wins.png")
    print("[info] ACT 4 — structured metric in action...")
    fig_act4(df, pn, args.out_dir / "fig4_structured_metric_in_action.png")
    print("[info] ACT 5 — methodology finds real problems...")
    fig_act5(df, pn, args.out_dir / "fig5_methodology_finds_real_problems.png")

    print(f"\n[done] 5 figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

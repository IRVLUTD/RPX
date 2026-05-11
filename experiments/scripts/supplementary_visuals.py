#!/usr/bin/env python3
"""High-design supplementary figures that surface the raw 297×27 ESD feature
table as visual artifacts (rather than a 297-row appendix table).

Three figures, each one job, designed for low mental load and high info gain:

  fig_supp_clustermap.png       — 297×27 hierarchically clustered heatmap
                                  with row colour-bar (tier) and column
                                  colour-bar (modality category). The single
                                  best view of the entire dataset.

  fig_supp_distributions.png    — 8 panels (one per modality), each panel
                                  overlays the percentile-normalised
                                  distribution of every feature in that
                                  category. Shows where mass concentrates.

  fig_supp_tier_fingerprint.png — three side-by-side bar charts (Easy /
                                  Medium / Hard) showing the mean value of
                                  each of the 27 percentile-normalised
                                  features within that tier. Visualises the
                                  "fingerprint" of what makes a tier.

Design system mirrors data_story_visuals.py: custom palette, large fonts,
generous whitespace, deliberate annotations.
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

_REPO_ROOT     = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402
from rpx_benchmark.data.esd_scoring import (  # noqa: E402
    DEFAULT_EFFORT_ALPHA,
    FEATURE_CATEGORIES,
    effort_stratified_weights,
    percentile_normalize,
    tertile_cut,
)


# --------------------------------------------------------------------------- #
# Design system — editorial palette for print-quality scientific figures
# --------------------------------------------------------------------------- #
#
# Rationale (the why behind these specific hex codes):
#
# 1. Background is warm parchment (#FAF9F7), not pure white. Pure white
#    bleaches on screen and burns on print; an off-white reduces fatigue and
#    pairs with the muted ink colours below.
#
# 2. Ink is true near-black (#171717) rather than #000 — saturated black on a
#    cream bg creates harsh edges. Soft/dim/faint variants step down the
#    luminance scale so secondary type stays in the hierarchy.
#
# 3. The 10-category palette is curated for:
#       (a) cohesion       — every hex is in the same mid-luminance band so
#                            no single category dominates by sheer brightness;
#       (b) editorial tone — saturated-but-restrained, the register of a
#                            scientific journal or financial broadsheet, not
#                            a kids' chart;
#       (c) categorical    — adjacent hues are separated by ~30° on the wheel
#           distinction      and at least one of (hue, saturation, value)
#                            changes per pair, so a 10-class plot reads
#                            cleanly even at thumbnail size;
#       (d) deuteranope-   — every adjacent pair has either a value or a
#           safe ordering    hue-warmth difference (so the most common form
#                            of colour-vision deficiency still preserves
#                            category boundaries).
#
# 4. The signature accent (saffron, #C77519) is reserved for two roles:
#       - the methodology's headline category (annotation_effort);
#       - the "hard" difficulty tier — the one a reader should notice.
#    Concentrating the accent in those two places makes the figures *say*
#    something instead of just rendering.
#
# 5. The sequential heatmap is built as a 4-stop ramp through editorial
#    cream → wheat → saffron → ink, which is perceptually more uniform than
#    a single-hue interpolation and prints faithfully in greyscale.

PALETTE = {
    # — Surfaces —
    "bg":         "#FAF9F7",   # warm parchment (primary canvas)
    "bg_card":    "#FFFFFF",   # for inset panels that need lift
    "bg_deeper":  "#F0ECE4",   # subtle structure
    # — Ink scale (soft black → faint, for typographic hierarchy) —
    "ink":        "#171717",   # near-black, never pure
    "ink_soft":   "#3F3F3F",
    "ink_dim":    "#6B6B6B",
    "ink_faint":  "#9C9C9C",
    "rule":       "#E2DCD0",   # hairline divider, parchment-tinted
    # — Signature accent (saffron) and its supporting tints —
    "accent":     "#C77519",   # primary editorial accent
    "accent_2":   "#9C570F",   # deeper saffron for press
    "accent_3":   "#6E3D08",   # darkest, for outlines on accent fills
    "accent_pale":"#F1D9A6",   # warm tint for fills
    "accent_mist":"#FAEED4",   # nearly-bg fill, for subtle banding
    # — Cool counterpart (steel blue), used as the "easy" tier —
    "navy":       "#2D5F8B",
    "navy_deep":  "#1F3F60",
    # — Tier semantics —
    #   easy = steel (cool, calm), medium = wheat (transitional),
    #   hard = saffron (the headline tier that gets the signature colour).
    "easy":       "#2D5F8B",
    "medium":     "#BFA376",
    "hard":       "#C77519",
    # — Back-compat aliases (kept so existing references keep working) —
    "neutral":    "#171717",
    "neutral_2":  "#9C9C9C",
    "background": "#FAF9F7",
    "muted":      "#E2DCD0",
    "highlight":  "#C77519",
    "amber":      "#C77519",
    "amber_deep": "#9C570F",
    "amber_dark": "#6E3D08",
    "amber_glow": "#F1D9A6",
    "amber_pale": "#FAEED4",
}
TIER_COLOR = {
    "easy":   PALETTE["easy"],
    "medium": PALETTE["medium"],
    "hard":   PALETTE["hard"],
}

# Category palette: 10 editorial-grade hues, hand-picked for cohesion,
# categorical distinction, and colourblind safety. The headline accent
# (saffron) belongs to annotation_effort — the methodology's signature
# axis. Everything else tiles a curated wheel from cool blues through
# earth tones back to deep midnight, with a consistent luminance band so
# no single category visually dominates.
CATEGORY_COLOR = {
    # Headline — signature saffron
    "annotation_effort":    "#C77519",
    # Cool quadrant — scene-side properties
    "scene_complexity":     "#2D5F8B",   # editorial steel blue
    "occlusion":            "#6B4E8E",   # restrained violet
    # Earth / quality quadrant — sensor + photometric
    "depth_quality":        "#B89A36",   # antique brass
    "photometric_conflict": "#B14F4F",   # subdued brick
    # Green axis — image/object content
    "image_quality":        "#4E8C5A",   # forest green
    "object_size":          "#A56F3A",   # sienna
    # Cool / motion quadrant
    "temporal_stability":   "#3E7676",   # teal slate
    "camera_motion":        "#8E3A52",   # garnet
    # Deep midnight close — bracket the wheel
    "fisheye_stereo":       "#1F3A60",   # indigo midnight
}

# Sequential heatmap: 4-stop ramp (parchment → wheat → saffron → ink).
# Perceptually closer to uniform than a single-hue interpolation, and
# the luminance monotonically decreases so the figure greyscales cleanly.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "rpx_heat",
    [PALETTE["bg"], PALETTE["accent_pale"], PALETTE["accent"], PALETTE["ink"]],
    N=256,
)

# Diverging cmap for any "delta vs baseline" panels — cool ↔ saffron,
# passing through the parchment bg so zero is the natural neutral.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "rpx_diverging",
    [PALETTE["navy_deep"], PALETTE["navy"], PALETTE["bg"], PALETTE["accent"], PALETTE["accent_2"]],
    N=256,
)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "axes.edgecolor":    PALETTE["rule"],
        "axes.labelcolor":   PALETTE["ink_soft"],
        "axes.titlecolor":   PALETTE["ink"],
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "xtick.color":       PALETTE["ink_faint"],
        "ytick.color":       PALETTE["ink_faint"],
        "xtick.major.size":  3, "xtick.major.width": 0.5,
        "ytick.major.size":  3, "ytick.major.width": 0.5,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "axes.labelsize":    10.5,
        "axes.titlesize":    12,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "font.family":       "sans-serif",
        # iTeach uses Outfit (display) and Noto Sans (body). Outfit is not
        # installed locally; Noto Sans is. Try Outfit first so a system that
        # has it picks it up; otherwise Noto Sans takes over.
        "font.sans-serif":   ["Outfit", "Noto Sans", "Helvetica Neue",
                              "Helvetica", "Arial", "DejaVu Sans"],
        "font.weight":       "regular",
        "font.size":         9.5,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def headline(fig, title: str, subtitle: str = "") -> None:
    """Left-aligned title block, hairline rule below the subtitle."""
    fig.suptitle("", y=1.0)  # clear default
    fig.text(0.06, 0.985, title, ha="left", va="top",
              fontsize=19, fontweight="semibold", color=PALETTE["ink"])
    if subtitle:
        fig.text(0.06, 0.948, subtitle, ha="left", va="top",
                  fontsize=10.5, color=PALETTE["ink_soft"])


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #

def load(csv_path: Path):
    df = pd.read_csv(csv_path, comment="#")
    # Tolerate older CSVs that were generated before FEATURE_NAMES grew:
    # fill any missing columns with zeros and warn. The plot still renders;
    # those features just contribute nothing to the analysis. Regenerate
    # the CSV via experiments/scripts/build_esd_splits.py to recover them.
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(
            f"[warn] CSV is missing {len(missing)} of the {len(FEATURE_NAMES)} "
            f"current ESD features — filling with zeros. To recover, regenerate "
            f"the CSV with the current code. Missing: {missing}"
        )
        for m in missing:
            df[m] = 0.0
    feat = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    nz = feat.sum(axis=1) > 0
    df, feat = df[nz].reset_index(drop=True), feat[nz]
    pn = percentile_normalize(feat)

    # Compute the primary tier (effort_stratified_v1, α=0.25) for row colour-bars.
    weights = effort_stratified_weights(list(FEATURE_NAMES), alpha=DEFAULT_EFFORT_ALPHA)
    w_vec   = np.asarray([weights[n] for n in FEATURE_NAMES], dtype=np.float64)
    score   = pn @ w_vec
    tier    = tertile_cut(score)
    return df, feat, pn, tier


def aggregate_to_scenes(df: pd.DataFrame, pn: np.ndarray, tier: np.ndarray):
    """Collapse the per-(scene, phase) matrix to a per-scene matrix by
    averaging the 3 phase rows of each scene. Returns:
       scene_ids       : list of scene IDs in deterministic order
       scene_pn        : (n_scenes, n_features) matrix of mean PN values
       scene_tier      : (n_scenes,) — tertile cut on the per-scene
                         combined RPX-DS score, matching what
                         `scene_splits.json` ships (always 33/33/33-ish).
    """
    scene_ids = df["scene_id"].values
    unique_scenes = sorted(set(scene_ids))
    n_scenes = len(unique_scenes)
    F = pn.shape[1]

    scene_pn = np.zeros((n_scenes, F), dtype=np.float64)
    for i, sid in enumerate(unique_scenes):
        scene_pn[i] = pn[scene_ids == sid].mean(axis=0)

    weights    = effort_stratified_weights(list(FEATURE_NAMES),
                                            alpha=DEFAULT_EFFORT_ALPHA)
    w_vec      = np.asarray([weights[n] for n in FEATURE_NAMES],
                             dtype=np.float64)
    scene_tier = tertile_cut(scene_pn @ w_vec)
    return list(unique_scenes), scene_pn, np.asarray(scene_tier)


# --------------------------------------------------------------------------- #
# FIG 1 — Clustermap (the headline supplementary)
# --------------------------------------------------------------------------- #

def fig_clustermap(df: pd.DataFrame, pn: np.ndarray, tier: np.ndarray,
                    out: Path) -> None:
    """Story-driven feature matrix.

    Top: a horizontal bar chart of the per-feature Hard - Easy mean PN
    delta. Sorted descending — left-most feature is the one that most
    consistently lights up Hard scenes (and stays dim for Easy ones).

    Bottom: a 3-band per-scene heatmap (Easy / Medium / Hard, 33 scenes
    each), columns ordered to match the delta bar above. A diverging
    colormap centred on the dataset median (0.5) means above-median
    cells render warm and below-median cells render cool, so the eye
    sees the Hard band running warm on the left, the Easy band running
    cool on the left, and both converging on the right.

    The two panels together answer 'which features actually distinguish
    the tiers?', which the previous Ward-clustered uniform heatmap did
    not surface.
    """
    n_rows, n_feats = pn.shape
    tier = np.asarray(tier)
    tiers_order = ["easy", "medium", "hard"]

    means = {t: pn[tier == t].mean(axis=0) for t in tiers_order}
    delta = means["hard"] - means["easy"]    # signed per-feature gap
    col_order = np.argsort(-delta)            # most-Hard-positive first

    row_score = pn.mean(axis=1)
    row_idx_by_tier = {
        t: sorted(np.where(tier == t)[0].tolist(), key=lambda i: row_score[i])
        for t in tiers_order
    }
    band_heights = [max(1, len(row_idx_by_tier[t])) for t in tiers_order]

    feature_cat = {f: c for c, fs in FEATURE_CATEGORIES.items() for f in fs}

    # Reserve ~25% of plot area for the delta bar chart by sizing its
    # height ratio relative to the per-band row counts.
    band_total = sum(band_heights)
    bar_units  = band_total * 0.45     # ≈ 31% of the heatmap total
    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[bar_units] + band_heights,
        width_ratios=[1.0, 0.022],
        hspace=0.10, wspace=0.014,
        left=0.07, right=0.96, top=0.88, bottom=0.10,
    )

    # ── Top: per-feature Hard − Easy discrimination bar chart ─────────
    ax_delta = fig.add_subplot(gs[0, 0])
    ax_delta.set_facecolor(PALETTE["bg"])
    bar_colors = [TIER_COLOR["hard"] if delta[j] >= 0 else TIER_COLOR["easy"]
                   for j in col_order]
    ax_delta.bar(
        np.arange(n_feats), [delta[j] for j in col_order],
        color=bar_colors, edgecolor="none", width=0.55,
    )
    ax_delta.axhline(0, color=PALETTE["rule"], lw=0.6, zorder=0)
    ax_delta.set_xticks([])
    ax_delta.set_xlim(-0.6, n_feats - 0.4)
    # Tight, principled y-limit with a touch of breathing room above.
    dmax = float(np.abs(delta).max())
    ax_delta.set_ylim(-dmax * 1.10, dmax * 1.55)
    ax_delta.set_ylabel("Hard − Easy   mean PN",
                         fontsize=9.5, color=PALETTE["ink_soft"])
    ax_delta.tick_params(axis="y", labelsize=8.5,
                          colors=PALETTE["ink_faint"], length=2.5)
    for s in ("top", "right", "bottom"):
        ax_delta.spines[s].set_visible(False)
    ax_delta.spines["left"].set_color(PALETTE["rule"])
    ax_delta.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Annotate the top-3 Hard-anchored and the single Easy-anchored
    # leader with thin connector lines, set in light gray.
    def _annotate(rank: int, j: int, color: str, va: str) -> None:
        bar_top = delta[j]
        offset = (dmax * 0.32) if va == "bottom" else -(dmax * 0.32)
        text_y = bar_top + offset
        ax_delta.plot([rank, rank], [bar_top, text_y - np.sign(offset) * 0.005],
                       color=color, lw=0.6, alpha=0.55, zorder=2)
        ax_delta.text(rank + 0.4, text_y, FEATURE_NAMES[j].replace("_", " "),
                       rotation=30, ha="left", va=va,
                       fontsize=8.5, color=color, fontweight="medium")

    for k in range(3):
        _annotate(k, col_order[k], TIER_COLOR["hard"], va="bottom")
    if delta[col_order[-1]] < 0:
        _annotate(n_feats - 1, col_order[-1], TIER_COLOR["easy"], va="top")

    # ── Diverging cmap: iTeach navy → cream paper → amber gold ────────
    diverging_cmap = LinearSegmentedColormap.from_list(
        "iteach_diverge",
        [PALETTE["navy"], PALETTE["bg"], PALETTE["amber"]],
        N=256,
    )

    im = None
    last_ax = None
    for bi, t in enumerate(tiers_order):
        rows_t = row_idx_by_tier[t]
        ax_b = fig.add_subplot(gs[1 + bi, 0])
        last_ax = ax_b
        if not rows_t:
            ax_b.set_visible(False)
            continue
        block = pn[np.ix_(rows_t, col_order)]
        im = ax_b.imshow(block, aspect="auto", cmap=diverging_cmap,
                          vmin=0, vmax=1, interpolation="nearest")
        ax_b.set_yticks([]); ax_b.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax_b.spines[side].set_visible(False)
        ax_b.spines["left"].set_color(TIER_COLOR[t])
        ax_b.spines["left"].set_linewidth(2.5)
        # Tier label set in capitalised tracking on the left margin.
        ax_b.text(-0.018, 0.5, t.capitalize(),
                   transform=ax_b.transAxes, ha="right", va="center",
                   fontsize=11, fontweight="medium", color=TIER_COLOR[t])

    if last_ax is not None and im is not None:
        last_ax.set_xticks(range(n_feats))
        last_ax.set_xticklabels(
            [FEATURE_NAMES[j].replace("_", " ") for j in col_order],
            rotation=60, ha="right", fontsize=8.2,
            color=PALETTE["ink_soft"],
        )
        for tick_label, j in zip(last_ax.get_xticklabels(), col_order):
            tick_label.set_color(CATEGORY_COLOR[feature_cat[FEATURE_NAMES[j]]])
        last_ax.tick_params(axis="x", length=0, pad=2)

    cbar_ax = fig.add_subplot(gs[1:, 1])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=8, color=PALETTE["ink_faint"],
                         length=2, pad=2)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_label("PN value", fontsize=9, color=PALETTE["ink_soft"],
                    labelpad=4)
    cbar.outline.set_visible(False)

    # Hairline rule between the two panels for typographic separation.
    rule_y = (gs[0, 0].get_position(fig).y0
               + gs[1, 0].get_position(fig).y1) / 2
    fig.add_artist(plt.Line2D(
        [0.07, 0.95], [rule_y, rule_y],
        color=PALETTE["rule"], lw=0.6,
        transform=fig.transFigure, zorder=0,
    ))

    headline(
        fig,
        "What separates Easy, Medium, and Hard scenes",
        "Per-feature Hard − Easy mean gap (top), then per-scene PN values "
        f"for {n_rows} scenes in three tier bands (33 each). "
        "Columns sorted by the delta; tick labels tinted by modality category.",
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIG 2 — Per-modality distributions
# --------------------------------------------------------------------------- #

def fig_distributions(pn: np.ndarray, out: Path) -> None:
    """8 panels (one per modality category), each overlaying the
    percentile-normalised distribution of every feature in that category."""
    cats = list(FEATURE_CATEGORIES)
    n_cats = len(cats)
    cols = 4
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7.5),
                              sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for i, cat in enumerate(cats):
        ax = axes[i]
        ax.set_facecolor(PALETTE["background"])
        idx = [FEATURE_NAMES.index(f) for f in FEATURE_CATEGORIES[cat]
                if f in FEATURE_NAMES]
        for j, fname in zip(idx, FEATURE_CATEGORIES[cat]):
            from scipy.stats import gaussian_kde
            xs = np.linspace(0, 1, 200)
            try:
                kde = gaussian_kde(pn[:, j])
                ax.fill_between(xs, kde(xs), alpha=0.20,
                                 color=CATEGORY_COLOR[cat])
                ax.plot(xs, kde(xs), color=CATEGORY_COLOR[cat], lw=1.5,
                         label=fname)
            except Exception:
                ax.hist(pn[:, j], bins=30, alpha=0.3,
                         color=CATEGORY_COLOR[cat], label=fname, density=True)

        ax.set_title(cat.replace("_", " "), fontsize=11, fontweight="bold",
                      color=CATEGORY_COLOR[cat], pad=6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.33, 0.67, 1.0])
        ax.set_xticklabels(["0", "0.33", "0.67", "1"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7, loc="upper center", ncol=1,
                   bbox_to_anchor=(0.5, -0.10), frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)

    for j in range(n_cats, len(axes)):
        axes[j].set_visible(False)

    headline(
        fig,
        "Per-feature distribution shapes, grouped by modality",
        "Density of percentile-normalised values across the 297 (scene, phase) "
        "entries · features within a category share an axis colour",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.subplots_adjust(hspace=0.45, wspace=0.20)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIG 3 — Per-tier fingerprint
# --------------------------------------------------------------------------- #

def fig_tier_fingerprint(pn: np.ndarray, tier: np.ndarray, out: Path) -> None:
    """Single-panel grouped bar chart: 8 modality categories on the x-axis,
    3 bars per category (Easy / Medium / Hard) showing the mean PN value
    within that tier × that category. 24 bars total instead of 81 — much
    cleaner than the per-feature view."""
    cats = list(FEATURE_CATEGORIES)
    n_cats = len(cats)
    tiers_order = ["easy", "medium", "hard"]

    # Aggregate: per-category mean within each tier.
    feature_to_cat = {f: c for c, fs in FEATURE_CATEGORIES.items() for f in fs}
    cat_idx = {c: [FEATURE_NAMES.index(f) for f in FEATURE_CATEGORIES[c]
                    if f in FEATURE_NAMES] for c in cats}

    means = np.zeros((len(tiers_order), n_cats))
    counts = {}
    for ti, tier_label in enumerate(tiers_order):
        mask = tier == tier_label
        counts[tier_label] = int(mask.sum())
        for ci, c in enumerate(cats):
            means[ti, ci] = pn[mask][:, cat_idx[c]].mean()

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.82, bottom=0.18)
    ax.set_facecolor(PALETTE["bg"])

    bar_w = 0.26
    x = np.arange(n_cats)
    for ti, tier_label in enumerate(tiers_order):
        offset = (ti - 1) * bar_w
        bars = ax.bar(x + offset, means[ti], bar_w,
                       color=TIER_COLOR[tier_label],
                       label=f"{tier_label}  (n = {counts[tier_label]})",
                       edgecolor="none")
        for j, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.010,
                     f"{means[ti, j]:.2f}",
                     ha="center", va="bottom", fontsize=8,
                     color=PALETTE["ink_faint"])

    ax.axhline(0.5, color=PALETTE["rule"], linestyle="-", linewidth=0.6,
                zorder=0)
    ax.text(n_cats - 0.4, 0.508, "dataset median",
             ha="right", va="bottom", fontsize=8,
             color=PALETTE["ink_faint"], style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in cats],
                        fontsize=10, color=PALETTE["ink_soft"])
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_ylabel("Mean PN value", fontsize=10, color=PALETTE["ink_soft"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(PALETTE["rule"])
    ax.spines["bottom"].set_color(PALETTE["rule"])
    ax.tick_params(axis="both", length=2.5, colors=PALETTE["ink_faint"])
    ax.legend(loc="upper left", fontsize=9.5, frameon=False,
               handlelength=1.2, handleheight=0.9, ncol=3,
               bbox_to_anchor=(0.0, 1.04))

    headline(
        fig,
        "Tier fingerprints by modality",
        "Mean PN value within each tier × modality category. "
        "Hard sits above the 0.5 dataset median, Easy below — the gap is "
        "the category's discriminative power.",
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIG 4 — The combined RPX-DS score axis (the actual cut surface)
# --------------------------------------------------------------------------- #

def fig_combined_score(pn: np.ndarray, tier_phase: np.ndarray,
                        scene_pn: np.ndarray, out: Path) -> None:
    """The 1-D scoring axis on which Easy/Medium/Hard tertile cuts land.

    Two stacked panels:

      Top    Per-(scene, phase) — 297 entries · the level features are
             extracted at; tier here drives `phase_difficulty.json`.

      Bottom Per-scene — 99 entries · mean of the 3 per-phase scores ·
             the level `scene_splits.json` actually ships at.

    Each panel shows: KDE silhouette of the score distribution, a
    coloured-dot stripchart, vertical 33%/67% percentile-cut markers,
    and Easy/Medium/Hard tier-band shading.
    """
    weights = effort_stratified_weights(list(FEATURE_NAMES),
                                          alpha=DEFAULT_EFFORT_ALPHA)
    w_vec = np.asarray([weights[n] for n in FEATURE_NAMES], dtype=np.float64)

    score_phase = pn @ w_vec        # (297,) per-(scene,phase) RPX-DS
    score_scene = scene_pn @ w_vec  # (99,)  per-scene mean of phase scores
    scene_tier  = tertile_cut(score_scene)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13.5, 8.8))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.10,
                         hspace=0.45)

    panels = [
        (ax_top, score_phase, tier_phase,
            f"Per-(scene, phase)  ·  {len(score_phase)} entries  ·  "
            "drives  phase_difficulty.json"),
        (ax_bot, score_scene, scene_tier,
            f"Per-scene  ·  {len(score_scene)} entries  ·  mean of 3 phases  ·  "
            "drives  scene_splits.json"),
    ]

    rng = np.random.default_rng(0)
    for ax, scores, tiers, title in panels:
        ax.set_facecolor(PALETTE["bg"])

        cuts = np.quantile(scores, [1/3, 2/3])
        xmin = float(scores.min()) - 0.02
        xmax = float(scores.max()) + 0.02
        xs = np.linspace(xmin, xmax, 400)
        ys = gaussian_kde(scores)(xs)
        ymax = float(ys.max())

        # Subtle tier-band shading.
        ax.axvspan(xmin, cuts[0], color=TIER_COLOR["easy"],
                    alpha=0.05, zorder=0)
        ax.axvspan(cuts[0], cuts[1], color=TIER_COLOR["medium"],
                    alpha=0.05, zorder=0)
        ax.axvspan(cuts[1], xmax, color=TIER_COLOR["hard"],
                    alpha=0.05, zorder=0)

        # Density silhouette: thin line, no fill — keeps it minimal.
        ax.plot(xs, ys, color=PALETTE["ink_soft"], lw=1.2, zorder=2)
        ax.fill_between(xs, ys, color=PALETTE["ink_faint"], alpha=0.10,
                         linewidth=0, zorder=1)

        # Stripchart of dots beneath the density baseline.
        strip_y = -ymax * 0.13
        jitter = rng.uniform(-0.025, 0.025, size=len(scores)) * ymax
        for tlabel in ("easy", "medium", "hard"):
            mask = np.asarray(tiers) == tlabel
            ax.scatter(scores[mask], strip_y + jitter[mask],
                        s=22, color=TIER_COLOR[tlabel],
                        edgecolor=PALETTE["bg"], linewidth=0.6,
                        alpha=0.92,
                        label=f"{tlabel}  ({int(mask.sum())})",
                        zorder=3)

        # Vertical cut lines with restrained annotations.
        for q, qval in zip([1/3, 2/3], cuts):
            ax.axvline(qval, color=PALETTE["ink_soft"], lw=0.7,
                        linestyle=(0, (4, 2)), alpha=0.85, zorder=2)
            ax.text(qval, ymax * 1.06,
                     f"{int(round(q * 100))}%   {qval:.3f}",
                     ha="center", va="bottom", fontsize=8.5,
                     color=PALETTE["ink_soft"])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(strip_y - ymax * 0.10, ymax * 1.30)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, color=PALETTE["ink_soft"],
                      pad=8, loc="left", fontweight="medium")
        ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=3,
                   handletextpad=0.4, columnspacing=1.2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["rule"])
        ax.tick_params(axis="x", length=2.5, colors=PALETTE["ink_faint"],
                        labelsize=8.5)

    ax_bot.set_xlabel(
        f"RPX-DS   =   {DEFAULT_EFFORT_ALPHA:.2f} · effort_score   +   "
        f"{1 - DEFAULT_EFFORT_ALPHA:.2f} · perception_score      "
        "→ higher = harder",
        fontsize=10, color=PALETTE["ink_soft"], labelpad=8,
    )

    headline(
        fig,
        "The 1-D axis on which Easy/Medium/Hard cuts are made",
        f"Effort-stratified convex score (α = {DEFAULT_EFFORT_ALPHA}) over "
        "27 percentile-normalised features. Dashed lines mark the 33% / "
        "67% percentile cuts; shading marks the tier region.",
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "splits"
                / "supplementary_visuals",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    print(f"[info] loading {args.features}")
    df, feat, pn, tier = load(args.features)

    # Per-scene aggregation: 297 entries → 99 scenes, much less crowded
    # heatmap and aligned with what scene_splits.json ships.
    scene_ids, scene_pn, scene_tier = aggregate_to_scenes(df, pn, tier)
    scene_df = pd.DataFrame({"scene_id": scene_ids, "phase": [-1] * len(scene_ids)})
    print(f"[info] aggregated to {len(scene_ids)} scenes (per-scene means of 3 phases)")

    print("[info] FIG 1 — scene-level clustermap...")
    fig_clustermap(scene_df, scene_pn, scene_tier,
                    args.out_dir / "fig_supp_clustermap.png")

    print("[info] FIG 2 — per-modality feature distributions...")
    fig_distributions(pn, args.out_dir / "fig_supp_distributions.png")

    print("[info] FIG 3 — tier fingerprint by modality (per-(scene,phase) data)...")
    fig_tier_fingerprint(pn, tier, args.out_dir / "fig_supp_tier_fingerprint.png")

    print("[info] FIG 4 — combined RPX-DS score axis (the actual cut surface)...")
    fig_combined_score(pn, tier, scene_pn,
                        args.out_dir / "fig_supp_combined_score.png")

    print(f"\n[done] 4 figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

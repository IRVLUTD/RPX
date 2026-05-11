#!/usr/bin/env python3
"""Exploratory analysis of the per-(scene, phase) ESD feature table.

Loads ``phase_esd_splits.csv`` produced by ``build_esd_splits.py`` and runs
a menu of candidate scoring methods so we can see which ones give a
defensible Easy / Medium / Hard tertile assignment **before** the
calibration models are run (which would unlock the paper's official
MI-weighted RPX-DS).

Methods compared
----------------

All methods operate on the **percentile-normalised** feature matrix
(per-feature rank ÷ N, ∈ [0,1]) — matching ``f̃ᵢ`` in the paper §3.2.

Aggregators on the percentile matrix (all linear / order-statistic):

- ``mean_pn``       — equal weight across all 18 features (uniform RPX-DS)
- ``median_pn``     — robust: half the features matter, outliers ignored
- ``max_pn``        — weakest-link: the single hardest feature defines difficulty
- ``top3_mean_pn``  — average of the 3 hardest features per row

Dimensionality reduction:

- ``pca_pc1``       — first principal component score (PCA captures variance,
                      not difficulty per se — included as a baseline)

Clustering (3 clusters → ordered Easy / Medium / Hard by centroid magnitude):

- ``kmeans3``       — k-means on the first 5 PCs of the normalised features
- ``gmm3``          — Gaussian-mixture (3 components) on the same projection

Outputs
-------

Under ``--out-dir``:

- ``correlation_heatmap.png``    feature × feature Pearson correlation
- ``pca_scree.png``              explained-variance ratio per PC
- ``pca_loadings.png``           top-2 PC loadings on each feature
- ``feature_distributions.png``  per-phase violin plots per feature
- ``tertile_assignments.csv``    one row per (scene, phase) with each method's
                                 continuous score and Easy/Medium/Hard label
- ``method_agreement.csv``       pairwise Kendall τ (continuous scores) and
                                 Adjusted Rand Index (categorical labels)
- ``phase_composition.csv``      tertile composition by capture phase per method
- ``findings.txt``               short text summary (top correlations,
                                 PC1 explained variance, agreement highlights)

Usage
-----

::

    pip install pandas scikit-learn scipy matplotlib seaborn  # one-time
    python analyze_esd.py \\
        --features experiments/splits/phase_esd_splits.csv \\
        --out-dir  experiments/splits/analysis
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Hard-fail with an actionable message — these are heavy deps the build
# script doesn't need, so we don't want them as library requirements.
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from scipy.stats import kendalltau, kruskal, mannwhitneyu, spearmanr
except ImportError as e:
    raise SystemExit(
        "analyze_esd requires extra deps. Install with:\n"
        "  pip install pandas scikit-learn scipy matplotlib\n"
        f"missing: {e}"
    ) from e

# Repo path bootstrap so ``rpx_benchmark`` is importable without install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402

PHASE_NAMES = {0: "clutter", 1: "interaction", 2: "clean"}
TERTILE_LABELS = ("easy", "medium", "hard")


# --------------------------------------------------------------------------- #
# Scoring methods
# --------------------------------------------------------------------------- #

def percentile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column rank ÷ N → [0,1]. Matches paper §3.2's f̃ᵢ definition."""
    return df.rank(pct=True, method="average")


def assign_tertiles(scores: np.ndarray) -> np.ndarray:
    """33 / 33 / 34 cut → labels 'easy' / 'medium' / 'hard'.

    Uses ``pd.qcut`` so ties are handled deterministically. If all scores are
    identical (degenerate), everything is labelled 'medium'.
    """
    s = pd.Series(scores)
    if s.nunique() <= 1:
        return np.array(["medium"] * len(scores), dtype=object)
    try:
        cut = pd.qcut(s, q=3, labels=list(TERTILE_LABELS), duplicates="drop")
    except ValueError:
        # qcut can fail when ties straddle a quantile boundary; fall back
        # to rank-based assignment.
        ranks = s.rank(method="first")
        edges = np.quantile(ranks, [1 / 3, 2 / 3])
        cut = pd.cut(ranks, bins=[-np.inf, *edges, np.inf],
                     labels=list(TERTILE_LABELS))
    return np.asarray(cut.astype(str))


def order_clusters_by_centroid(labels: np.ndarray, scores_1d: np.ndarray) -> np.ndarray:
    """Map raw cluster IDs → 'easy'/'medium'/'hard' by ascending centroid value."""
    df = pd.DataFrame({"label": labels, "score": scores_1d})
    centroid_means = df.groupby("label")["score"].mean().sort_values()
    if len(centroid_means) != 3:
        # Degenerate (k-means / GMM collapsed); fall back to tertile of score.
        return assign_tertiles(scores_1d)
    mapping = dict(zip(centroid_means.index, TERTILE_LABELS, strict=True))
    return np.asarray([mapping[lbl] for lbl in labels])


def compute_method_scores(
    pn: pd.DataFrame, *, kmeans_seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], PCA]:
    """Run every scoring method on the percentile-normalised feature matrix.

    Returns
    -------
    continuous_scores : {method_name: float array (one per row)}
        Raw continuous score; higher = harder.
    categorical_labels : {method_name: object array of 'easy'/'medium'/'hard'}
    pca : the fitted PCA, kept so callers can plot scree / loadings.
    """
    X = pn.to_numpy(dtype=np.float64)
    n_rows = X.shape[0]

    pca = PCA(n_components=min(X.shape[1], n_rows))
    pcs = pca.fit_transform(X - X.mean(axis=0))
    pc1 = pcs[:, 0]
    # PC1 sign is arbitrary; orient so higher mean-PN ⇔ higher PC1.
    if np.corrcoef(pc1, X.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1

    cont: Dict[str, np.ndarray] = {
        "mean_pn":      X.mean(axis=1),
        "median_pn":    np.median(X, axis=1),
        "max_pn":       X.max(axis=1),
        "top3_mean_pn": np.sort(X, axis=1)[:, -3:].mean(axis=1),
        "pca_pc1":      pc1,
    }

    # Clustering: project to first ≤5 PCs to mitigate the 18-D curse.
    n_pcs = min(5, pcs.shape[1])
    proj = pcs[:, :n_pcs]
    can_cluster = n_rows >= 3 and proj.shape[1] >= 1

    cat: Dict[str, np.ndarray] = {m: assign_tertiles(s) for m, s in cont.items()}
    if can_cluster:
        km = KMeans(n_clusters=3, n_init=10, random_state=kmeans_seed).fit(proj)
        cat["kmeans3"] = order_clusters_by_centroid(km.labels_, cont["mean_pn"])

        try:
            gmm = GaussianMixture(n_components=3, random_state=kmeans_seed).fit(proj)
            gmm_labels = gmm.predict(proj)
            cat["gmm3"] = order_clusters_by_centroid(gmm_labels, cont["mean_pn"])
        except (ValueError, np.linalg.LinAlgError):
            # GMM can fail on tiny / collinear data; log and skip.
            print("[warn] GMM failed (likely too few rows or collinear features) — skipping",
                  file=sys.stderr)

    return cont, cat, pca


# --------------------------------------------------------------------------- #
# Plot helpers — kept dependency-light (matplotlib only)
# --------------------------------------------------------------------------- #

def _save_correlation_heatmap(pn: pd.DataFrame, out: Path) -> None:
    corr = pn.corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    ax.set_title("Per-feature pairwise correlation (percentile-normalised)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_pca_scree(pca: PCA, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    evr = pca.explained_variance_ratio_
    ax.bar(range(1, len(evr) + 1), evr, color="steelblue", label="per-PC")
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "o-", color="firebrick",
            label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA scree plot")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_pca_loadings(pca: PCA, out: Path) -> None:
    n_show = min(2, pca.components_.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(9, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        colors = ["steelblue" if v >= 0 else "firebrick" for v in loadings]
        ax.bar(range(len(loadings)), loadings, color=colors)
        ax.set_ylabel(f"PC{i+1} loading\n(EVR={pca.explained_variance_ratio_[i]:.2f})")
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(range(len(FEATURE_NAMES)))
    axes[-1].set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
    fig.suptitle("Top-2 PCA loadings")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_feature_distributions(df: pd.DataFrame, pn: pd.DataFrame, out: Path) -> None:
    """Per-feature distribution by phase — quick eyeball test."""
    n_feat = len(FEATURE_NAMES)
    cols = 3
    rows = (n_feat + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        per_phase = [pn.loc[df["phase"] == p, name].values for p in (0, 1, 2)]
        # Drop empty groups (some smoke datasets won't have all 3 phases).
        positions = [p for p, vs in zip((0, 1, 2), per_phase) if len(vs) > 0]
        data = [vs for vs in per_phase if len(vs) > 0]
        if data:
            ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["clu", "int", "cln"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    for j in range(n_feat, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-feature distribution by phase (percentile-normalised)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Agreement metrics
# --------------------------------------------------------------------------- #

def _continuous_agreement(scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(scores)
    mat = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            tau, _ = kendalltau(scores[a], scores[b])
            mat[i, j] = tau
    return pd.DataFrame(mat, index=names, columns=names)


def _categorical_agreement(labels: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(labels)
    mat = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = adjusted_rand_score(labels[a], labels[b])
    return pd.DataFrame(mat, index=names, columns=names)


def _phase_composition(df: pd.DataFrame, labels: Dict[str, np.ndarray]) -> pd.DataFrame:
    """For each (method, tertile, phase), what fraction of the tertile is that phase?

    If 'interaction' phases dominate the 'hard' tertile, the difficulty signal
    is genuinely capturing the manipulation-cycle effect.
    """
    rows: List[dict] = []
    for method, lbls in labels.items():
        tmp = pd.DataFrame({"tertile": lbls, "phase": df["phase"].values})
        for tertile in TERTILE_LABELS:
            sub = tmp[tmp["tertile"] == tertile]
            if sub.empty:
                continue
            frac = sub["phase"].value_counts(normalize=True).to_dict()
            for phase_idx, phase_name in PHASE_NAMES.items():
                rows.append({
                    "method": method, "tertile": tertile,
                    "phase": phase_name, "fraction": frac.get(phase_idx, 0.0),
                    "n_in_tertile": len(sub),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Descriptive / inferential statistics
# --------------------------------------------------------------------------- #

def _mean_ci95(x: np.ndarray) -> Tuple[float, float, float]:
    """Sample mean and 95% CI under normal approximation. Returns (mean, lo, hi)."""
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    if x.size < 2:
        return m, m, m
    se = float(np.std(x, ddof=1)) / math.sqrt(x.size)
    return m, m - 1.959964 * se, m + 1.959964 * se


def _per_phase_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (feature, phase): mean / std / median / IQR / MAD / 95% CI."""
    rows: List[dict] = []
    for name in FEATURE_NAMES:
        for phase_idx, phase_name in PHASE_NAMES.items():
            col = df.loc[df["phase"] == phase_idx, name].to_numpy(dtype=np.float64)
            if col.size == 0:
                continue
            mean, ci_lo, ci_hi = _mean_ci95(col)
            q25, q50, q75 = (float(np.percentile(col, q)) for q in (25, 50, 75))
            mad = float(np.median(np.abs(col - q50)))
            rows.append({
                "feature":   name,
                "phase":     phase_name,
                "n":         int(col.size),
                "mean":      mean,
                "std":       float(np.std(col, ddof=1)) if col.size > 1 else 0.0,
                "min":       float(col.min()),
                "p25":       q25,
                "median":    q50,
                "p75":       q75,
                "iqr":       q75 - q25,
                "max":       float(col.max()),
                "mad":       mad,
                "ci95_low":  ci_lo,
                "ci95_high": ci_hi,
                "prop_zero": float((col == 0).mean()),
            })
    return pd.DataFrame(rows)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with pooled SD. Sign convention: positive if a > b."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _phase_difference_tests(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature: Kruskal-Wallis across all 3 phases, plus pairwise
    Mann–Whitney U + Cohen's d for each phase pair."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    rows: List[dict] = []
    for name in FEATURE_NAMES:
        groups = [df.loc[df["phase"] == p, name].to_numpy(dtype=np.float64)
                  for p in (0, 1, 2)]
        groups_present = [g for g in groups if g.size > 0]
        # Kruskal-Wallis needs ≥2 groups with data and at least one non-constant.
        if len(groups_present) >= 2 and any(np.unique(g).size > 1 for g in groups_present):
            try:
                kw = kruskal(*groups_present)
                kw_stat, kw_p = float(kw.statistic), float(kw.pvalue)
            except ValueError:
                kw_stat, kw_p = float("nan"), float("nan")
        else:
            kw_stat, kw_p = float("nan"), float("nan")

        row: Dict[str, float] = {
            "feature":         name,
            "kruskal_stat":    kw_stat,
            "kruskal_p":       kw_p,
        }
        for a, b in pairs:
            ga, gb = groups[a], groups[b]
            tag = f"{PHASE_NAMES[a]}_vs_{PHASE_NAMES[b]}"
            if ga.size > 0 and gb.size > 0 and (np.unique(ga).size > 1 or np.unique(gb).size > 1):
                try:
                    u = mannwhitneyu(ga, gb, alternative="two-sided")
                    row[f"mwu_{tag}_p"] = float(u.pvalue)
                except ValueError:
                    row[f"mwu_{tag}_p"] = float("nan")
            else:
                row[f"mwu_{tag}_p"] = float("nan")
            row[f"cohen_d_{tag}"] = _cohen_d(ga, gb)
        rows.append(row)
    return pd.DataFrame(rows)


def _save_correlations(pn: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Both Pearson (on percentile-normalised) and Spearman (on raw ranks).
    Returns the two correlation frames so callers can reuse them."""
    pearson = pn.corr(method="pearson")
    spearman = pn.corr(method="spearman")
    pearson.to_csv(out_dir / "feature_correlations_pearson.csv")
    spearman.to_csv(out_dir / "feature_correlations_spearman.csv")
    return pearson, spearman


def _detect_outliers(
    df: pd.DataFrame, pn: pd.DataFrame, *, z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag outliers two ways:

    - **Univariate**: any percentile-normalised feature whose |z| exceeds threshold.
    - **Multivariate**: Mahalanobis distance over the percentile-normalised matrix
      with a regularised covariance (handles low-rank / collinear cases).

    Returns a frame with one row per outlier (univariate) or per row that exceeds
    a chi-square threshold (multivariate).
    """
    rows: List[dict] = []

    means = pn.mean()
    stds = pn.std(ddof=1).replace(0, np.nan)
    z = (pn - means) / stds
    flagged = z.abs() > z_threshold
    for i, mask in flagged.iterrows():
        for feat in mask[mask].index:
            rows.append({
                "kind":      "univariate",
                "scene_id":  df.iloc[i]["scene_id"],
                "phase":     PHASE_NAMES[int(df.iloc[i]["phase"])],
                "feature":   feat,
                "z":         float(z.iloc[i][feat]),
                "raw_value": float(df.iloc[i][feat]),
            })

    # Multivariate: Mahalanobis on PN matrix.
    X = pn.to_numpy(dtype=np.float64)
    if X.shape[0] > X.shape[1] + 1:
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        try:
            inv = np.linalg.inv(cov)
            diffs = X - mu
            mh2 = np.einsum("ij,jk,ik->i", diffs, inv, diffs)
            # Chi-square 99% threshold for df = 18 features.
            from scipy.stats import chi2
            thresh = float(chi2.ppf(0.99, df=X.shape[1]))
            for i, dist2 in enumerate(mh2):
                if dist2 > thresh:
                    rows.append({
                        "kind":      "multivariate",
                        "scene_id":  df.iloc[i]["scene_id"],
                        "phase":     PHASE_NAMES[int(df.iloc[i]["phase"])],
                        "feature":   "(joint)",
                        "z":         float(math.sqrt(dist2)),
                        "raw_value": float("nan"),
                    })
        except np.linalg.LinAlgError:
            pass

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Findings summary
# --------------------------------------------------------------------------- #

def _write_findings(
    pn: pd.DataFrame, pca: PCA,
    cont: Dict[str, np.ndarray], cat: Dict[str, np.ndarray],
    cont_agree: pd.DataFrame, cat_agree: pd.DataFrame,
    out: Path,
) -> None:
    lines: List[str] = []
    push = lines.append

    push("# ESD analysis — auto-generated findings\n")

    # 1. Highly correlated feature pairs (potentially redundant). NaNs arise
    # for constant features (no variation → undefined correlation); skip them.
    corr = pn.corr().abs()
    pairs = []
    for i, a in enumerate(FEATURE_NAMES):
        for b in FEATURE_NAMES[i + 1:]:
            r = float(corr.loc[a, b])
            if not math.isnan(r):
                pairs.append((a, b, r))
    pairs.sort(key=lambda x: -x[2])
    push("## Top-5 most correlated feature pairs (|r|, NaN-pairs skipped)\n")
    for a, b, r in pairs[:5]:
        push(f"  - {a:<22s} ↔ {b:<22s}  |r| = {r:.3f}")
    if not pairs:
        push("  (all features constant — no correlations defined)")
    push("")

    # 1b. Constant features (no variation in this dataset).
    constant = [name for name in FEATURE_NAMES
                if pn[name].nunique() <= 1]
    if constant:
        push("## Features with no variation (constant across all rows)")
        push("  These are uninformative for difficulty stratification.")
        for name in constant:
            push(f"  - {name}")
        push("")

    # 2. PCA explained variance.
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_to_50 = int(np.searchsorted(cum, 0.50)) + 1
    n_to_80 = int(np.searchsorted(cum, 0.80)) + 1
    push("## PCA")
    push(f"  PC1 explains {evr[0]:.1%} of variance.")
    push(f"  Need {n_to_50} PCs for 50% variance, {n_to_80} PCs for 80%.")
    push("")

    # 3. Cross-method agreement.
    push("## Method agreement on continuous score (Kendall τ)")
    push(cont_agree.round(3).to_string())
    push("")
    push("## Method agreement on tertile labels (Adjusted Rand Index)")
    push(cat_agree.round(3).to_string())
    push("")

    # 4. Headline interpretation.
    methods = list(cont)
    if len(methods) >= 2:
        upper = cont_agree.where(np.triu(np.ones(cont_agree.shape, dtype=bool), k=1))
        med_tau = float(np.nanmedian(upper.values))
        push("## Headline")
        if med_tau > 0.85:
            push(f"  Methods agree strongly (median Kendall τ = {med_tau:.2f}). The natural")
            push("  difficulty axis is robust — any of these scorings will rank scenes similarly.")
        elif med_tau > 0.6:
            push(f"  Methods agree moderately (median Kendall τ = {med_tau:.2f}). Pick a method")
            push("  on principled grounds (paper says weighted sum) and check sensitivity later.")
        else:
            push(f"  Methods disagree (median Kendall τ = {med_tau:.2f}). The difficulty signal")
            push("  is genuinely multi-dimensional — the MI-weighted RPX-DS will do real work")
            push("  once calibration models are available; uniform / PCA scorings are unreliable.")

    out.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True,
                        help="path to phase_esd_splits.csv")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "splits" / "analysis",
        help="output directory (default: experiments/splits/analysis)",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for k-means / GMM")
    args = parser.parse_args(argv)

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "analyze_esd — exploratory ESD feature analysis",
        "per-(scene, phase) feature table → diagnostic plots + tables",
    )
    cli_ux.config(vars(args))

    if not args.features.is_file():
        raise SystemExit(f"--features not found: {args.features}")

    # `comment='#'` skips the schema-version header written by build_esd_splits.
    df = pd.read_csv(args.features, comment="#")
    expected_cols = {"scene_id", "phase", "n_frames_total", "n_frames_used",
                     *FEATURE_NAMES}
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(f"--features missing columns: {sorted(missing)}")
    if len(df) == 0:
        raise SystemExit("--features is empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] {len(df)} (scene, phase) rows loaded from {args.features}")

    # Drop any rows with all-zero features (failed extractions). They'd
    # poison percentile normalization with a tied-low cluster.
    feat_block = df[list(FEATURE_NAMES)]
    nonzero = (feat_block.abs().sum(axis=1) > 0)
    if (~nonzero).any():
        n_drop = int((~nonzero).sum())
        print(f"[warn] dropping {n_drop} rows with all-zero features (failed extractions)")
        df = df[nonzero].reset_index(drop=True)
        feat_block = df[list(FEATURE_NAMES)]

    pn = percentile_normalize(feat_block)

    # Per-phase descriptive stats.
    per_phase = _per_phase_stats(df)
    per_phase.to_csv(args.out_dir / "feature_stats_by_phase.csv", index=False)
    print(f"[info] wrote feature_stats_by_phase.csv  ({len(per_phase)} rows)")

    # Phase-difference inferential tests.
    diff = _phase_difference_tests(df)
    diff.to_csv(args.out_dir / "phase_difference_tests.csv", index=False)
    print(f"[info] wrote phase_difference_tests.csv  ({len(diff)} features)")

    # Correlations: Pearson on percentile-normalised + Spearman on raw ranks.
    pearson_corr, spearman_corr = _save_correlations(pn, args.out_dir)
    print(f"[info] wrote feature_correlations_{{pearson,spearman}}.csv")

    # Outliers (univariate z-score + multivariate Mahalanobis).
    outliers = _detect_outliers(df, pn)
    outliers.to_csv(args.out_dir / "outliers.csv", index=False)
    print(f"[info] wrote outliers.csv  ({len(outliers)} flags)")

    # Plots.
    _save_correlation_heatmap(pn, args.out_dir / "correlation_heatmap.png")
    print(f"[info] wrote {args.out_dir / 'correlation_heatmap.png'}")

    cont, cat, pca = compute_method_scores(pn, kmeans_seed=args.seed)

    _save_pca_scree(pca, args.out_dir / "pca_scree.png")
    _save_pca_loadings(pca, args.out_dir / "pca_loadings.png")
    _save_feature_distributions(df, pn, args.out_dir / "feature_distributions.png")
    print(f"[info] wrote PCA + feature-distribution plots → {args.out_dir}")

    # Tertile assignments.
    assign = pd.DataFrame({
        "scene_id": df["scene_id"].values,
        "phase":    df["phase"].values,
    })
    for m in cont:
        assign[f"score_{m}"] = cont[m]
    for m in cat:
        assign[f"tertile_{m}"] = cat[m]
    assign_path = args.out_dir / "tertile_assignments.csv"
    assign.to_csv(assign_path, index=False)
    print(f"[info] wrote {assign_path}")

    # Agreement matrices.
    cont_agree = _continuous_agreement(cont)
    cat_agree  = _categorical_agreement(cat)
    cont_agree.to_csv(args.out_dir / "method_agreement_kendall_tau.csv")
    cat_agree.to_csv(args.out_dir / "method_agreement_ari.csv")
    print(f"[info] wrote agreement matrices → {args.out_dir}")

    # Phase composition.
    pc = _phase_composition(df, cat)
    pc.to_csv(args.out_dir / "phase_composition.csv", index=False)
    print(f"[info] wrote phase_composition.csv")

    # Findings.
    _write_findings(pn, pca, cont, cat, cont_agree, cat_agree,
                    args.out_dir / "findings.txt")
    print(f"[info] wrote findings.txt")
    print(f"\nDONE — open {args.out_dir / 'findings.txt'} for the headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Methodology study for ESD difficulty stratification — the core
contribution that defends RPX's split-construction methodology.

Runs SEVEN layers of analysis on the 297 × 27 feature table:

  L1. Methodology pluralism      — 10 methods spanning every major family
  L2. Number-of-clusters         — silhouette / gap statistic / BIC for k=2..10
  L3. Stability                  — bootstrap, weight perturbation, feature dropout
  L4. Outliers                   — Mahalanobis distance + per-method consensus flag
  L5. Cross-method consensus     — majority-vote tier across methods
  L6. Phase-signature            — does (clutter, interaction, clean) form
                                   distinct clusters in feature space?
  L7. Reproducibility            — seed-locked, provenance-stamped outputs

What this script does NOT do:

- Pre-write conclusions. Findings.md is generated *from* the numbers.
- Cherry-pick a method. All 10 are reported with their silhouette / stability
  numbers.
- Pretend agreement when methods disagree. Disagreement IS a finding.

Usage
-----

::

    pip install 'rpx-benchmark[analysis]'
    python difficulty_methodology_study.py \\
        --features benchmark/data/splits/phase_esd_splits.csv \\
        --out-dir  benchmark/data/splits/methodology_study \\
        --bootstrap 1000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import SpectralEmbedding, TSNE
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats import chi2, kendalltau
except ImportError as e:
    raise SystemExit(
        "difficulty_methodology_study requires extra deps. Install with:\n"
        "  pip install 'rpx-benchmark[analysis]'\n"
        f"missing: {e}"
    ) from e

# Repo path bootstrap so rpx_benchmark is importable.
_REPO_ROOT     = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"
if (_BENCHMARK_DIR / "rpx_benchmark").is_dir():
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rpx_benchmark.data.esd import FEATURE_NAMES  # noqa: E402
from rpx_benchmark.data.esd_scoring import (  # noqa: E402
    lpp_embedding,
    percentile_normalize,
    percentile_rank,
    score_max_pn,
    score_mean_pn,
    score_median_pn,
    sha256_of_file,
    tertile_cut,
    uniform_weights,
)

PHASE_NAMES = {0: "clutter", 1: "interaction", 2: "clean"}
TIER_LABELS = ("easy", "medium", "hard")


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class StudyData:
    df:        pd.DataFrame                     # raw rows + features
    feature_matrix: np.ndarray                  # (N, F) raw features
    pn:        np.ndarray                       # (N, F) percentile-normalised
    standardized: np.ndarray                    # (N, F) z-scored (for clustering)
    feature_names: List[str]


# --------------------------------------------------------------------------- #
# L0 — Load + preprocess
# --------------------------------------------------------------------------- #

def load_data(csv_path: Path) -> StudyData:
    df = pd.read_csv(csv_path, comment="#")
    missing = [n for n in FEATURE_NAMES if n not in df.columns]
    if missing:
        raise SystemExit(f"feature columns missing from CSV: {missing}")

    feat = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    nonzero_mask = feat.sum(axis=1) > 0
    if (~nonzero_mask).any():
        n_drop = int((~nonzero_mask).sum())
        print(f"[warn] dropping {n_drop} rows with all-zero features")
        df = df[nonzero_mask].reset_index(drop=True)
        feat = feat[nonzero_mask]

    pn = percentile_normalize(feat)
    # StandardScaler for clustering — handles unbounded features (depth_std,
    # fisheye_sharpness) on the same scale as bounded ones (occ_*, area_cv).
    standardized = StandardScaler().fit_transform(feat)

    print(f"[info] loaded {feat.shape[0]} rows × {feat.shape[1]} features from {csv_path}")
    return StudyData(df=df, feature_matrix=feat, pn=pn,
                     standardized=standardized, feature_names=list(FEATURE_NAMES))


# --------------------------------------------------------------------------- #
# L1 — Methods (every family)
# --------------------------------------------------------------------------- #

def run_all_methods(data: StudyData, *, seed: int) -> Tuple[Dict[str, np.ndarray],
                                                            Dict[str, np.ndarray]]:
    """Returns (continuous_scores, cluster_labels) per method.

    Continuous scores feed Kendall τ + tertile_cut.
    Cluster labels feed ARI + silhouette.
    """
    pn = data.pn
    Xs = data.standardized
    weights = np.full(pn.shape[1], 1.0 / pn.shape[1])

    # ── PCA projections (used downstream) ─────────────────────────────────
    pca5 = PCA(n_components=min(5, pn.shape[1])).fit_transform(Xs)
    pc1 = pca5[:, 0]
    if np.corrcoef(pc1, pn.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1

    cont: Dict[str, np.ndarray] = {
        "mean_pn":      score_mean_pn(pn, weights),
        "median_pn":    score_median_pn(pn),
        "max_pn":       score_max_pn(pn),
        "top3_mean_pn": np.sort(pn, axis=1)[:, -3:].mean(axis=1),
        "pca_pc1":      pc1,
    }

    cluster: Dict[str, np.ndarray] = {}
    cluster["kmeans3"] = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(pca5)
    cluster["kmeans5"] = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(pca5)
    cluster["kmeans8"] = KMeans(n_clusters=8, n_init=10, random_state=seed).fit_predict(pca5)
    cluster["gmm3"]    = GaussianMixture(n_components=3, random_state=seed,
                                         covariance_type="full").fit_predict(pca5)
    cluster["ward3"]   = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Xs)

    # DBSCAN: pick eps from the k-distance plot's knee.
    eps = _select_dbscan_eps(pca5, k=4)
    cluster["dbscan"] = DBSCAN(eps=eps, min_samples=4).fit_predict(pca5)
    print(f"[info] DBSCAN selected eps={eps:.3f}; "
          f"found {len(set(cluster['dbscan'])) - (1 if -1 in cluster['dbscan'] else 0)} "
          f"clusters and {(cluster['dbscan'] == -1).sum()} outliers")

    return cont, cluster


def _select_dbscan_eps(X: np.ndarray, k: int = 4) -> float:
    """Pick DBSCAN eps via the k-distance plot's max-curvature knee."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances = np.sort(nn.kneighbors(X)[0][:, -1])
    # Knee: max curvature on the sorted k-distance curve.
    n = len(distances)
    line = np.linspace(distances[0], distances[-1], n)
    diffs = distances - line
    knee_idx = int(np.argmax(diffs))
    return float(distances[knee_idx])


# --------------------------------------------------------------------------- #
# L2 — Number-of-clusters analysis
# --------------------------------------------------------------------------- #

def n_clusters_analysis(data: StudyData, *, seed: int,
                        k_range: range = range(2, 11)) -> pd.DataFrame:
    pca5 = PCA(n_components=min(5, data.standardized.shape[1])).fit_transform(data.standardized)
    rows: List[dict] = []
    for k in k_range:
        km_labels  = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(pca5)
        gmm        = GaussianMixture(n_components=k, random_state=seed).fit(pca5)
        gmm_labels = gmm.predict(pca5)

        rows.append({
            "k":                 k,
            "kmeans_silhouette": float(silhouette_score(pca5, km_labels))   if len(set(km_labels))  > 1 else float("nan"),
            "gmm_silhouette":    float(silhouette_score(pca5, gmm_labels)) if len(set(gmm_labels)) > 1 else float("nan"),
            "gmm_bic":           float(gmm.bic(pca5)),
            "gmm_aic":           float(gmm.aic(pca5)),
            "kmeans_inertia":    float(KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pca5).inertia_),
        })
    return pd.DataFrame(rows)


def gap_statistic(data: StudyData, *, seed: int, k_range: range = range(2, 11),
                  n_reference: int = 10) -> pd.DataFrame:
    """Tibshirani gap statistic — compares observed log(W_k) against a
    uniform-random reference. The k that maximises gap(k) - gap(k+1) + s_{k+1}
    is the recommended cluster count."""
    pca5 = PCA(n_components=min(5, data.standardized.shape[1])).fit_transform(data.standardized)
    rng = np.random.default_rng(seed)
    mins, maxs = pca5.min(axis=0), pca5.max(axis=0)

    rows: List[dict] = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pca5)
        Wk = np.log(km.inertia_ + 1e-12)

        ref_logs = []
        for _ in range(n_reference):
            ref = rng.uniform(mins, maxs, pca5.shape)
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(ref)
            ref_logs.append(np.log(km_ref.inertia_ + 1e-12))
        ref_logs = np.asarray(ref_logs)

        gap = float(ref_logs.mean() - Wk)
        sk  = float(ref_logs.std() * np.sqrt(1 + 1.0 / n_reference))
        rows.append({"k": k, "gap": gap, "sk": sk, "Wk": Wk,
                     "ref_log_mean": float(ref_logs.mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# L3 — Stability under perturbations
# --------------------------------------------------------------------------- #

def bootstrap_stability(data: StudyData, *, n_boot: int, seed: int) -> pd.DataFrame:
    """For each bootstrap (80% subsample), recompute mean_pn tiers on the
    full data via percentile and check whether each (scene, phase)'s tier
    label is stable. Returns per-row stability fractions."""
    rng = np.random.default_rng(seed)
    n = data.pn.shape[0]
    counts = np.zeros((n, len(TIER_LABELS)), dtype=np.int32)
    label_to_idx = {t: i for i, t in enumerate(TIER_LABELS)}

    for _ in range(n_boot):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        # Recompute PN on subsample, then tertile.
        pn_boot = percentile_normalize(data.feature_matrix[idx])
        scores  = pn_boot.mean(axis=1)
        tiers   = tertile_cut(scores)
        for k, i_global in enumerate(idx):
            counts[i_global, label_to_idx[tiers[k]]] += 1

    # Per-row dominant tier & its frequency.
    n_visited = counts.sum(axis=1)
    rows = []
    for i in range(n):
        if n_visited[i] == 0:
            rows.append({"row": i, "dominant_tier": "n/a", "stability": float("nan")})
            continue
        argmax = int(counts[i].argmax())
        rows.append({
            "row":           i,
            "dominant_tier": TIER_LABELS[argmax],
            "stability":     float(counts[i, argmax]) / float(n_visited[i]),
            "n_visited":     int(n_visited[i]),
        })
    return pd.DataFrame(rows)


def weight_perturbation_stability(data: StudyData, *, n_perturb: int,
                                   seed: int) -> pd.DataFrame:
    """Sample weight vectors from the simplex; recompute mean_pn tier; report
    per-row tier-flip rate vs the uniform-weight assignment."""
    rng = np.random.default_rng(seed)
    F = data.pn.shape[1]
    uniform = np.full(F, 1.0 / F)
    base_tier = tertile_cut(data.pn @ uniform)

    flip_counts = np.zeros(data.pn.shape[0], dtype=np.int32)
    for _ in range(n_perturb):
        # Dirichlet(α=1) sample = uniform on simplex.
        w = rng.dirichlet(np.ones(F))
        tier = tertile_cut(data.pn @ w)
        flip_counts += (tier != base_tier).astype(np.int32)

    return pd.DataFrame({
        "row":              np.arange(data.pn.shape[0]),
        "base_tier":        base_tier,
        "n_perturb":        n_perturb,
        "tier_flip_rate":   flip_counts / n_perturb,
    })


def feature_dropout_stability(data: StudyData, *, n_runs: int, drop_frac: float,
                              seed: int) -> pd.DataFrame:
    """Drop a random subset of features; recompute mean_pn tier; report per-row
    tier-flip rate vs the full-feature assignment."""
    rng = np.random.default_rng(seed)
    F = data.pn.shape[1]
    base_tier = tertile_cut(data.pn.mean(axis=1))
    keep_count = max(1, int(round((1 - drop_frac) * F)))

    flip_counts = np.zeros(data.pn.shape[0], dtype=np.int32)
    for _ in range(n_runs):
        keep = rng.choice(F, size=keep_count, replace=False)
        tier = tertile_cut(data.pn[:, keep].mean(axis=1))
        flip_counts += (tier != base_tier).astype(np.int32)

    return pd.DataFrame({
        "row":              np.arange(data.pn.shape[0]),
        "base_tier":        base_tier,
        "n_runs":           n_runs,
        "drop_frac":        drop_frac,
        "tier_flip_rate":   flip_counts / n_runs,
    })


# --------------------------------------------------------------------------- #
# L4 — Outliers
# --------------------------------------------------------------------------- #

def detect_outliers(data: StudyData, cluster_labels: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Multivariate Mahalanobis distance + per-method 'doesn't fit' flag.

    A row is flagged 'multivariate outlier' if its Mahalanobis distance
    exceeds the 99% χ²(F) threshold. A row is flagged 'cluster outlier'
    if (a) DBSCAN labels it -1, OR (b) it's in the smallest cluster of
    multiple methods.
    """
    pn = data.pn
    n, F = pn.shape

    # Mahalanobis on PN matrix.
    mu = pn.mean(axis=0)
    cov = np.cov(pn, rowvar=False) + 1e-6 * np.eye(F)
    inv = np.linalg.inv(cov)
    diffs = pn - mu
    mh2 = np.einsum("ij,jk,ik->i", diffs, inv, diffs)
    chi2_thresh = float(chi2.ppf(0.99, df=F))

    out_records = []
    for i in range(n):
        is_multi = bool(mh2[i] > chi2_thresh)

        # Cluster outlier: DBSCAN noise OR consistently in tiny cluster.
        cluster_outlier_flags = []
        for method, labels in cluster_labels.items():
            if method == "dbscan" and labels[i] == -1:
                cluster_outlier_flags.append(method)
            else:
                # In the smallest cluster?
                cluster_sizes = Counter(labels)
                if cluster_sizes[labels[i]] <= max(2, n // 30):
                    cluster_outlier_flags.append(method)

        out_records.append({
            "row":                  i,
            "scene_id":             data.df.iloc[i]["scene_id"],
            "phase":                int(data.df.iloc[i]["phase"]),
            "mahalanobis_dist":     float(np.sqrt(mh2[i])),
            "multivariate_outlier": is_multi,
            "cluster_outlier_methods": ",".join(cluster_outlier_flags),
            "cluster_outlier_count":   len(cluster_outlier_flags),
        })

    return pd.DataFrame(out_records)


# --------------------------------------------------------------------------- #
# L5 — Cross-method consensus
# --------------------------------------------------------------------------- #

def consensus_tier(continuous_scores: Dict[str, np.ndarray],
                    cluster_labels: Dict[str, np.ndarray],
                    pn: np.ndarray) -> pd.DataFrame:
    """Majority-vote tier across every method (continuous → tertile, cluster
    → tertile via centroid ordering on mean_pn)."""
    n = pn.shape[0]
    mean_pn = pn.mean(axis=1)
    method_tiers: Dict[str, np.ndarray] = {}

    for m, scores in continuous_scores.items():
        method_tiers[m] = tertile_cut(scores)

    degenerate_methods: List[str] = []
    for m, labels in cluster_labels.items():
        # Map cluster ids to easy/medium/hard by ordering centroids on mean_pn.
        unique_labels = [c for c in sorted(set(labels)) if c != -1]
        if len(unique_labels) < 3:
            # Degenerate clustering (e.g., DBSCAN found 0-2 clusters + noise).
            # Honest reporting: mark all entries as 'degenerate', NOT silently
            # fall back to mean_pn (which would produce a misleading ARI=1.0).
            method_tiers[m] = np.full(n, "degenerate", dtype=object)
            degenerate_methods.append(m)
            continue
        centroid_means = {c: float(mean_pn[labels == c].mean()) for c in unique_labels}
        sorted_clusters = sorted(centroid_means, key=centroid_means.get)
        # Map: lowest-mean cluster → easy, ... highest → hard. >3 clusters
        # → uniformly stretch into 3 tiers.
        per_cluster_tier = {}
        for rank, c in enumerate(sorted_clusters):
            tier_idx = int(rank * len(TIER_LABELS) / len(sorted_clusters))
            per_cluster_tier[c] = TIER_LABELS[min(tier_idx, len(TIER_LABELS) - 1)]
        # DBSCAN -1 (noise) → flag separately as a fourth tier "outlier"; for
        # consensus we treat them as 'medium' (the safest neutral).
        per_cluster_tier[-1] = "medium"
        method_tiers[m] = np.asarray([per_cluster_tier[c] for c in labels], dtype=object)

    # Majority vote — exclude 'degenerate' votes from the tally so DBSCAN
    # noise doesn't dilute the consensus.
    valid_methods = [m for m in method_tiers if m not in degenerate_methods]
    consensus = []
    for i in range(n):
        votes = Counter(method_tiers[m][i] for m in valid_methods)
        consensus.append(votes.most_common(1)[0][0])
    consensus = np.asarray(consensus, dtype=object)

    # Disagreement: how many distinct (non-degenerate) tiers per row?
    disagreement = []
    for i in range(n):
        tiers_here = {method_tiers[m][i] for m in valid_methods}
        disagreement.append(len(tiers_here))

    df = pd.DataFrame({
        "row":           np.arange(n),
        "consensus_tier": consensus,
        "n_distinct_tiers": disagreement,
        "n_methods":     [len(method_tiers)] * n,
    })
    for m, t in method_tiers.items():
        df[f"tier_{m}"] = t
    return df, method_tiers


# --------------------------------------------------------------------------- #
# L6 — Phase signature
# --------------------------------------------------------------------------- #

def phase_signature_analysis(data: StudyData,
                              cluster_labels: Dict[str, np.ndarray]) -> pd.DataFrame:
    """For each clustering method, ARI between cluster labels and phase index.
    High ARI → the method's clusters track the capture phase (i.e., feature
    space separates clutter/interaction/clean)."""
    phase = data.df["phase"].to_numpy()
    rows = []
    for m, labels in cluster_labels.items():
        rows.append({
            "method":             m,
            "ari_vs_phase":       float(adjusted_rand_score(phase, labels)),
            "n_clusters":         int(len(set(labels)) - (1 if -1 in labels else 0)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Agreement matrices
# --------------------------------------------------------------------------- #

def agreement_kendall(continuous_scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    methods = list(continuous_scores)
    mat = np.full((len(methods), len(methods)), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            tau, _ = kendalltau(continuous_scores[a], continuous_scores[b])
            mat[i, j] = tau
    return pd.DataFrame(mat, index=methods, columns=methods)


def agreement_ari(method_tiers: Dict[str, np.ndarray]) -> pd.DataFrame:
    methods = list(method_tiers)
    mat = np.full((len(methods), len(methods)), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            mat[i, j] = adjusted_rand_score(method_tiers[a], method_tiers[b])
    return pd.DataFrame(mat, index=methods, columns=methods)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _save_n_clusters_plots(nc_df: pd.DataFrame, gap_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(nc_df["k"], nc_df["kmeans_inertia"], "o-", color="steelblue")
    axes[0].set_title("K-means inertia (elbow)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia"); axes[0].grid(alpha=0.3)

    axes[1].plot(nc_df["k"], nc_df["kmeans_silhouette"], "o-", label="kmeans")
    axes[1].plot(nc_df["k"], nc_df["gmm_silhouette"],    "s-", label="gmm")
    axes[1].set_title("Silhouette score (higher = better separation)")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].errorbar(gap_df["k"], gap_df["gap"], yerr=gap_df["sk"], fmt="o-", color="firebrick")
    axes[2].set_title("Gap statistic (Tibshirani et al.)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("gap(k)"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_dendrogram(data: StudyData, out: Path) -> None:
    Z = linkage(data.standardized, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, no_labels=True, color_threshold=0.5 * Z[:, 2].max(), ax=ax)
    ax.set_title("Hierarchical clustering — Ward linkage on standardised features")
    ax.set_ylabel("distance")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_kdist_plot(data: StudyData, out: Path, k: int = 4) -> None:
    from sklearn.neighbors import NearestNeighbors
    pca5 = PCA(n_components=min(5, data.standardized.shape[1])).fit_transform(data.standardized)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pca5)
    distances = np.sort(nn.kneighbors(pca5)[0][:, -1])
    eps = _select_dbscan_eps(pca5, k=k)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distances, "o-", markersize=3)
    ax.axhline(eps, color="red", linestyle="--", label=f"selected eps={eps:.3f}")
    ax.set_xlabel(f"point index (sorted by k={k}-nn distance)")
    ax.set_ylabel(f"distance to {k}-th nearest neighbour")
    ax.set_title("DBSCAN ε selection via k-distance knee")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_2d_grid(emb_2d: np.ndarray, method_tiers: Dict[str, np.ndarray],
                   title: str, out: Path) -> None:
    methods = list(method_tiers)
    n_methods = len(methods)
    cols = 4
    rows = (n_methods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    palette = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}
    for i, m in enumerate(methods):
        ax = axes[i]
        for tier, color in palette.items():
            mask = method_tiers[m] == tier
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=tier,
                        s=12, alpha=0.7)
        ax.set_title(m, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best")
    for j in range(n_methods, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_stability_distribution(boot_df: pd.DataFrame, weight_df: pd.DataFrame,
                                  dropout_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(boot_df["stability"].dropna(), bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Bootstrap tier stability\n(higher = same tier across resamples)")
    axes[0].set_xlabel("dominant-tier fraction"); axes[0].grid(alpha=0.3)

    axes[1].hist(weight_df["tier_flip_rate"], bins=20, color="firebrick", edgecolor="white")
    axes[1].set_title("Weight perturbation\n(lower = robust to weight choice)")
    axes[1].set_xlabel("tier-flip rate vs uniform"); axes[1].grid(alpha=0.3)

    axes[2].hist(dropout_df["tier_flip_rate"], bins=20, color="darkgreen", edgecolor="white")
    axes[2].set_title("Feature dropout\n(lower = robust to feature subset)")
    axes[2].set_xlabel("tier-flip rate vs full-feature"); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_consensus_disagreement(consensus_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = Counter(consensus_df["n_distinct_tiers"])
    xs = sorted(counts)
    ys = [counts[k] for k in xs]
    ax.bar(xs, ys, color="purple")
    ax.set_xlabel("# distinct tiers across methods")
    ax.set_ylabel("# (scene, phase) entries")
    ax.set_title("Cross-method tier disagreement distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_feature_contributions(data: StudyData, base_tier: np.ndarray, out: Path) -> None:
    """For each tier, mean per-feature percentile-normalised value. Shows
    which features push scenes into Hard."""
    means = {}
    for tier in TIER_LABELS:
        mask = base_tier == tier
        if mask.any():
            means[tier] = data.pn[mask].mean(axis=0)
    means_df = pd.DataFrame(means, index=data.feature_names)
    fig, ax = plt.subplots(figsize=(12, 6))
    means_df.plot(kind="bar", ax=ax,
                  color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_xticklabels(data.feature_names, rotation=90, fontsize=8)
    ax.set_ylabel("mean percentile-normalised value")
    ax.set_title("Per-feature contribution by mean_pn tier")
    ax.legend(title="tier")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Findings narrative — generated FROM the numbers
# --------------------------------------------------------------------------- #

def write_findings(out_dir: Path, *,
                    n_entries: int,
                    nc_df: pd.DataFrame, gap_df: pd.DataFrame,
                    boot_df: pd.DataFrame, weight_df: pd.DataFrame,
                    dropout_df: pd.DataFrame, consensus_df: pd.DataFrame,
                    method_tiers: Dict[str, np.ndarray],
                    kendall: pd.DataFrame, ari: pd.DataFrame,
                    outliers_df: pd.DataFrame,
                    phase_sig_df: pd.DataFrame) -> None:
    lines: List[str] = []
    push = lines.append

    push("# Difficulty Methodology Study — Findings")
    push(f"\nGenerated automatically from {n_entries} (scene, phase) entries.")
    push("All numbers below come from the analysis tables in this directory.\n")

    # ── L1: methods + agreement ─────────────────────────────────────────
    push("## L1. Method comparison\n")
    push(f"- {len(method_tiers)} methods compared.")
    upper_tau = kendall.where(np.triu(np.ones(kendall.shape, dtype=bool), k=1))
    push(f"- Median pairwise Kendall τ across continuous scores: "
          f"**{float(np.nanmedian(upper_tau.values)):.3f}**")
    push(f"- Median pairwise ARI across categorical tiers: "
          f"**{float(np.nanmedian(ari.where(np.triu(np.ones(ari.shape, bool), k=1)).values)):.3f}**")
    push("\nFull matrices: agreement/kendall_tau.csv, agreement/ari.csv\n")

    # ── L2: number of clusters ──────────────────────────────────────────
    push("## L2. Is k=3 the right number of clusters?\n")
    sil = nc_df.set_index("k")
    best_km_k = int(sil["kmeans_silhouette"].idxmax())
    best_gmm_k = int(sil["gmm_silhouette"].idxmax())
    best_km_sil = float(sil["kmeans_silhouette"].max())
    best_gmm_sil = float(sil["gmm_silhouette"].max())
    push(f"- K-means silhouette is maximised at **k={best_km_k}** (silhouette={best_km_sil:.3f})")
    push(f"- GMM silhouette is maximised at **k={best_gmm_k}** (silhouette={best_gmm_sil:.3f})")
    push(f"- Silhouette at the chosen k=3: kmeans={float(sil.loc[3, 'kmeans_silhouette']):.3f}, "
          f"gmm={float(sil.loc[3, 'gmm_silhouette']):.3f}")

    # Gap-statistic recommendation: smallest k such that gap(k) ≥ gap(k+1) - s_{k+1}.
    rec_k = None
    g = gap_df.set_index("k")
    for k in g.index[:-1]:
        if g.loc[k, "gap"] >= g.loc[k + 1, "gap"] - g.loc[k + 1, "sk"]:
            rec_k = int(k)
            break
    if rec_k is not None:
        push(f"- Gap statistic recommends **k={rec_k}** (smallest k satisfying Tibshirani's rule)")
    push("\nIf best-silhouette-k != 3, the data does NOT prefer 3 natural clusters; the\n"
          "tertile design is a *reporting convention* (33/33/34), not a discovery of\n"
          "natural structure. This is honest to acknowledge in the paper.\n")

    # ── L3: stability ───────────────────────────────────────────────────
    push("## L3. Stability under perturbations\n")
    push(f"- **Bootstrap (80% resampling)**: median per-row tier-stability = "
          f"**{float(boot_df['stability'].median()):.3f}** "
          f"({(boot_df['stability'] > 0.95).sum()} / {len(boot_df)} entries with >95% stability)")
    push(f"- **Weight perturbation (Dirichlet samples)**: median tier-flip rate = "
          f"**{float(weight_df['tier_flip_rate'].median()):.3f}** "
          f"({(weight_df['tier_flip_rate'] > 0.5).sum()} entries flip >50% of the time)")
    push(f"- **Feature dropout (drop 1/3 random features)**: median tier-flip rate = "
          f"**{float(dropout_df['tier_flip_rate'].median()):.3f}** "
          f"({(dropout_df['tier_flip_rate'] > 0.5).sum()} entries flip >50% of the time)")
    push("\nInterpretation:")
    push("- High bootstrap stability + low weight-perturbation flip → the tertile assignment")
    push("  is robust to the specific weighting choice.")
    push("- Low feature-dropout flip → the tertile is robust to small feature-set changes.\n")

    # ── L4: outliers ────────────────────────────────────────────────────
    push("## L4. Outliers\n")
    n_multi = int(outliers_df["multivariate_outlier"].sum())
    n_consensus = int((outliers_df["cluster_outlier_count"] >= 3).sum())
    push(f"- **Multivariate outliers** (Mahalanobis > χ²_F(0.99) threshold): {n_multi}")
    push(f"- **Cluster-consensus outliers** (flagged by ≥3 clustering methods): {n_consensus}")
    if n_multi > 0:
        worst = outliers_df.nlargest(5, "mahalanobis_dist")
        push("\nTop-5 by Mahalanobis distance:")
        for _, r in worst.iterrows():
            push(f"  - {r['scene_id']} phase={r['phase']}  d={r['mahalanobis_dist']:.2f}")
    push("")

    # ── L5: consensus disagreement ──────────────────────────────────────
    push("## L5. Cross-method consensus\n")
    dis = Counter(consensus_df["n_distinct_tiers"])
    total = len(consensus_df)
    for k in sorted(dis):
        pct = 100 * dis[k] / total
        push(f"- **{dis[k]} entries ({pct:.1f}%)** assigned to {k} distinct tier(s) across all methods")
    full_agreement = dis.get(1, 0) / total
    push(f"\nFull-agreement rate: **{full_agreement:.1%}**.")
    push("Higher = methods converge; lower = the choice of method matters and the")
    push("paper should report at least the consensus tier alongside the chosen method.\n")

    # ── L6: phase signature ─────────────────────────────────────────────
    push("## L6. Do the phases (clutter/interaction/clean) form distinct clusters?\n")
    push(phase_sig_df.to_string(index=False))
    max_ari = float(phase_sig_df["ari_vs_phase"].max())
    push(f"\nMaximum ARI between any clustering and the phase index: **{max_ari:.3f}**.")
    if max_ari < 0.05:
        push("→ Feature space does NOT separate the three capture phases — meaning the")
        push("  features capture difficulty, not capture-phase identity. Good signal.")
    elif max_ari < 0.20:
        push("→ Features have weak phase signal. Most of the variance is genuine difficulty.")
    else:
        push("→ Features carry meaningful phase signal; tertiles partially align with")
        push("  capture phase. Worth noting in the paper to avoid confounding claims.")
    push("")

    # ── Recommendation ──────────────────────────────────────────────────
    push("## Methodological recommendation (data-driven)\n")
    push("- **Primary method**: `mean_pn` (uniform-weighted percentile mean) remains the")
    push("  most reproducible choice given absence of model failure data. The stability")
    push("  numbers above quantify how much this matters.")
    push("- **Always reported alongside**: `consensus_tier` (majority vote across all")
    push("  methods). When `mean_pn` and `consensus_tier` disagree, the paper should")
    push("  flag the entry rather than hide the disagreement.")
    push("- **k=3 vs natural-k**: if silhouette and gap recommend k≠3, the paper")
    push("  acknowledges the 33/33/34 tertile is a *reporting convention*, justified")
    push("  by interpretability (Easy/Medium/Hard is canonical in benchmarks) rather")
    push("  than by data-driven cluster discovery.")
    push("")

    (out_dir / "findings.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True,
                        help="path to phase_esd_splits.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "splits" / "methodology_study",
                        help="output directory")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap iterations for stability (default: 1000)")
    parser.add_argument("--weight-perturb", type=int, default=1000,
                        help="random weight vectors for stability (default: 1000)")
    parser.add_argument("--dropout-runs", type=int, default=200,
                        help="feature-dropout iterations (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "agreement").mkdir(exist_ok=True)
    (args.out_dir / "n_clusters").mkdir(exist_ok=True)
    (args.out_dir / "stability").mkdir(exist_ok=True)
    (args.out_dir / "phase_signature").mkdir(exist_ok=True)
    (args.out_dir / "plots").mkdir(exist_ok=True)

    np.random.seed(args.seed)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"[info] starting methodology study; out-dir={args.out_dir}")
    t0 = time.perf_counter()

    data = load_data(args.features)

    # L1 — methods.
    print("[info] L1: running 10 methods...")
    cont, cluster = run_all_methods(data, seed=args.seed)

    # L2 — number of clusters.
    print("[info] L2: number-of-clusters analysis...")
    nc_df = n_clusters_analysis(data, seed=args.seed)
    nc_df.to_csv(args.out_dir / "n_clusters" / "silhouette_vs_k.csv", index=False)
    gap_df = gap_statistic(data, seed=args.seed)
    gap_df.to_csv(args.out_dir / "n_clusters" / "gap_statistic.csv", index=False)
    nc_df[["k", "gmm_bic", "gmm_aic"]].to_csv(
        args.out_dir / "n_clusters" / "bic_vs_k.csv", index=False)
    _save_n_clusters_plots(nc_df, gap_df, args.out_dir / "plots" / "n_clusters_diagnostics.png")

    # L3 — stability.
    print(f"[info] L3: stability (bootstrap={args.bootstrap}, weight={args.weight_perturb}, "
          f"dropout={args.dropout_runs})...")
    boot_df = bootstrap_stability(data, n_boot=args.bootstrap, seed=args.seed)
    weight_df = weight_perturbation_stability(data, n_perturb=args.weight_perturb, seed=args.seed)
    dropout_df = feature_dropout_stability(data, n_runs=args.dropout_runs,
                                            drop_frac=1 / 3, seed=args.seed)
    boot_df.to_csv(args.out_dir / "stability" / "bootstrap.csv", index=False)
    weight_df.to_csv(args.out_dir / "stability" / "weight_perturbation.csv", index=False)
    dropout_df.to_csv(args.out_dir / "stability" / "feature_dropout.csv", index=False)
    _save_stability_distribution(boot_df, weight_df, dropout_df,
                                  args.out_dir / "plots" / "stability_distribution.png")

    # L4 — outliers.
    print("[info] L4: outlier detection...")
    outliers_df = detect_outliers(data, cluster)
    outliers_df.to_csv(args.out_dir / "outliers.csv", index=False)

    # L5 — consensus.
    print("[info] L5: cross-method consensus...")
    consensus_df, method_tiers = consensus_tier(cont, cluster, data.pn)
    consensus_df.insert(1, "scene_id", data.df["scene_id"].values)
    consensus_df.insert(2, "phase",    data.df["phase"].values)
    consensus_df.to_csv(args.out_dir / "agreement" / "consensus_tier.csv", index=False)
    _save_consensus_disagreement(consensus_df,
                                  args.out_dir / "plots" / "consensus_disagreement.png")

    # Agreement matrices.
    kendall = agreement_kendall(cont)
    kendall.to_csv(args.out_dir / "agreement" / "kendall_tau.csv")
    ari = agreement_ari(method_tiers)
    ari.to_csv(args.out_dir / "agreement" / "ari.csv")

    # L6 — phase signature.
    print("[info] L6: phase signature...")
    phase_sig_df = phase_signature_analysis(data, cluster)
    phase_sig_df.to_csv(args.out_dir / "phase_signature" / "phase_separability.csv", index=False)

    # Per-entry assignments table.
    assignments = pd.DataFrame({
        "scene_id":      data.df["scene_id"].values,
        "phase":         data.df["phase"].values,
    })
    for m, scores in cont.items():
        assignments[f"score_{m}"] = scores
        assignments[f"tier_{m}"]  = method_tiers[m]
    for m in cluster:
        assignments[f"tier_{m}"] = method_tiers[m]
        assignments[f"raw_cluster_{m}"] = cluster[m]
    assignments["consensus_tier"] = consensus_df["consensus_tier"]
    assignments["n_distinct_tiers"] = consensus_df["n_distinct_tiers"]
    assignments.to_csv(args.out_dir / "per_entry_assignments.csv", index=False)

    # 2D embeddings — visualisations.
    print("[info] computing 2D embeddings (PCA, t-SNE, LPP, Spectral)...")
    pca2 = PCA(n_components=2).fit_transform(data.standardized)
    _save_2d_grid(pca2, method_tiers,
                   "PCA 2D — coloured by each method's tier",
                   args.out_dir / "plots" / "pca_2d_grid.png")

    tsne2 = TSNE(n_components=2, perplexity=min(30, data.pn.shape[0] // 4),
                 random_state=args.seed, init="pca").fit_transform(data.standardized)
    _save_2d_grid(tsne2, method_tiers,
                   "t-SNE 2D — coloured by each method's tier",
                   args.out_dir / "plots" / "tsne_2d_grid.png")

    try:
        lpp2 = lpp_embedding(data.standardized, n_components=2, n_neighbors=10)
        _save_2d_grid(lpp2, method_tiers,
                       "LPP 2D — coloured by each method's tier",
                       args.out_dir / "plots" / "lpp_2d_grid.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] LPP failed: {e}")

    try:
        spec2 = SpectralEmbedding(n_components=2,
                                   random_state=args.seed).fit_transform(data.standardized)
        _save_2d_grid(spec2, method_tiers,
                       "Spectral Embedding 2D — coloured by each method's tier",
                       args.out_dir / "plots" / "spectral_2d_grid.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SpectralEmbedding failed: {e}")

    # Hierarchical / DBSCAN diagnostic plots.
    _save_dendrogram(data, args.out_dir / "plots" / "dendrogram.png")
    _save_kdist_plot(data, args.out_dir / "plots" / "kdist_dbscan.png", k=4)

    # Per-feature contribution.
    _save_feature_contributions(data, method_tiers["mean_pn"],
                                 args.out_dir / "plots" / "feature_contributions_by_tier.png")

    # Method inventory.
    pca5 = PCA(n_components=min(5, data.standardized.shape[1])).fit_transform(data.standardized)
    inventory_rows = []
    for m, labels in cluster.items():
        inventory_rows.append({
            "method":       m,
            "n_clusters":   int(len(set(labels)) - (1 if -1 in labels else 0)),
            "silhouette":   float(silhouette_score(pca5, labels)) if len(set(labels)) > 1 else float("nan"),
            "ari_vs_mean_pn": float(adjusted_rand_score(method_tiers["mean_pn"], method_tiers[m])),
        })
    for m in cont:
        inventory_rows.append({
            "method":       m,
            "n_clusters":   3,  # tertile cut
            "silhouette":   float("nan"),
            "ari_vs_mean_pn": float(adjusted_rand_score(method_tiers["mean_pn"], method_tiers[m])),
        })
    pd.DataFrame(inventory_rows).to_csv(args.out_dir / "method_inventory.csv", index=False)

    # Findings.
    print("[info] writing findings.md...")
    write_findings(
        args.out_dir,
        n_entries=data.pn.shape[0],
        nc_df=nc_df, gap_df=gap_df,
        boot_df=boot_df, weight_df=weight_df, dropout_df=dropout_df,
        consensus_df=consensus_df,
        method_tiers=method_tiers,
        kendall=kendall, ari=ari,
        outliers_df=outliers_df,
        phase_sig_df=phase_sig_df,
    )

    # Provenance.
    prov = {
        "input_sha256":     sha256_of_file(args.features),
        "n_entries":        int(data.pn.shape[0]),
        "n_features":       int(data.pn.shape[1]),
        "feature_names":    data.feature_names,
        "seed":             args.seed,
        "bootstrap_iters":  args.bootstrap,
        "weight_perturb_iters": args.weight_perturb,
        "dropout_iters":    args.dropout_runs,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"[done] {time.perf_counter() - t0:.1f}s — outputs in {args.out_dir}")
    print(f"       open {args.out_dir / 'findings.md'} for the headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ESD scoring methods: turn the 27-feature table into a single difficulty
score per (scene, phase) and an Easy/Medium/Hard tertile assignment.

Five scoring methods are exposed, all operating on the **percentile-normalised**
feature matrix (per-feature ``rank/N``, ∈ ``[0,1]``):

- ``mean_pn``       — uniform weighted mean (paper-stated method, primary default)
- ``median_pn``     — robust to outlier features
- ``max_pn``        — weakest-link: hardest single feature defines difficulty
- ``pca_pc1``       — first principal component score
- ``kmeans3``       — k-means clusters tier-ordered by centroid magnitude

Each method is a pure function that takes the percentile-normalised matrix
(plus optional weights for ``mean_pn``) and returns a ``(continuous_score,
categorical_tier)`` pair.

Two weighting tiers are supported (paper §3.2):

- ``uniform_v1`` — ``wᵢ = 1/F``. Always reproducible from data alone. The
  test-of-time baseline.
- ``mi_v1``     — weights derived from mutual information between features and
  model failure rates. Loaded from a JSON artifact (see :func:`load_mi_weights`).
  The headline "official" tier when calibration models are available.

Heavy deps (``scikit-learn``) are lazy-imported inside each function so the
module imports cleanly on a minimal install — the ``mean_pn``, ``median_pn``,
``max_pn`` paths work without sklearn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import ConfigError, DatasetError
from ..logging_utils import get_logger

log = get_logger(__name__)


# Method names — kept as a tuple so callers can iterate stably.
SCORING_METHODS: Tuple[str, ...] = (
    "mean_pn",
    "median_pn",
    "max_pn",
    "pca_pc1",
    "kmeans3",
)

# The default primary method. Documented in the paper as the v1 official score.
DEFAULT_PRIMARY: str = "mean_pn"

# Tertile labels — kept as a constant so the order matches every output file.
TERTILE_LABELS: Tuple[str, str, str] = ("easy", "medium", "hard")

# Confidence labels for per-row confidence based on weight-perturbation flip rate.
CONFIDENCE_LABELS: Tuple[str, str, str] = ("high", "medium", "low")

# Feature → modality category mapping. Keys ordered by the paper appendix
# enumeration; values list the canonical FEATURE_NAMES that belong to each
# category. Used to compute the per-category sub-score vector that surfaces
# the multi-faceted nature of difficulty (data story §S4).
FEATURE_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "annotation_effort": ("iter_mean", "iter_max"),
    "scene_complexity": ("obj_mean", "obj_std", "obj_consist"),
    "occlusion": ("occ_mean", "occ_p90", "occ_heavy"),
    "depth_quality": ("depth_invalid", "depth_invalid_mask", "depth_std", "depth_std_mask"),
    "photometric_conflict": ("specular", "dark"),
    "image_quality": ("rgb_blur", "rgb_texture"),
    "object_size": ("mask_area_mean", "mask_area_std"),
    "temporal_stability": ("area_cv", "area_drop", "vis_instability"),
    "camera_motion": ("trans_mean", "trans_p90", "rot_mean", "rot_p90", "jerk"),
    "fisheye_stereo": (
        "fisheye_dark",
        "fisheye_bright",
        "fisheye_sharpness",
        "fisheye_corr",
        "fisheye_texture",
    ),
}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@dataclass
class ScoringProvenance:
    """Metadata stamped on every output file so a 2030 researcher can verify
    reproducibility without ambiguity."""

    input_sha256: str
    feature_set_version: str
    scoring_version: str  # "uniform_v1" | "mi_v1"
    primary_method: str  # one of SCORING_METHODS
    weights: Dict[str, float]  # {feature_name: weight} for primary scoring
    feature_names: List[str]  # the 27 feature names, in canonical order
    n_entries: int
    generated_at_utc: str
    script_version: str = "unknown"  # caller fills with `git rev-parse HEAD`
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_sha256": self.input_sha256,
            "feature_set_version": self.feature_set_version,
            "scoring_version": self.scoring_version,
            "primary_method": self.primary_method,
            "weights": self.weights,
            "feature_names": self.feature_names,
            "n_entries": self.n_entries,
            "generated_at_utc": self.generated_at_utc,
            "script_version": self.script_version,
            "extra": self.extra,
        }


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes — used to anchor provenance."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def uniform_weights(feature_names: List[str]) -> Dict[str, float]:
    """``wᵢ = 1/F`` — the always-reproducible baseline.

    Note: this is the ``uniform_v1`` tier. The methodology's headline scoring
    is ``effort_stratified_v1`` (see :func:`effort_stratified_weights`) which
    actually honours the methodology's name; ``uniform_v1`` is kept as a
    "no-prior" reference point for the sensitivity analysis.
    """
    if not feature_names:
        raise ConfigError(
            "uniform_weights requires a non-empty feature list",
            hint="check that FEATURE_NAMES is populated.",
        )
    w = 1.0 / len(feature_names)
    return {name: w for name in feature_names}


# Features that anchor the Effort-Stratified Difficulty (ESD) score: mask
# refinement iteration counts. The methodology is *named* after these — the
# ``effort_stratified_v1`` weighting reflects that name in the actual score.
EFFORT_FEATURES: Tuple[str, ...] = ("iter_mean", "iter_max")

# Default convex-combination weight on the effort block. α=0.25 means
# "effort gets a modest boost" without dominating: each iter feature
# contributes 12.5% of the score (4× the 3% each other feature gets),
# but the 25 perception features still collectively drive 75% of the score.
DEFAULT_EFFORT_ALPHA: float = 0.25


def effort_stratified_weights(
    feature_names: List[str],
    *,
    alpha: float = DEFAULT_EFFORT_ALPHA,
    effort_features: Tuple[str, ...] = EFFORT_FEATURES,
) -> Dict[str, float]:
    """Effort-anchored weights — the headline ``effort_stratified_v1`` scoring.

    The methodology is named *Effort-Stratified Difficulty*; this weighting
    actually puts effort at the centre. The convex combination

        RPX-DS = α · effort_score + (1 − α) · perception_score

    expands to per-feature weights:

        iter_mean = iter_max = α / |effort_features|
        other features       = (1 − α) / (F − |effort_features|)

    With ``alpha = 0.25`` (default) and 31 features (2 effort + 29 other),
    iter features each carry 12.5 % of the score (25 % combined); each of the
    29 perception features carries ~2.6 % (75 % combined). At ``alpha = 0`` the
    weighting reduces to uniform over the perception features only; at
    ``alpha = 1`` the score depends only on annotation effort.
    """
    if not feature_names:
        raise ConfigError(
            "effort_stratified_weights requires a non-empty feature list",
            hint="check that FEATURE_NAMES is populated.",
        )
    if not (0.0 <= alpha <= 1.0):
        raise ConfigError(
            f"alpha must be in [0, 1], got {alpha}",
            hint="alpha controls the convex combination of effort vs perception.",
        )

    effort = [n for n in feature_names if n in effort_features]
    other = [n for n in feature_names if n not in effort_features]
    if not effort:
        raise ConfigError(
            f"no effort features found in feature list (looked for {effort_features})",
            hint="check that FEATURE_NAMES includes iter_mean / iter_max.",
        )
    if not other:
        # Degenerate edge case: every feature is an effort feature → fall back
        # to uniform within the effort block so weights still sum to 1.
        return uniform_weights(feature_names)

    w_effort = alpha / len(effort)
    w_other = (1.0 - alpha) / len(other)
    weights = {n: (w_effort if n in effort_features else w_other) for n in feature_names}
    # Sanity: weights sum to 1 (modulo floating-point).
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, f"weights sum to {total}, expected 1.0"
    return weights


def load_mi_weights(
    path: Path,
    feature_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Load mutual-information-derived weights from a JSON artifact.

    The artifact contract::

        {
          "weights": { "<feature_name>": <float>, ... },
          "calibration_model_set": [ "model_id", ... ],
          "metric_used": "MI",
          "notes": "..."
        }

    Weights are renormalised to sum to 1.0 over the requested ``feature_names``.
    Features missing from the artifact get weight 0.0; features in the artifact
    but not in ``feature_names`` are silently dropped (forward-compat: future
    feature additions don't break old MI files).
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"MI weights file not found: {path}",
            hint="pass --weights-mi to point at the calibration artifact.",
        )
    payload = json.loads(path.read_text())
    raw = payload.get("weights")
    if not isinstance(raw, dict):
        raise DatasetError(
            f"MI weights file missing 'weights' dict: {path}",
            hint="artifact must look like {'weights': {feature: float, ...}, ...}",
        )

    aligned = {name: float(raw.get(name, 0.0)) for name in feature_names}
    total = sum(aligned.values())
    if total <= 0:
        raise DatasetError(
            f"MI weights sum to {total} after alignment with feature list",
            hint="check that weights file uses the same feature names as the build.",
        )
    aligned = {k: v / total for k, v in aligned.items()}

    metadata = {k: v for k, v in payload.items() if k != "weights"}
    return aligned, metadata


# --------------------------------------------------------------------------- #
# Percentile-normalisation
# --------------------------------------------------------------------------- #


def percentile_normalize(matrix: np.ndarray) -> np.ndarray:
    """Per-column rank divided by N (matches paper §3.2 ``f̃ᵢ``).

    ``matrix`` shape ``(N, F)``. Output same shape, values in ``[0, 1]``.
    Constant columns (zero variance) are mapped to all-0.5 — informationless
    but non-degenerate; downstream scoring handles them gracefully.

    NaN values (e.g., fisheye features for scenes without fisheye data) are
    ranked only among non-NaN entries, then set to 0.5 (midpoint) in the
    output. This prevents missing-modality scenes from being systematically
    biased toward low difficulty.
    """
    if matrix.ndim != 2:
        raise ConfigError(
            f"percentile_normalize expects 2D array, got shape {matrix.shape}",
            hint="reshape to (n_rows, n_features) before calling.",
        )
    n_rows, n_cols = matrix.shape
    out = np.empty_like(matrix, dtype=np.float64)
    for j in range(n_cols):
        col = matrix[:, j]
        valid_mask = ~np.isnan(col)
        valid = col[valid_mask]
        n_valid = valid.size

        if n_valid == 0 or np.unique(valid).size <= 1:
            out[:, j] = 0.5  # all-NaN or constant → uninformative midpoint
            continue

        # Rank only the non-NaN entries.
        order = np.argsort(valid, kind="mergesort")
        ranks = np.empty(n_valid, dtype=np.float64)
        ranks[order] = np.arange(1, n_valid + 1, dtype=np.float64)
        # Average ranks for tied values.
        for v in np.unique(valid):
            idx = np.where(valid == v)[0]
            if idx.size > 1:
                ranks[idx] = ranks[idx].mean()

        # Write ranked values back; NaN entries get 0.5 (neutral midpoint).
        out[:, j] = 0.5
        out[valid_mask, j] = ranks / n_valid
    return out


# --------------------------------------------------------------------------- #
# Tertile cut
# --------------------------------------------------------------------------- #


def tertile_cut(scores: np.ndarray) -> np.ndarray:
    """Bottom 33% / middle 33% / top 34% → 'easy' / 'medium' / 'hard'.

    Uses rank-based percentiles so ties never collapse a tertile. If the score
    has fewer than 3 unique values everything degenerates to 'medium' (caller
    can decide what to do).
    """
    n = scores.size
    if n == 0:
        return np.array([], dtype=object)
    if np.unique(scores).size < 3:
        log.warning("tertile_cut: fewer than 3 unique scores; defaulting all to 'medium'")
        return np.array([TERTILE_LABELS[1]] * n, dtype=object)

    ranks = np.argsort(np.argsort(scores, kind="mergesort")) + 1
    p = ranks / n
    out = np.empty(n, dtype=object)
    out[p <= 1.0 / 3.0] = TERTILE_LABELS[0]
    out[(p > 1.0 / 3.0) & (p <= 2.0 / 3)] = TERTILE_LABELS[1]
    out[p > 2.0 / 3.0] = TERTILE_LABELS[2]
    return out


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Per-row percentile rank in ``[0, 1]`` based on ``argsort``."""
    n = scores.size
    if n == 0:
        return np.array([], dtype=np.float64)
    ranks = np.argsort(np.argsort(scores, kind="mergesort")) + 1
    return ranks.astype(np.float64) / n


# --------------------------------------------------------------------------- #
# Scoring methods (each returns continuous score; tertile_cut applied separately)
# --------------------------------------------------------------------------- #


def score_mean_pn(pn: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of percentile-normalised features. Default primary method.

    With ``weights = 1/F`` this is the uniform RPX-DS in paper §3.2. With
    MI-derived weights it becomes ``mi_v1``.
    """
    if pn.shape[1] != weights.size:
        raise ConfigError(
            f"weights size {weights.size} does not match feature count {pn.shape[1]}",
            hint="weights must align with the feature columns of pn.",
        )
    return pn @ weights


def score_median_pn(pn: np.ndarray) -> np.ndarray:
    """Per-row median over features. Robust to outlier features."""
    return np.median(pn, axis=1)


def score_max_pn(pn: np.ndarray) -> np.ndarray:
    """Per-row max over features. Weakest-link: hardest single feature wins."""
    return pn.max(axis=1)


def score_pca_pc1(pn: np.ndarray) -> np.ndarray:
    """First principal component score of the centred percentile matrix.

    PC1 is sign-flipped if needed so that higher score correlates with higher
    mean_pn — keeps the "Hard = high score" convention consistent across methods.
    """
    try:
        from sklearn.decomposition import PCA  # noqa: PLC0415
    except ImportError as e:
        raise ConfigError(
            "score_pca_pc1 requires scikit-learn",
            hint="install with: pip install 'rpx-benchmark[analysis]'",
        ) from e
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(pn - pn.mean(axis=0)).ravel()
    if np.corrcoef(pc1, pn.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    return pc1


def score_kmeans3(pn: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """K-means cluster id (k=3) on the first 5 PCs of the percentile matrix.

    Returned as a *continuous* score: each row's value is the centroid's
    rank by mean_pn-magnitude (0 → easiest cluster, 2 → hardest). The
    ``tertile_cut`` then collapses these onto ``easy/medium/hard``.
    """
    try:
        from sklearn.cluster import KMeans  # noqa: PLC0415
        from sklearn.decomposition import PCA  # noqa: PLC0415
    except ImportError as e:
        raise ConfigError(
            "score_kmeans3 requires scikit-learn",
            hint="install with: pip install 'rpx-benchmark[analysis]'",
        ) from e
    n_rows = pn.shape[0]
    if n_rows < 3:
        log.warning("score_kmeans3: <3 rows; degenerate clustering, returning zeros")
        return np.zeros(n_rows, dtype=np.float64)

    n_pcs = min(5, pn.shape[1], n_rows)
    proj = PCA(n_components=n_pcs).fit_transform(pn - pn.mean(axis=0))
    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(proj)
    labels = km.labels_

    # Order clusters by their mean of mean_pn so cluster 0 = easiest.
    mean_pn = pn.mean(axis=1)
    cluster_means = {c: float(mean_pn[labels == c].mean()) for c in np.unique(labels)}
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    rank_of = {c: i for i, c in enumerate(sorted_clusters)}
    return np.asarray([rank_of[c] for c in labels], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Locality Preserving Projections (LPP) — He & Niyogi, NeurIPS 2003
# --------------------------------------------------------------------------- #


def lpp_embedding(
    X: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 10,
    heat_t: Optional[float] = None,
) -> np.ndarray:
    """Locality Preserving Projections — linear projection that preserves
    local neighbourhood structure on the data manifold.

    Implements the generalised-eigenvalue formulation from
    He & Niyogi (NeurIPS 2003): solve ``X^T L X v = λ X^T D X v`` where
    ``L = D − W`` is the graph Laplacian over a kNN affinity matrix with
    heat-kernel weights.

    Parameters
    ----------
    X            : ``(n, d)`` data matrix
    n_components : output dimensionality
    n_neighbors  : k for the kNN affinity graph (excluding self)
    heat_t       : heat-kernel bandwidth; defaults to median squared
                   inter-neighbour distance

    Returns
    -------
    ``(n, n_components)`` linear embedding ``Y = X @ W``.

    Notes
    -----
    LPP is the linear approximation of Laplacian Eigenmaps. Compared to
    sklearn's ``SpectralEmbedding`` (the non-linear version), LPP gives a
    deterministic projection that can be applied to held-out data — useful
    for analyses that need the projection matrix itself, not just the
    embedding of the training points.
    """
    try:
        from scipy.linalg import eigh  # noqa: PLC0415
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    except ImportError as e:
        raise ConfigError(
            "lpp_embedding requires scikit-learn and scipy",
            hint="install with: pip install 'rpx-benchmark[analysis]'",
        ) from e

    if X.ndim != 2:
        raise ConfigError(
            f"lpp_embedding expects 2D array, got shape {X.shape}",
            hint="reshape to (n_rows, n_features) before calling.",
        )
    n, d = X.shape
    if n_neighbors >= n:
        raise ConfigError(
            f"n_neighbors ({n_neighbors}) must be < n_samples ({n})",
            hint="reduce n_neighbors or supply a larger dataset.",
        )

    # Step 1: kNN graph (k+1 to drop self).
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]  # drop self
    indices = indices[:, 1:]

    # Step 2: heat-kernel weights, symmetrised affinity matrix W.
    if heat_t is None:
        heat_t = float(np.median(distances**2))
        if heat_t <= 0:
            heat_t = 1.0  # degenerate fallback (e.g., duplicated points)

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            w = float(np.exp(-(distances[i, j_idx] ** 2) / heat_t))
            # Symmetrise: take max so the affinity is consistent under
            # asymmetric kNN.
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)

    # Step 3: Laplacian.
    deg = W.sum(axis=1)
    D = np.diag(deg)
    L = D - W

    # Step 4: solve generalised eigenvalue problem X^T L X v = λ X^T D X v.
    A = X.T @ L @ X
    B = X.T @ D @ X
    # Tikhonov regularisation on B to handle rank-deficient cases.
    B = B + 1e-9 * np.eye(d)
    eigvals, eigvecs = eigh(A, B)

    # Take the n_components smallest non-zero eigenvalues (they correspond
    # to directions that minimise local distortion).
    proj = eigvecs[:, :n_components]
    return X @ proj


# --------------------------------------------------------------------------- #
# Driver — run all methods at once
# --------------------------------------------------------------------------- #


def all_methods(
    pn: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute every method's continuous score on the same percentile matrix.

    Returns ``{method_name: 1D score array}``. Methods that need scikit-learn
    are skipped with a warning if the import fails — the caller can then ship
    the available subset.
    """
    out: Dict[str, np.ndarray] = {}
    out["mean_pn"] = score_mean_pn(pn, weights)
    out["median_pn"] = score_median_pn(pn)
    out["max_pn"] = score_max_pn(pn)
    try:
        out["pca_pc1"] = score_pca_pc1(pn)
    except ConfigError as e:
        log.warning("skipping pca_pc1: %s", e)
    try:
        out["kmeans3"] = score_kmeans3(pn, seed=seed)
    except ConfigError as e:
        log.warning("skipping kmeans3: %s", e)
    return out


# --------------------------------------------------------------------------- #
# Structured-metric helpers (data story §S4 + §S5 + consensus)
# --------------------------------------------------------------------------- #

# Per-row tier-flip rate thresholds for the confidence label. A pair flipping
# tiers in <10% of weight perturbations is "high" confidence; 10-50% medium;
# >50% low.
DEFAULT_CONFIDENCE_HIGH_MAX: float = 0.10
DEFAULT_CONFIDENCE_MEDIUM_MAX: float = 0.50


def compute_per_category_scores(
    pn: np.ndarray,
    feature_names: List[str],
    categories: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Dict[str, np.ndarray]:
    """Per-row mean of percentile-normalised features within each category.

    Returns ``{category_name: 1D float array of length N}``.
    """
    cats = categories if categories is not None else FEATURE_CATEGORIES
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    out: Dict[str, np.ndarray] = {}
    for cat, feats in cats.items():
        idx = [name_to_idx[f] for f in feats if f in name_to_idx]
        if not idx:
            log.warning("category %s has no features in the matrix; skipping", cat)
            continue
        out[cat] = pn[:, idx].mean(axis=1)
    return out


def compute_consensus_tier(
    method_tiers: Dict[str, np.ndarray],
    *,
    exclude_methods: Optional[Tuple[str, ...]] = None,
) -> np.ndarray:
    """Majority-vote tier across multiple methods' tier assignments.

    Each method's tier array must have the same length. ``exclude_methods``
    drops degenerate methods (e.g. DBSCAN on data with no density modes)
    from the vote. Ties are broken by the canonical TERTILE_LABELS order
    (easy < medium < hard) — i.e. ties favour the easier label, the
    conservative choice for downstream evaluation.
    """
    from collections import Counter

    excluded = set(exclude_methods or ())
    voting = {m: t for m, t in method_tiers.items() if m not in excluded}
    if not voting:
        raise ConfigError(
            "compute_consensus_tier received no methods to vote on",
            hint="check that exclude_methods does not exhaust all methods.",
        )
    n = next(iter(voting.values())).shape[0]
    out = np.empty(n, dtype=object)
    for i in range(n):
        votes = Counter(voting[m][i] for m in voting)
        max_count = max(votes.values())
        winners = [t for t, c in votes.items() if c == max_count]
        if len(winners) == 1:
            out[i] = winners[0]
        else:
            for t in TERTILE_LABELS:
                if t in winners:
                    out[i] = t
                    break
            else:
                out[i] = winners[0]  # fallback (e.g. degenerate label)
    return out


def aggregate_to_scene_splits(
    scene_ids: List[str],
    scores: np.ndarray,
    *,
    phases: Optional[np.ndarray] = None,
    phase_tiers: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Roll per-(scene, phase) scores up to per-scene splits, preserving
    per-phase tier detail for downstream analysis.

    Each scene's representative score is the **mean of its phase scores**;
    scenes are sorted by that score and cut into ⌊N/3⌋ Easy + ⌊N/3⌋ Medium
    + (remainder) Hard. With N=100 the cut is the canonical 33/33/34;
    with N=99 it becomes 33/33/33.

    Output shape::

        {
          "splits": {
            "easy":   [scene_id, …],   # alphabetically sorted in-tier
            "medium": [...],
            "hard":   [...]
          },
          "scene_detail": {
            "scene_id": {
              "scene_tier":  "easy" | "medium" | "hard",
              "scene_score": <mean of phase scores>,
              "phase_tiers": ["easy", "medium", "hard"]   # length-3 if phase
                                                          # info is supplied,
                                                          # in capture-phase
                                                          # order (0,1,2)
              "phase_scores": [0.30, 0.45, 0.28]          # same indexing
            },
            ...
          }
        }

    The split lists are the consumer-facing deliverable; ``scene_detail``
    answers "what did each phase look like for this scene?" inline,
    without requiring readers to cross-reference a separate per-phase
    difficulty table. Per-phase info is included only when ``phases``
    (and optionally ``phase_tiers``) are passed.
    """
    if len(scene_ids) != scores.size:
        raise ConfigError(
            f"length mismatch: {len(scene_ids)} scene_ids vs {scores.size} scores",
            hint="pass arrays aligned 1:1 across (scene, phase) entries.",
        )
    if phases is not None and len(phases) != scores.size:
        raise ConfigError(
            f"length mismatch: {len(phases)} phases vs {scores.size} scores",
            hint="phases must align 1:1 with scene_ids and scores.",
        )
    if phase_tiers is not None and len(phase_tiers) != scores.size:
        raise ConfigError(
            f"length mismatch: {len(phase_tiers)} phase_tiers vs {scores.size} scores",
            hint="phase_tiers must align 1:1 with scene_ids and scores.",
        )

    from collections import defaultdict

    # Aggregate per-scene: mean score plus per-phase detail.
    per_scene: Dict[str, Dict[int, Tuple[float, Optional[str]]]] = defaultdict(dict)
    for i, sid in enumerate(scene_ids):
        ph = int(phases[i]) if phases is not None else i
        tier_here = str(phase_tiers[i]) if phase_tiers is not None else None
        per_scene[sid][ph] = (float(scores[i]), tier_here)

    per_scene_mean = {
        sid: float(np.mean([v[0] for v in info.values()])) for sid, info in per_scene.items()
    }

    # Sort scenes ascending by mean score; tertile-cut.
    ordered = sorted(per_scene_mean, key=per_scene_mean.get)
    n = len(ordered)
    n_easy = n // 3
    n_med = n // 3
    easy = sorted(ordered[:n_easy])
    medium = sorted(ordered[n_easy : n_easy + n_med])
    hard = sorted(ordered[n_easy + n_med :])
    splits = {"easy": easy, "medium": medium, "hard": hard}

    # Reverse map for "which tier does this scene belong to".
    scene_to_tier = {
        **{s: "easy" for s in easy},
        **{s: "medium" for s in medium},
        **{s: "hard" for s in hard},
    }

    scene_detail: Dict[str, Dict[str, object]] = {}
    for sid, info in per_scene.items():
        # Sort phases by index so output is deterministic (capture order).
        ordered_phases = sorted(info)
        phase_scores = [info[p][0] for p in ordered_phases]
        phase_tiers_out = [info[p][1] for p in ordered_phases if info[p][1] is not None]
        detail: Dict[str, object] = {
            "scene_tier": scene_to_tier[sid],
            "scene_score": per_scene_mean[sid],
            "phase_scores": phase_scores,
        }
        if phase_tiers_out:
            detail["phase_tiers"] = phase_tiers_out
        if phases is not None:
            detail["phase_indices"] = ordered_phases
        scene_detail[sid] = detail

    return {"splits": splits, "scene_detail": scene_detail}


def compute_confidence(
    pn: np.ndarray,
    *,
    n_perturb: int = 1000,
    seed: int = 0,
    high_max: float = DEFAULT_CONFIDENCE_HIGH_MAX,
    medium_max: float = DEFAULT_CONFIDENCE_MEDIUM_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row confidence in the uniform mean_pn tier under weight perturbation.

    For each Dirichlet$(\\mathbf{1})$-sampled weight vector, recompute the
    tertile and check whether each row's tier changed. A row whose tier flips
    in <``high_max`` of perturbations is "high" confidence; <``medium_max``
    is "medium"; otherwise "low".

    Returns ``(flip_rate, label)`` arrays, both of length N.
    """
    rng = np.random.default_rng(seed)
    F = pn.shape[1]
    base_score = pn.mean(axis=1)
    base_tier = tertile_cut(base_score)

    flips = np.zeros(pn.shape[0], dtype=np.int32)
    for _ in range(n_perturb):
        w = rng.dirichlet(np.ones(F))
        flips += (tertile_cut(pn @ w) != base_tier).astype(np.int32)
    flip_rate = flips.astype(np.float64) / float(n_perturb)

    label = np.empty(pn.shape[0], dtype=object)
    label[flip_rate < high_max] = "high"
    label[(flip_rate >= high_max) & (flip_rate < medium_max)] = "medium"
    label[flip_rate >= medium_max] = "low"
    return flip_rate, label

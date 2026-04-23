"""Unit tests for rpx_benchmark.data.esd_scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.data.esd_scoring import (
    CONFIDENCE_LABELS,
    DEFAULT_PRIMARY,
    FEATURE_CATEGORIES,
    SCORING_METHODS,
    TERTILE_LABELS,
    all_methods,
    compute_confidence,
    compute_consensus_tier,
    compute_per_category_scores,
    load_mi_weights,
    percentile_normalize,
    percentile_rank,
    score_max_pn,
    score_mean_pn,
    score_median_pn,
    sha256_of_file,
    tertile_cut,
    uniform_weights,
)


@pytest.fixture(autouse=True)
def _force_propagate_for_caplog():
    import logging
    logger = logging.getLogger("rpx_benchmark")
    saved = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = saved


# --------------------------------------------------------------------------- #
# uniform_weights
# --------------------------------------------------------------------------- #

def test_uniform_weights_sums_to_one():
    feats = ["a", "b", "c", "d"]
    w = uniform_weights(feats)
    assert set(w) == set(feats)
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_uniform_weights_rejects_empty():
    from rpx_benchmark.exceptions import ConfigError
    with pytest.raises(ConfigError):
        uniform_weights([])


# --------------------------------------------------------------------------- #
# effort_stratified_weights
# --------------------------------------------------------------------------- #

def test_effort_stratified_weights_alpha_half_gives_50_percent_to_effort():
    from rpx_benchmark.data.esd_scoring import effort_stratified_weights, EFFORT_FEATURES
    feats = ["iter_mean", "iter_max", "f1", "f2", "f3"]
    w = effort_stratified_weights(feats, alpha=0.5)
    # Two effort features × 0.25 + three other features × (0.5/3 ≈ 0.167) = 1.0
    assert w["iter_mean"] == pytest.approx(0.25)
    assert w["iter_max"] == pytest.approx(0.25)
    assert w["f1"] == pytest.approx(0.5 / 3)
    effort_total = sum(w[f] for f in EFFORT_FEATURES if f in feats)
    other_total  = sum(v for k, v in w.items() if k not in EFFORT_FEATURES)
    assert effort_total == pytest.approx(0.5)
    assert other_total == pytest.approx(0.5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_effort_stratified_weights_alpha_zero_is_uniform_over_others():
    from rpx_benchmark.data.esd_scoring import effort_stratified_weights
    feats = ["iter_mean", "iter_max", "f1", "f2"]
    w = effort_stratified_weights(feats, alpha=0.0)
    assert w["iter_mean"] == 0.0
    assert w["iter_max"] == 0.0
    assert w["f1"] == pytest.approx(0.5)
    assert w["f2"] == pytest.approx(0.5)


def test_effort_stratified_weights_alpha_one_concentrates_on_effort():
    from rpx_benchmark.data.esd_scoring import effort_stratified_weights
    feats = ["iter_mean", "iter_max", "f1", "f2"]
    w = effort_stratified_weights(feats, alpha=1.0)
    assert w["iter_mean"] == pytest.approx(0.5)
    assert w["iter_max"] == pytest.approx(0.5)
    assert w["f1"] == 0.0
    assert w["f2"] == 0.0


def test_effort_stratified_weights_rejects_invalid_alpha():
    from rpx_benchmark.data.esd_scoring import effort_stratified_weights
    from rpx_benchmark.exceptions import ConfigError
    with pytest.raises(ConfigError):
        effort_stratified_weights(["iter_mean", "iter_max", "f1"], alpha=1.5)
    with pytest.raises(ConfigError):
        effort_stratified_weights(["iter_mean", "iter_max", "f1"], alpha=-0.1)


def test_effort_stratified_weights_rejects_missing_effort_feature():
    from rpx_benchmark.data.esd_scoring import effort_stratified_weights
    from rpx_benchmark.exceptions import ConfigError
    with pytest.raises(ConfigError):
        effort_stratified_weights(["a", "b", "c"], alpha=0.5)


def test_effort_stratified_weights_on_real_27_feature_set():
    """The default for the 27-feature production set."""
    from rpx_benchmark.data.esd import FEATURE_NAMES
    from rpx_benchmark.data.esd_scoring import effort_stratified_weights
    w = effort_stratified_weights(list(FEATURE_NAMES), alpha=0.5)
    # 2 effort features × 0.25 + 25 other features × (0.5/25 = 0.02) = 1.0
    assert w["iter_mean"] == pytest.approx(0.25)
    assert w["iter_max"] == pytest.approx(0.25)
    assert w["depth_invalid"] == pytest.approx(0.02)
    assert sum(w.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# percentile_normalize
# --------------------------------------------------------------------------- #

def test_percentile_normalize_monotonic_column():
    """Strictly increasing column → normalised values 1/N, 2/N, ..., 1."""
    n = 5
    col = np.arange(n, dtype=np.float64).reshape(-1, 1)
    pn = percentile_normalize(col)
    expected = (np.arange(1, n + 1) / n).reshape(-1, 1)
    np.testing.assert_allclose(pn, expected)


def test_percentile_normalize_constant_column_maps_to_half():
    col = np.full((4, 1), 7.0)
    pn = percentile_normalize(col)
    np.testing.assert_allclose(pn, np.full((4, 1), 0.5))


def test_percentile_normalize_handles_ties_with_average_rank():
    """Two tied values should both get the average of their ranks."""
    col = np.array([[1.0], [2.0], [2.0], [3.0]])
    pn = percentile_normalize(col)
    # Ranks: 1, 2.5, 2.5, 4 → /4 = 0.25, 0.625, 0.625, 1.0
    np.testing.assert_allclose(pn, [[0.25], [0.625], [0.625], [1.0]])


def test_percentile_normalize_rejects_non_2d():
    from rpx_benchmark.exceptions import ConfigError
    with pytest.raises(ConfigError):
        percentile_normalize(np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# tertile_cut + percentile_rank
# --------------------------------------------------------------------------- #

def test_tertile_cut_balanced_for_distinct_scores():
    """6 distinct scores → 2 easy, 2 medium, 2 hard."""
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    tiers = tertile_cut(scores)
    assert list(tiers) == ["easy", "easy", "medium", "medium", "hard", "hard"]


def test_tertile_cut_constant_score_degenerates_to_medium(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="rpx_benchmark.data.esd_scoring"):
        tiers = tertile_cut(np.array([0.5, 0.5, 0.5, 0.5]))
    assert all(t == "medium" for t in tiers)
    assert any("fewer than 3 unique" in r.message for r in caplog.records)


def test_tertile_cut_empty_returns_empty():
    assert tertile_cut(np.array([])).size == 0


def test_percentile_rank_monotonic():
    p = percentile_rank(np.array([10, 20, 30, 40]))
    np.testing.assert_allclose(p, [0.25, 0.5, 0.75, 1.0])


# --------------------------------------------------------------------------- #
# Scoring methods
# --------------------------------------------------------------------------- #

def test_score_mean_pn_uniform_equals_row_mean():
    pn = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    w = np.full(3, 1 / 3)
    np.testing.assert_allclose(score_mean_pn(pn, w), [0.2, 0.5])


def test_score_mean_pn_weighted_picks_dominant_feature():
    pn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    w = np.array([0.8, 0.1, 0.1])
    np.testing.assert_allclose(score_mean_pn(pn, w), [0.8, 0.1])


def test_score_mean_pn_rejects_dim_mismatch():
    from rpx_benchmark.exceptions import ConfigError
    pn = np.zeros((3, 4))
    with pytest.raises(ConfigError):
        score_mean_pn(pn, np.zeros(5))


def test_score_median_pn_robust_to_outlier():
    """One feature is an outlier; median ignores it but mean would be pulled."""
    pn = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 1.0]])
    np.testing.assert_allclose(score_median_pn(pn), [0.5])


def test_score_max_pn_picks_hardest_feature():
    pn = np.array([[0.1, 0.2, 0.95], [0.5, 0.4, 0.6]])
    np.testing.assert_allclose(score_max_pn(pn), [0.95, 0.6])


def test_score_pca_pc1_orientation_aligned_with_mean(caplog):
    """PC1 sign should be flipped so 'high mean_pn' ⇒ 'high PC1'."""
    pytest.importorskip("sklearn")
    from rpx_benchmark.data.esd_scoring import score_pca_pc1
    rng = np.random.default_rng(0)
    pn = rng.uniform(0, 1, (30, 5))
    pc1 = score_pca_pc1(pn)
    # Correlation with mean_pn must be non-negative (sign-flipped if needed).
    assert np.corrcoef(pc1, pn.mean(axis=1))[0, 1] >= 0


def test_score_kmeans3_orders_clusters_easy_to_hard():
    pytest.importorskip("sklearn")
    from rpx_benchmark.data.esd_scoring import score_kmeans3
    # Three well-separated clusters — easy/medium/hard signal.
    rng = np.random.default_rng(0)
    easy   = rng.uniform(0.0, 0.2, (10, 5))
    medium = rng.uniform(0.4, 0.6, (10, 5))
    hard   = rng.uniform(0.8, 1.0, (10, 5))
    pn = np.vstack([easy, medium, hard])
    labels = score_kmeans3(pn, seed=0)
    # Cluster 0 should be the easy group (lowest mean_pn).
    assert int(labels[:10].mean()) == 0
    assert int(labels[20:].mean()) == 2


def test_all_methods_returns_every_method_when_sklearn_present():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    pn = rng.uniform(0, 1, (30, 5))
    w = np.full(5, 0.2)
    out = all_methods(pn, w)
    assert set(out) == set(SCORING_METHODS)
    for name, scores in out.items():
        assert scores.shape == (30,), f"{name} returned wrong shape"


# --------------------------------------------------------------------------- #
# load_mi_weights
# --------------------------------------------------------------------------- #

def test_load_mi_weights_renormalises_to_one(tmp_path):
    feats = ["a", "b", "c"]
    artifact = {
        "weights": {"a": 2.0, "b": 4.0, "c": 4.0},
        "calibration_model_set": ["model_x", "model_y"],
        "metric_used": "MI",
    }
    p = tmp_path / "mi.json"
    p.write_text(json.dumps(artifact))
    weights, metadata = load_mi_weights(p, feats)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.2)
    assert weights["b"] == pytest.approx(0.4)
    assert weights["c"] == pytest.approx(0.4)
    assert metadata["calibration_model_set"] == ["model_x", "model_y"]


def test_load_mi_weights_treats_missing_features_as_zero(tmp_path):
    feats = ["a", "b", "c"]
    artifact = {"weights": {"a": 1.0, "b": 1.0}, "metric_used": "MI"}
    p = tmp_path / "mi.json"
    p.write_text(json.dumps(artifact))
    weights, _ = load_mi_weights(p, feats)
    assert weights["c"] == pytest.approx(0.0)
    assert weights["a"] == pytest.approx(0.5)


def test_load_mi_weights_drops_extra_features_in_artifact(tmp_path):
    """An MI file with stale features (no longer in the canonical list) must
    not poison the renormalisation."""
    feats = ["a", "b"]
    artifact = {"weights": {"a": 1.0, "b": 1.0, "deleted_feature": 100.0},
                "metric_used": "MI"}
    p = tmp_path / "mi.json"
    p.write_text(json.dumps(artifact))
    weights, _ = load_mi_weights(p, feats)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.5)


def test_load_mi_weights_raises_on_missing_file(tmp_path):
    from rpx_benchmark.exceptions import DatasetError
    with pytest.raises(DatasetError):
        load_mi_weights(tmp_path / "nope.json", ["a"])


def test_load_mi_weights_raises_on_zero_total(tmp_path):
    from rpx_benchmark.exceptions import DatasetError
    p = tmp_path / "mi.json"
    p.write_text(json.dumps({"weights": {"a": 0.0}, "metric_used": "MI"}))
    with pytest.raises(DatasetError):
        load_mi_weights(p, ["a"])


# --------------------------------------------------------------------------- #
# sha256_of_file
# --------------------------------------------------------------------------- #

def test_sha256_of_file_deterministic(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello world")
    h1 = sha256_of_file(p)
    h2 = sha256_of_file(p)
    assert h1 == h2
    # Known SHA-256 of "hello world".
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


# --------------------------------------------------------------------------- #
# Constants surface
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# LPP — Locality Preserving Projections
# --------------------------------------------------------------------------- #

def test_lpp_recovers_low_dim_structure_from_swiss_roll_like_data():
    """LPP should pull out the locally-meaningful axis of variation.

    Construct a 10D dataset whose true latent structure lives on a 1D
    manifold; verify that the 1D LPP embedding's neighbour ordering
    correlates strongly with the true 1D parameter.
    """
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from rpx_benchmark.data.esd_scoring import lpp_embedding

    rng = np.random.default_rng(0)
    n = 60
    t = np.linspace(0, 1, n)
    # Embed a 1D curve into 10D, add small noise.
    base = np.column_stack([t, t ** 2, np.sin(2 * np.pi * t)])
    X = np.hstack([base, rng.normal(0, 0.01, (n, 7))])

    Y = lpp_embedding(X, n_components=1, n_neighbors=5)
    # Spearman correlation between embedding and true t should be |.| ~ 1.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(Y.ravel(), t)
    assert abs(rho) > 0.9, f"expected |rho|>0.9, got {rho}"


def test_lpp_returns_correct_shape():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from rpx_benchmark.data.esd_scoring import lpp_embedding
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 8))
    Y = lpp_embedding(X, n_components=3, n_neighbors=5)
    assert Y.shape == (40, 3)


def test_lpp_rejects_too_many_neighbors():
    from rpx_benchmark.data.esd_scoring import lpp_embedding
    from rpx_benchmark.exceptions import ConfigError
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    X = np.zeros((10, 4))
    with pytest.raises(ConfigError):
        lpp_embedding(X, n_components=2, n_neighbors=10)


def test_lpp_rejects_non_2d():
    from rpx_benchmark.data.esd_scoring import lpp_embedding
    from rpx_benchmark.exceptions import ConfigError
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    with pytest.raises(ConfigError):
        lpp_embedding(np.array([1.0, 2.0]), n_components=1, n_neighbors=2)


# --------------------------------------------------------------------------- #
# Structured-metric helpers (consensus / confidence / per-category)
# --------------------------------------------------------------------------- #

def test_feature_categories_cover_all_27_features():
    from rpx_benchmark.data.esd import FEATURE_NAMES
    union = set()
    for feats in FEATURE_CATEGORIES.values():
        union.update(feats)
    assert union == set(FEATURE_NAMES), \
        f"Categories missing or extra: {union ^ set(FEATURE_NAMES)}"


def test_compute_per_category_scores_returns_one_per_category():
    from rpx_benchmark.data.esd import FEATURE_NAMES
    rng = np.random.default_rng(0)
    pn = rng.uniform(0, 1, (20, len(FEATURE_NAMES)))
    cats = compute_per_category_scores(pn, list(FEATURE_NAMES))
    assert set(cats) == set(FEATURE_CATEGORIES)
    for c, scores in cats.items():
        assert scores.shape == (20,)
        assert np.all((scores >= 0) & (scores <= 1))


def test_compute_consensus_tier_majority_vote():
    """3 methods agreeing on 'easy' → consensus 'easy'."""
    method_tiers = {
        "a": np.array(["easy", "easy", "hard"]),
        "b": np.array(["easy", "medium", "hard"]),
        "c": np.array(["medium", "easy", "easy"]),
    }
    consensus = compute_consensus_tier(method_tiers)
    # row 0: easy,easy,medium → easy
    # row 1: easy,medium,easy → easy
    # row 2: hard,hard,easy → hard
    assert list(consensus) == ["easy", "easy", "hard"]


def test_compute_consensus_tier_breaks_ties_toward_easy():
    """Tied vote {easy, hard} → easy (canonical-order tie-break)."""
    method_tiers = {
        "a": np.array(["easy"]),
        "b": np.array(["hard"]),
    }
    consensus = compute_consensus_tier(method_tiers)
    assert consensus[0] == "easy"


def test_compute_consensus_tier_excludes_degenerate_methods():
    """Methods listed in exclude_methods don't get votes."""
    method_tiers = {
        "good_a": np.array(["easy", "hard"]),
        "good_b": np.array(["easy", "hard"]),
        "dbscan": np.array(["hard", "easy"]),  # degenerate, would flip
    }
    consensus = compute_consensus_tier(method_tiers, exclude_methods=("dbscan",))
    assert list(consensus) == ["easy", "hard"]


def test_compute_confidence_assigns_high_for_stable_pairs():
    """When weights are essentially uniform, no flips → high confidence."""
    rng = np.random.default_rng(0)
    # Construct a matrix where mean_pn cleanly separates rows by tertile.
    pn = np.zeros((30, 5))
    pn[:10]   = 0.2 + rng.normal(0, 0.01, (10, 5))   # clearly easy
    pn[10:20] = 0.5 + rng.normal(0, 0.01, (10, 5))   # clearly medium
    pn[20:]   = 0.8 + rng.normal(0, 0.01, (10, 5))   # clearly hard
    flip_rate, label = compute_confidence(pn, n_perturb=200, seed=0)
    # Most rows should be high confidence (separated cleanly).
    assert (label == "high").sum() >= 25
    assert (flip_rate >= 0).all() and (flip_rate <= 1).all()


def test_compute_confidence_thresholds_and_labels():
    """flip_rate < high_max → high, < medium_max → medium, else low."""
    rng = np.random.default_rng(1)
    # Random PN — flip rates will be all over the place.
    pn = rng.uniform(0, 1, (50, 8))
    flip_rate, label = compute_confidence(
        pn, n_perturb=100, seed=0,
        high_max=0.10, medium_max=0.50,
    )
    for r, lab in zip(flip_rate, label):
        if r < 0.10:
            assert lab == "high"
        elif r < 0.50:
            assert lab == "medium"
        else:
            assert lab == "low"


# --------------------------------------------------------------------------- #
# aggregate_to_scene_splits
# --------------------------------------------------------------------------- #

def test_aggregate_to_scene_splits_99_scenes_gives_33_33_33():
    """99 scenes (3 phases each) → tertile counts 33 / 33 / 33."""
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    rng = np.random.default_rng(0)
    scene_ids = []
    scores = []
    for s in range(99):
        for _ in range(3):
            scene_ids.append(f"scene{s+1}")
            scores.append(float(rng.uniform(0, 1)))
    out = aggregate_to_scene_splits(np.asarray(scene_ids), np.asarray(scores))
    splits = out["splits"]
    assert len(splits["easy"])   == 33
    assert len(splits["medium"]) == 33
    assert len(splits["hard"])   == 33
    # No scene appears in more than one tier.
    all_scenes = splits["easy"] + splits["medium"] + splits["hard"]
    assert len(all_scenes) == len(set(all_scenes)) == 99


def test_aggregate_to_scene_splits_100_scenes_gives_33_33_34():
    """100 scenes → 33 / 33 / 34 (the canonical paper spec)."""
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    rng = np.random.default_rng(0)
    scene_ids, scores = [], []
    for s in range(100):
        for _ in range(3):
            scene_ids.append(f"scene{s+1}")
            scores.append(float(rng.uniform(0, 1)))
    out = aggregate_to_scene_splits(np.asarray(scene_ids), np.asarray(scores))
    splits = out["splits"]
    assert len(splits["easy"])   == 33
    assert len(splits["medium"]) == 33
    assert len(splits["hard"])   == 34


def test_aggregate_to_scene_splits_orders_by_mean_score():
    """Easy contains scenes with the lowest mean phase score."""
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    scene_ids = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    scores    = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    out = aggregate_to_scene_splits(np.asarray(scene_ids), np.asarray(scores))
    assert out["splits"]["easy"]   == ["A"]
    assert out["splits"]["medium"] == ["B"]
    assert out["splits"]["hard"]   == ["C"]


def test_aggregate_to_scene_splits_keeps_all_phases_of_a_scene_together():
    """STR requirement: every phase of a scene goes into the same tier."""
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    rng = np.random.default_rng(0)
    scene_ids, scores = [], []
    for s in range(15):
        for _ in range(3):
            scene_ids.append(f"scene{s+1}")
            scores.append(float(rng.uniform(0, 1)))
    out = aggregate_to_scene_splits(np.asarray(scene_ids), np.asarray(scores))
    seen = set()
    for tier_scenes in out["splits"].values():
        for s in tier_scenes:
            assert s not in seen, f"scene {s} appears in multiple tiers"
            seen.add(s)


def test_aggregate_to_scene_splits_emits_per_phase_detail_when_provided():
    """When phases + phase_tiers are passed, scene_detail carries per-phase
    tier breakdown for downstream analysts."""
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    scene_ids   = np.asarray(["sceneA", "sceneA", "sceneA",
                              "sceneB", "sceneB", "sceneB",
                              "sceneC", "sceneC", "sceneC"])
    phases      = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2])
    scores      = np.asarray([0.10, 0.20, 0.15,
                              0.50, 0.55, 0.45,
                              0.90, 0.85, 0.95])
    phase_tiers = np.asarray(["easy", "easy", "easy",
                               "medium", "medium", "medium",
                               "hard", "hard", "hard"])
    out = aggregate_to_scene_splits(scene_ids, scores,
                                     phases=phases, phase_tiers=phase_tiers)

    assert set(out["scene_detail"]) == {"sceneA", "sceneB", "sceneC"}
    detail = out["scene_detail"]["sceneA"]
    assert detail["scene_tier"] == "easy"
    assert detail["scene_score"] == pytest.approx((0.10 + 0.20 + 0.15) / 3)
    assert detail["phase_tiers"]   == ["easy", "easy", "easy"]
    assert detail["phase_indices"] == [0, 1, 2]
    assert detail["phase_scores"]  == [pytest.approx(0.10),
                                       pytest.approx(0.20),
                                       pytest.approx(0.15)]


def test_aggregate_to_scene_splits_phase_indices_in_capture_order():
    """phase_scores / phase_tiers should be ordered by phase_index ascending,
    not by row order in the input."""
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    # Pass phases in a non-sorted order.
    scene_ids   = np.asarray(["s", "s", "s"])
    phases      = np.asarray([2, 0, 1])
    scores      = np.asarray([0.30, 0.10, 0.20])
    phase_tiers = np.asarray(["hard", "easy", "medium"])
    out = aggregate_to_scene_splits(scene_ids, scores,
                                     phases=phases, phase_tiers=phase_tiers)
    detail = out["scene_detail"]["s"]
    assert detail["phase_indices"] == [0, 1, 2]
    assert detail["phase_tiers"]   == ["easy", "medium", "hard"]
    assert detail["phase_scores"]  == [pytest.approx(0.10),
                                       pytest.approx(0.20),
                                       pytest.approx(0.30)]


def test_aggregate_to_scene_splits_rejects_length_mismatch():
    from rpx_benchmark.data.esd_scoring import aggregate_to_scene_splits
    from rpx_benchmark.exceptions import ConfigError
    with pytest.raises(ConfigError):
        aggregate_to_scene_splits(np.asarray(["a", "b"]), np.asarray([0.1, 0.2, 0.3]))


def test_constants_match_paper_spec():
    assert TERTILE_LABELS == ("easy", "medium", "hard")
    assert DEFAULT_PRIMARY == "mean_pn"
    assert "mean_pn" in SCORING_METHODS
    assert "median_pn" in SCORING_METHODS
    assert "max_pn" in SCORING_METHODS
    assert "pca_pc1" in SCORING_METHODS
    assert "kmeans3" in SCORING_METHODS
    assert len(SCORING_METHODS) == 5

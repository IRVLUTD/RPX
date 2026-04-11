"""Unit tests for deployment-readiness computations.

Covers the ESD / WPS / STR / Temporal Stability / SGC helpers in
:mod:`rpx_benchmark.deployment`. The goal is to lock in the algebra
so refactors don't silently change reported numbers.
"""

from __future__ import annotations

import numpy as np

from rpx_benchmark.api import Difficulty, ESD_WEIGHTS, Phase
from rpx_benchmark.deployment import (
    ESDResult,
    StateTransitionRobustnessResult,
    WeightedPhaseScore,
    compute_esd,
    compute_sgc,
    compute_str,
    compute_temporal_stability_depth,
    compute_weighted_phase_score,
)


# --------------------------------------------------------------------------- #
# ESDResult
# --------------------------------------------------------------------------- #

def test_esd_result_weighted_score_matches_formula():
    r = ESDResult(easy=0.10, medium=0.20, hard=0.30, metric_key="absrel")
    expected = 0.25 * 0.10 + 0.35 * 0.20 + 0.40 * 0.30
    assert abs(r.weighted_score() - expected) < 1e-9


def test_esd_result_handles_missing_buckets():
    """When a bucket is None, the remaining weights renormalise."""
    r = ESDResult(easy=0.10, medium=None, hard=0.30, metric_key="absrel")
    # Weighted sum over (0.25 + 0.40), renormalised:
    expected = (0.25 * 0.10 + 0.40 * 0.30) / (0.25 + 0.40)
    assert abs(r.weighted_score() - expected) < 1e-9


def test_esd_weights_sum_to_one():
    assert abs(sum(ESD_WEIGHTS.values()) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# compute_esd
# --------------------------------------------------------------------------- #

def test_compute_esd_buckets_samples_by_difficulty():
    metrics = [{"absrel": 0.10}, {"absrel": 0.20}, {"absrel": 0.30}]
    diffs = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    r = compute_esd(metrics, diffs, "absrel")
    assert r.easy == 0.10
    assert r.medium == 0.20
    assert r.hard == 0.30


def test_compute_esd_ignores_samples_without_metric_key():
    metrics = [{"absrel": 0.10}, {"rmse": 0.2}, {"absrel": 0.30}]
    diffs = [Difficulty.EASY, Difficulty.EASY, Difficulty.EASY]
    r = compute_esd(metrics, diffs, "absrel")
    assert r.easy == 0.20  # mean of 0.10 and 0.30
    assert r.medium is None
    assert r.hard is None


# --------------------------------------------------------------------------- #
# compute_weighted_phase_score
# --------------------------------------------------------------------------- #

def test_compute_weighted_phase_score_produces_per_phase_breakdown():
    metrics = [
        {"absrel": 0.1}, {"absrel": 0.2}, {"absrel": 0.3},  # clutter easy/med/hard
        {"absrel": 0.2}, {"absrel": 0.3}, {"absrel": 0.4},  # interaction
        {"absrel": 0.1}, {"absrel": 0.1}, {"absrel": 0.1},  # clean
    ]
    phases = [Phase.CLUTTER] * 3 + [Phase.INTERACTION] * 3 + [Phase.CLEAN] * 3
    diffs = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD] * 3
    wps = compute_weighted_phase_score(metrics, phases, diffs, "absrel")
    assert isinstance(wps, WeightedPhaseScore)
    # Interaction phase should be worst (highest absrel)
    assert wps.s_interaction > wps.s_clutter
    assert wps.s_clean < wps.s_clutter  # uniform 0.1 clean is best
    assert wps.delta_int > 0  # degrades from clutter to interaction
    assert wps.delta_rec < 0  # recovers from interaction to clean


def test_weighted_phase_score_to_dict_roundtrip():
    metrics = [{"absrel": 0.1}]
    r = compute_weighted_phase_score(
        metrics, [Phase.CLUTTER], [Difficulty.HARD], "absrel",
    )
    d = r.to_dict()
    for key in ("s_clutter", "s_interaction", "s_clean", "s_overall",
                "delta_int", "delta_rec"):
        assert key in d


# --------------------------------------------------------------------------- #
# compute_str
# --------------------------------------------------------------------------- #

def test_compute_str_deltas_signs():
    r = compute_str({
        Phase.CLUTTER: 0.10,
        Phase.INTERACTION: 0.15,
        Phase.CLEAN: 0.12,
    })
    assert isinstance(r, StateTransitionRobustnessResult)
    assert abs(r.str_c_to_i - 0.05) < 1e-9
    assert abs(r.str_i_to_l - (-0.03)) < 1e-9
    assert r.metric_clutter == 0.10
    assert r.metric_interaction == 0.15
    assert r.metric_clean == 0.12


# --------------------------------------------------------------------------- #
# Temporal stability (depth)
# --------------------------------------------------------------------------- #

def test_temporal_stability_identical_frames_is_one():
    """Two copies of the same depth map → perfect stability."""
    d = np.full((10, 10), 2.0, dtype=np.float32)
    r = compute_temporal_stability_depth(
        pred_depths=[d, d.copy()],
        camera_poses=[None, None],
    )
    assert r.num_pairs == 1
    assert r.ts_score == 1.0


def test_temporal_stability_single_frame_returns_neutral():
    """Fewer than 2 frames can't form a pair."""
    d = np.ones((4, 4), dtype=np.float32)
    r = compute_temporal_stability_depth([d], [None])
    assert r.num_pairs == 0
    assert r.ts_score == 1.0


def test_temporal_stability_handles_all_invalid():
    """When both frames have zero valid pixels we return the unity sentinel."""
    d = np.zeros((4, 4), dtype=np.float32)
    r = compute_temporal_stability_depth([d, d.copy()], [None, None])
    assert r.num_pairs == 1
    assert r.ts_score == 1.0


# --------------------------------------------------------------------------- #
# SGC
# --------------------------------------------------------------------------- #

def test_sgc_zero_samples_short_circuits():
    r = compute_sgc(pred_masks=[], pred_depths=[])
    assert r.num_samples == 0
    assert r.sgc_score == 0.0

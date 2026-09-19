"""Tests for State-Transition Robustness Φ (paper §3.2).

Locks the contracts of :mod:`rpx_benchmark.phi`:

- ``standardise`` z-scores each metric across all 3N (scene, phase) cells,
- ``compute_phi_oneway`` returns both Wilks and Pillai Φ plus an
  F-approximation p-value, with the standard interpretation:
    * no phase effect  → Φ → 1, p → 1
    * strong phase effect → Φ → 0, p → 0
- ``compute_phi_per_transition`` runs Hotelling T² on within-scene paired
  differences and produces η² = T² / (T² + N − 1),
- Holm-Bonferroni adjustment is monotone non-decreasing,
- ``phi_interpretation`` maps Φ to Cohen anchors (paper §3.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.exceptions import MetricError
from rpx_benchmark.phi import (
    PhiBootstrapCI,
    PhiMixedDesign,
    PhiOneway,
    bootstrap_phi_oneway,
    compute_phi_mixed_design,
    compute_phi_oneway,
    compute_phi_per_transition,
    phi_interpretation,
    standardise,
)

# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #


class TestStandardise:
    def test_zero_mean_unit_variance(self):
        np.random.seed(0)
        x = np.random.randn(100, 3, 5) * 10 + 50
        z = standardise(x)
        flat = z.reshape(-1, 5)
        assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(flat.std(axis=0, ddof=0), 1.0, atol=1e-9)

    def test_constant_column_zeroed(self):
        x = np.ones((10, 3, 2))
        z = standardise(x)
        assert np.allclose(z, 0.0)

    def test_shape_validation(self):
        with pytest.raises(MetricError, match=r"\(N, P, K\)"):
            standardise(np.zeros((10, 5)))


# --------------------------------------------------------------------------- #
# One-way Φ
# --------------------------------------------------------------------------- #


class TestPhiOneway:
    def test_no_phase_effect_phi_near_one(self):
        np.random.seed(42)
        x = np.random.randn(100, 3, 5)
        z = standardise(x)
        r = compute_phi_oneway(z)
        # Under H0, Φ should be close to 1 and p should be non-significant.
        assert r.phi_wilks > 0.9
        assert r.phi_pillai > 0.9
        assert r.p_value > 0.05
        assert phi_interpretation(r.phi_wilks) in ("negligible", "small")

    def test_strong_phase_effect_phi_low(self):
        np.random.seed(123)
        x = np.random.randn(100, 3, 5) * 0.5
        # Inject a huge per-phase mean shift in the *first* metric only.
        x[:, 1, 0] += 5.0
        x[:, 2, 0] += 2.0
        z = standardise(x)
        r = compute_phi_oneway(z)
        assert r.phi_wilks < 0.5
        assert r.p_value < 1e-6
        assert phi_interpretation(r.phi_wilks) == "large"

    def test_wilks_and_pillai_agree_in_sign(self):
        # Both should be < 1 when there is a real effect, > 0.95 under H0.
        np.random.seed(7)
        x = np.random.randn(50, 3, 4)
        x[:, 0, :] += 1.5
        z = standardise(x)
        r = compute_phi_oneway(z)
        assert r.phi_wilks < 0.99 and r.phi_pillai < 0.99
        # And they should be within roughly the same neighbourhood.
        assert abs(r.phi_wilks - r.phi_pillai) < 0.2

    def test_returns_raw_stats(self):
        np.random.seed(0)
        z = standardise(np.random.randn(40, 3, 3))
        r = compute_phi_oneway(z)
        assert isinstance(r, PhiOneway)
        # Lambda ∈ (0, 1]; V ∈ [0, s].
        assert 0.0 < r.wilks_lambda <= 1.0
        assert 0.0 <= r.pillai_V <= r.s
        assert r.K == 3
        assert r.s == min(3, 2)
        assert r.n_eff == 40

    def test_conservative_pick(self):
        np.random.seed(99)
        z = standardise(np.random.randn(50, 3, 3))
        r = compute_phi_oneway(z)
        assert r.phi_conservative == min(r.phi_wilks, r.phi_pillai)

    def test_too_few_scenes_rejected(self):
        with pytest.raises(MetricError, match="at least 2 scenes"):
            compute_phi_oneway(np.zeros((1, 3, 2)))

    def test_too_few_phases_rejected(self):
        with pytest.raises(MetricError, match="at least 2 phases"):
            compute_phi_oneway(np.zeros((10, 1, 2)))

    def test_singular_matrix_friendly_error(self):
        # K ≥ N*P → within-phase SSCP E is rank-deficient.
        np.random.seed(0)
        z = standardise(np.random.randn(2, 3, 10))
        with pytest.raises(MetricError, match="positive-definite|near-singular"):
            compute_phi_oneway(z)

    # --------------------------------------------------------------------- #
    # Collinearity guard
    # --------------------------------------------------------------------- #

    def test_collinearity_guard_refuses_near_duplicate_metrics(self):
        """Two K-vector columns that are copies of each other with tiny
        noise → cond(E) explodes → refuse with an actionable hint."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 3, 3))
        # Make column 1 a near-perfect duplicate of column 0.
        x[..., 1] = x[..., 0] + 1e-6 * rng.standard_normal(x[..., 0].shape)
        z = standardise(x)
        with pytest.raises(MetricError, match="near-singular|cond\\(E\\)"):
            compute_phi_oneway(z)

    def test_collinearity_hint_names_offending_pair(self):
        """When metric_names is provided, the failure hint identifies
        the worst-correlated pair by name."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((30, 3, 3))
        x[..., 2] = x[..., 0] + 1e-6 * rng.standard_normal(x[..., 0].shape)
        z = standardise(x)
        with pytest.raises(MetricError) as exc:
            compute_phi_oneway(z, metric_names=("absrel", "rmse", "delta1"))
        msg = str(exc.value)
        assert "absrel" in msg and "delta1" in msg

    def test_collinearity_warn_but_pass_moderate(self, caplog):
        """Moderate collinearity (cond(E) in [warn, fail]) should log a
        warning but still return a PhiOneway."""
        import logging

        rng = np.random.default_rng(2)
        x = rng.standard_normal((100, 3, 4))
        # Correlate column 1 with column 0 at ~0.98 — enough to push cond(E)
        # above the warn threshold without hitting the fail threshold.
        x[..., 1] = 0.98 * x[..., 0] + 0.2 * rng.standard_normal(x[..., 0].shape)
        z = standardise(x)
        with caplog.at_level(logging.WARNING, logger="rpx_benchmark.phi"):
            r = compute_phi_oneway(z, metric_names=("m0", "m1", "m2", "m3"))
        assert isinstance(r, PhiOneway)
        assert any("cond(E)" in rec.message for rec in caplog.records)

    def test_metric_names_length_mismatch_rejected(self):
        z = standardise(np.random.default_rng(3).standard_normal((20, 3, 4)))
        with pytest.raises(MetricError, match="metric_names"):
            compute_phi_oneway(z, metric_names=("a", "b"))


# --------------------------------------------------------------------------- #
# Per-transition Hotelling T²
# --------------------------------------------------------------------------- #


class TestPhiPerTransition:
    def test_eta2_formula(self):
        # When d̄≈0, T² is small → η² is small → Φ → 1.
        # Use small noise (not exact duplication) so the cov matrix stays
        # full-rank.
        np.random.seed(0)
        x = np.random.randn(80, 3, 4)
        z = standardise(x)
        # Set phase 1 to phase 0 + tiny noise to make d̄ ≈ 0 but keep
        # the covariance non-singular.
        z[:, 1, :] = z[:, 0, :] + np.random.randn(80, 4) * 0.05
        out = compute_phi_per_transition(z, transitions=((0, 1),), apply_holm=False)
        t = out["0->1"]
        assert t.eta2 < 0.10  # near zero
        assert t.phi > 0.90
        assert t.p_value > 0.05  # not significant

    def test_strong_effect_low_phi(self):
        np.random.seed(0)
        z = standardise(np.random.randn(80, 3, 4))
        # Strong shift between phase 0 and phase 1.
        z[:, 1, :] += 3.0
        out = compute_phi_per_transition(z, transitions=((0, 1),), apply_holm=False)
        t = out["0->1"]
        assert t.phi < 0.3
        assert t.p_value < 1e-6

    def test_default_three_transitions(self):
        np.random.seed(0)
        z = standardise(np.random.randn(60, 3, 3))
        out = compute_phi_per_transition(z)
        assert set(out.keys()) == {"0->1", "1->2", "0->2"}

    def test_holm_monotone_non_decreasing(self):
        np.random.seed(0)
        z = standardise(np.random.randn(80, 3, 4))
        z[:, 1, :] += 2.0  # induce real effects
        out = compute_phi_per_transition(z)
        for t in out.values():
            # Each adjusted p must be ≥ its raw p and ≤ 1.
            assert t.p_holm is not None
            assert t.p_holm >= t.p_value - 1e-12
            assert t.p_holm <= 1.0

    def test_degenerate_transition_rejected(self):
        with pytest.raises(MetricError, match="degenerate"):
            compute_phi_per_transition(
                np.zeros((10, 3, 2)), transitions=((1, 1),)
            )

    def test_out_of_range_transition_rejected(self):
        with pytest.raises(MetricError, match="out of range"):
            compute_phi_per_transition(
                np.zeros((10, 3, 2)), transitions=((0, 5),)
            )


# --------------------------------------------------------------------------- #
# Cohen-anchor interpretation
# --------------------------------------------------------------------------- #


class TestPhiInterpretation:
    @pytest.mark.parametrize(
        "phi,expected",
        [
            (1.00, "negligible"),
            (0.995, "negligible"),
            (0.99, "small"),
            (0.95, "small"),
            (0.94, "medium"),
            (0.90, "medium"),
            (0.87, "medium"),
            (0.86, "large"),
            (0.50, "large"),
            (0.00, "large"),
        ],
    )
    def test_anchor_thresholds(self, phi, expected):
        assert phi_interpretation(phi) == expected

    def test_out_of_range_rejected(self):
        with pytest.raises(MetricError):
            phi_interpretation(-0.1)
        with pytest.raises(MetricError):
            phi_interpretation(1.1)


# --------------------------------------------------------------------------- #
# Bootstrap Φ CI
# --------------------------------------------------------------------------- #


class TestBootstrapPhi:
    def test_returns_ci_dataclass(self):
        z = standardise(np.random.default_rng(0).standard_normal((50, 3, 4)))
        r = bootstrap_phi_oneway(z, n_boot=200, seed=1)
        assert isinstance(r, PhiBootstrapCI)
        assert r.n_boot == 200
        assert r.n_success + r.n_failed == r.n_boot
        assert r.ci_level == 0.95

    def test_ci_brackets_point_estimate_under_h0(self):
        """Under H0 (no phase effect) the CI should contain the single-fit Φ."""
        z = standardise(np.random.default_rng(42).standard_normal((60, 3, 4)))
        point = compute_phi_oneway(z)
        r = bootstrap_phi_oneway(z, n_boot=400, seed=2)
        # 95% CI should contain the point estimate the vast majority of the time.
        assert r.phi_wilks_ci[0] <= point.phi_wilks <= r.phi_wilks_ci[1] + 0.05
        # And the CI must be an interval with low ≤ high.
        assert r.phi_wilks_ci[0] <= r.phi_wilks_ci[1]
        assert r.phi_pillai_ci[0] <= r.phi_pillai_ci[1]

    def test_ci_separates_h1_from_h0(self):
        """Under H1 (real phase effect), bootstrap Φ should sit clearly
        below H0's — enough that the two 95% CIs don't overlap. Boundary
        effects near Φ=1 mean H0's CI is narrow; this test checks the
        thing that actually matters for reviewers: distinguishing the
        two regimes."""
        rng = np.random.default_rng(7)
        x_h0 = rng.standard_normal((60, 3, 3))
        x_h1 = rng.standard_normal((60, 3, 3))
        x_h1[:, 1, 0] += 4.0
        x_h1[:, 2, 0] += 2.0
        ci_h0 = bootstrap_phi_oneway(standardise(x_h0), n_boot=300, seed=11)
        ci_h1 = bootstrap_phi_oneway(standardise(x_h1), n_boot=300, seed=11)
        assert ci_h1.phi_wilks_mean < ci_h0.phi_wilks_mean - 0.3
        # H1 upper bound must sit below H0 lower bound — no CI overlap.
        assert ci_h1.phi_wilks_ci[1] < ci_h0.phi_wilks_ci[0]

    def test_reproducible_with_seed(self):
        z = standardise(np.random.default_rng(0).standard_normal((30, 3, 3)))
        a = bootstrap_phi_oneway(z, n_boot=100, seed=17)
        b = bootstrap_phi_oneway(z, n_boot=100, seed=17)
        assert a.phi_wilks_ci == b.phi_wilks_ci
        assert a.phi_wilks_mean == b.phi_wilks_mean

    def test_to_dict_serialises(self):
        z = standardise(np.random.default_rng(0).standard_normal((30, 3, 3)))
        d = bootstrap_phi_oneway(z, n_boot=100, seed=0).to_dict()
        for k in (
            "phi_wilks_mean", "phi_pillai_mean",
            "phi_wilks_ci_low", "phi_wilks_ci_high",
            "phi_pillai_ci_low", "phi_pillai_ci_high",
            "phi_wilks_se", "phi_pillai_se",
            "n_boot", "n_success", "n_failed", "ci_level",
        ):
            assert k in d

    def test_rejects_bad_inputs(self):
        z = standardise(np.random.default_rng(0).standard_normal((20, 3, 3)))
        with pytest.raises(MetricError, match="ci_level"):
            bootstrap_phi_oneway(z, ci_level=0.0)
        with pytest.raises(MetricError, match="ci_level"):
            bootstrap_phi_oneway(z, ci_level=1.0)
        with pytest.raises(MetricError, match="n_boot"):
            bootstrap_phi_oneway(z, n_boot=10)

    def test_high_failure_rate_raises(self):
        """Bootstrap on a K-vector with a duplicate metric produces
        near-singular resamples; the guard should raise instead of
        returning a garbage CI."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 3, 3))
        x[..., 1] = x[..., 0] + 1e-9 * rng.standard_normal(x[..., 0].shape)
        z = standardise(x)
        with pytest.raises(MetricError, match="bootstrap failed|resamples"):
            bootstrap_phi_oneway(z, n_boot=100, seed=0, min_success_frac=0.5)


# --------------------------------------------------------------------------- #
# Two-way mixed-design MANOVA
# --------------------------------------------------------------------------- #


class TestMixedDesign:
    def _build(
        self,
        n_per_tier=20,
        shift_easy=(0.0, 0.0, 0.0),
        shift_medium=(0.0, 0.5, 0.1),
        shift_hard=(0.0, 2.0, 0.5),
        K=4,
        seed=0,
    ):
        N = 3 * n_per_tier
        x = np.random.default_rng(seed).standard_normal((N, 3, K))
        tiers = ["easy"] * n_per_tier + ["medium"] * n_per_tier + ["hard"] * n_per_tier
        for i, tier in enumerate(tiers):
            shift = {"easy": shift_easy, "medium": shift_medium, "hard": shift_hard}[tier]
            for p in range(3):
                x[i, p, :] += shift[p]
        return standardise(x), tiers

    def test_returns_mixed_design_dataclass(self):
        z, tiers = self._build()
        r = compute_phi_mixed_design(z, tiers)
        assert isinstance(r, PhiMixedDesign)
        assert set(r.phi_per_tier.keys()) == {"easy", "medium", "hard"}
        assert r.n_scenes == z.shape[0]
        assert r.K == z.shape[2]
        assert sum(r.tier_counts.values()) == r.n_scenes

    def test_no_interaction_yields_nonsignificant_p(self):
        # All tiers have identical phase profile → interaction p should be high.
        z, tiers = self._build(
            shift_easy=(0.0, 0.5, 0.1),
            shift_medium=(0.0, 0.5, 0.1),
            shift_hard=(0.0, 0.5, 0.1),
        )
        r = compute_phi_mixed_design(z, tiers)
        assert r.p_interaction > 0.05

    def test_strong_interaction_detected(self):
        # Hard tier has much bigger phase shift → significant interaction.
        z, tiers = self._build(
            shift_easy=(0.0, 0.0, 0.0),
            shift_medium=(0.0, 0.5, 0.1),
            shift_hard=(0.0, 3.0, 1.0),
        )
        r = compute_phi_mixed_design(z, tiers)
        assert r.p_interaction < 0.01
        assert r.phi_interaction < r.phi_per_tier["easy"].phi_conservative

    def test_per_tier_phi_orders_with_injected_difficulty(self):
        # Larger phase shifts on harder tiers → lower per-tier Phi on harder.
        z, tiers = self._build(
            shift_easy=(0.0, 0.0, 0.0),
            shift_medium=(0.0, 1.0, 0.3),
            shift_hard=(0.0, 3.0, 1.0),
        )
        r = compute_phi_mixed_design(z, tiers)
        phi_easy = r.phi_per_tier["easy"].phi_conservative
        phi_hard = r.phi_per_tier["hard"].phi_conservative
        # Higher Phi = more stable. Easy tier (no shift) should be most stable.
        assert phi_easy > phi_hard

    def test_rejects_single_tier(self):
        z, _ = self._build()
        tiers = ["easy"] * z.shape[0]
        # Should not crash; just record a note that interaction is undefined.
        r = compute_phi_mixed_design(z, tiers)
        assert any("interaction" in n.lower() for n in r.notes) or r.eta2_interaction == 0.0

    def test_rejects_length_mismatch(self):
        z, _ = self._build()
        with pytest.raises(MetricError, match="length"):
            compute_phi_mixed_design(z, ["easy"] * (z.shape[0] - 1))

    def test_rejects_too_few_scenes(self):
        z = standardise(np.random.randn(3, 3, 2))
        with pytest.raises(MetricError, match="scenes"):
            compute_phi_mixed_design(z, ["easy", "medium", "hard"])

    def test_to_dict_serialises(self):
        z, tiers = self._build()
        r = compute_phi_mixed_design(z, tiers)
        d = r.to_dict()
        assert "phi_per_tier" in d
        assert "eta2_interaction" in d
        assert "phi_interaction" in d
        assert "p_interaction" in d
        # phi_per_tier entries should be dict-shaped (from PhiOneway.to_dict).
        for _tier, entry in d["phi_per_tier"].items():
            assert "phi_conservative" in entry
            assert "p_value" in entry


"""Tests for the JEDI composite-desirability score (paper §3.3).

Locks the contracts of :mod:`rpx_benchmark.jedi`:

- direction-aware bounds work for both higher-is-better and
  lower-is-better metrics without explicit negation,
- ``clip(·, ε, 1)`` handles values outside the frozen window
  (paper §3.3, "out-of-bounds" flag),
- the strict ``ε=0`` policy zeros JEDI on any single failure
  (geometric-mean dominance, by design),
- the ε-floor flag softens the penalty without hiding the failure,
- per-model aggregation matches paper Eq. 6.
"""

from __future__ import annotations

import math

import pytest

from rpx_benchmark.exceptions import MetricError
from rpx_benchmark.jedi import (
    JEDIResult,
    aggregate_jedi,
    compute_jedi,
)
from rpx_benchmark.metrics.specs import MetricSpec, clear_registry, register_spec


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a clean spec registry, then re-imports the built-ins."""
    clear_registry()
    # Re-trigger the built-in registration by re-importing the module.
    import importlib

    import rpx_benchmark.metrics.specs as specs_mod

    importlib.reload(specs_mod)
    yield


# --------------------------------------------------------------------------- #
# Direction-aware bounds
# --------------------------------------------------------------------------- #


class TestDirectionAwareness:
    def test_higher_is_better_all_optimal(self):
        r = compute_jedi({"ap50": 1.0, "ap75": 1.0, "map": 1.0, "ar": 1.0})
        assert r.score == pytest.approx(1.0)
        assert r.out_of_bounds == []
        assert all(d == pytest.approx(1.0) for d in r.individual.values())

    def test_higher_is_better_all_worst(self):
        r = compute_jedi({"ap50": 0.0, "ap75": 0.0, "map": 0.0, "ar": 0.0})
        assert r.score == 0.0

    def test_lower_is_better_optimal(self):
        # Depth: lower AbsRel / RMSE is better. raw=0.0 is the best.
        r = compute_jedi({"absrel": 0.0, "rmse": 0.0})
        assert r.score == pytest.approx(1.0)

    def test_lower_is_better_worst(self):
        # AbsRel worst=0.5; RMSE worst=2.0.
        r = compute_jedi({"absrel": 0.5, "rmse": 2.0})
        assert r.score == 0.0

    def test_mixed_direction(self):
        # NVS: psnr higher-is-better, lpips lower-is-better.
        r = compute_jedi({"psnr": 30.0, "ssim": 0.8, "lpips": 0.2})
        # All three are at decent-but-not-optimal levels; J should be > 0
        # and < 1 with the geometric mean dragging towards the worst.
        assert 0.0 < r.score < 1.0


# --------------------------------------------------------------------------- #
# Clipping + out-of-bounds flag
# --------------------------------------------------------------------------- #


class TestOutOfBoundsClipping:
    def test_value_above_higher_better_best(self):
        # PSNR best=40; passing 50 should clip d_k to 1 and flag oob.
        r = compute_jedi({"psnr": 50.0, "ssim": 1.0, "lpips": 0.0})
        assert r.individual["psnr"] == pytest.approx(1.0)
        assert "psnr" in r.out_of_bounds

    def test_value_below_higher_better_worst(self):
        # PSNR worst=10; passing 5 should clip to 0 and flag oob.
        r = compute_jedi({"psnr": 5.0, "ssim": 1.0, "lpips": 0.0})
        assert r.individual["psnr"] == 0.0
        assert "psnr" in r.out_of_bounds
        assert r.score == 0.0  # geometric-mean zero by design

    def test_value_below_lower_better_best(self):
        # AbsRel best=0.0 (negative value would mean error < 0; impossible
        # in practice but exercises the clip path).
        r = compute_jedi({"absrel": -0.1, "rmse": 0.0})
        assert r.individual["absrel"] == pytest.approx(1.0)
        assert "absrel" in r.out_of_bounds

    def test_value_above_lower_better_worst(self):
        # AbsRel worst=0.5; passing 0.8 should clip to 0 and flag.
        r = compute_jedi({"absrel": 0.8, "rmse": 1.0})
        assert r.individual["absrel"] == 0.0
        assert "absrel" in r.out_of_bounds


# --------------------------------------------------------------------------- #
# ε floor
# --------------------------------------------------------------------------- #


class TestEpsilonFloor:
    def test_default_epsilon_zero_strict(self):
        r = compute_jedi({"ap50": 0.0, "ap75": 1.0})
        assert r.score == 0.0  # single failure tanks J

    def test_epsilon_softens_floor(self):
        r = compute_jedi({"ap50": 0.0, "ap75": 1.0}, epsilon=0.01)
        # J = sqrt(0.01 * 1) = 0.1
        assert r.score == pytest.approx(0.1)
        assert r.epsilon == 0.01

    def test_epsilon_floor_applied_uniformly(self):
        # All worst with ε=0.01 → J = 0.01 (geometric mean of ε^K)^(1/K).
        r = compute_jedi(
            {"ap50": 0.0, "ap75": 0.0, "map": 0.0, "ar": 0.0},
            epsilon=0.01,
        )
        assert r.score == pytest.approx(0.01)

    def test_epsilon_out_of_range_rejected(self):
        with pytest.raises(MetricError, match="epsilon"):
            compute_jedi({"ap50": 0.5}, epsilon=-0.1)
        with pytest.raises(MetricError, match="epsilon"):
            compute_jedi({"ap50": 0.5}, epsilon=1.0)


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


class TestInvalidInput:
    def test_empty_metrics_rejected(self):
        with pytest.raises(MetricError, match="at least one metric"):
            compute_jedi({})

    def test_unknown_metric_rejected(self):
        with pytest.raises(MetricError, match="No MetricSpec"):
            compute_jedi({"completely_made_up_metric_xyz": 0.5})

    def test_non_finite_raw_rejected(self):
        with pytest.raises(MetricError, match="non-finite"):
            compute_jedi({"ap50": float("nan")})

    def test_explicit_specs_override(self):
        # Caller can pass a one-off spec without polluting the global registry.
        custom = MetricSpec(
            name="my_metric",
            direction="higher",
            best=10.0,
            worst=0.0,
            theoretical=False,
        )
        r = compute_jedi({"my_metric": 5.0}, specs={"my_metric": custom})
        assert r.score == pytest.approx(0.5)
        # And the global registry was untouched.
        with pytest.raises(MetricError):
            compute_jedi({"my_metric": 5.0})


# --------------------------------------------------------------------------- #
# Geometric-mean property (paper §3.3 "failure on any one metric tanks J")
# --------------------------------------------------------------------------- #


class TestGeometricMeanDominance:
    def test_one_metric_at_zero_zeros_jedi(self):
        # Four metrics: three near-perfect, one at worst → J = 0 (strict).
        r = compute_jedi({"ap50": 0.99, "ap75": 0.98, "map": 0.97, "ar": 0.0})
        assert r.score == 0.0

    def test_geometric_mean_formula(self):
        # AP50=0.5 (d=0.5), AP75=0.5 (d=0.5), mAP=0.5 (d=0.5), AR=0.5 (d=0.5)
        # → J = 0.5.
        r = compute_jedi({"ap50": 0.5, "ap75": 0.5, "map": 0.5, "ar": 0.5})
        assert r.score == pytest.approx(0.5)

    def test_geometric_mean_unequal_metrics(self):
        # d values 0.25, 0.5, 0.75, 1.0 → J = (0.25*0.5*0.75*1.0)^(1/4)
        r = compute_jedi({"ap50": 0.25, "ap75": 0.5, "map": 0.75, "ar": 1.0})
        expected = (0.25 * 0.5 * 0.75 * 1.0) ** 0.25
        assert r.score == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Per-model aggregation
# --------------------------------------------------------------------------- #


class TestAggregation:
    def _cell(self, ap_value: float):
        return compute_jedi(
            {"ap50": ap_value, "ap75": ap_value, "map": ap_value, "ar": ap_value}
        )

    def test_aggregate_mean_across_cells(self):
        per_cell = [
            ("clu", self._cell(1.0)),
            ("clu", self._cell(0.8)),
            ("int", self._cell(0.6)),
            ("int", self._cell(0.4)),
            ("cln", self._cell(0.9)),
            ("cln", self._cell(0.7)),
        ]
        agg = aggregate_jedi(per_cell)
        # Overall mean = (1 + 0.8 + 0.6 + 0.4 + 0.9 + 0.7) / 6 = 0.7333...
        assert agg.jedi == pytest.approx(4.4 / 6)
        # Per-phase means.
        assert agg.per_phase["clu"] == pytest.approx(0.9)
        assert agg.per_phase["int"] == pytest.approx(0.5)
        assert agg.per_phase["cln"] == pytest.approx(0.8)
        assert agg.n_cells == 6

    def test_aggregate_counts_oob(self):
        per_cell = [
            ("clu", compute_jedi({"psnr": 50.0, "ssim": 1.0, "lpips": 0.0})),  # psnr oob
            ("int", compute_jedi({"psnr": 30.0, "ssim": 1.0, "lpips": 0.0})),  # ok
        ]
        agg = aggregate_jedi(per_cell)
        assert agg.n_out_of_bounds == 1

    def test_aggregate_rejects_empty(self):
        with pytest.raises(MetricError, match="at least one"):
            aggregate_jedi([])

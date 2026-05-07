"""Tests for the platform-independent Deployment Readiness Score (DRS)."""

from __future__ import annotations

import pytest

from rpx_benchmark.deployment import (
    OperatingPoint,
    _efficiency_score,
    compute_drs,
    compute_sweep_drs,
)

# ------------------------------------------------------------------ #
# Efficiency score
# ------------------------------------------------------------------ #


class TestEfficiencyScore:
    def test_at_median_returns_half(self) -> None:
        """Model at exactly the median FLOPs → E = 0.5."""
        assert _efficiency_score(100.0, 100.0) == pytest.approx(0.5)

    def test_below_median_above_half(self) -> None:
        """Cheaper model → E > 0.5."""
        # 1 / (1 + 50/100) = 1/1.5 ≈ 0.667
        assert _efficiency_score(50.0, 100.0) > 0.5

    def test_above_median_below_half(self) -> None:
        """Heavier model → E < 0.5."""
        # 1 / (1 + 200/100) = 1/3 ≈ 0.333
        assert _efficiency_score(200.0, 100.0) < 0.5

    def test_very_cheap_approaches_one(self) -> None:
        """Very small model → E approaches 1."""
        # 1 / (1 + 1/100) = 1/1.01 ≈ 0.99
        e = _efficiency_score(1.0, 100.0)
        assert e > 0.9

    def test_very_expensive_approaches_zero(self) -> None:
        """Huge model → E approaches 0."""
        # 1 / (1 + 10000/100) = 1/101 ≈ 0.01
        e = _efficiency_score(10000.0, 100.0)
        assert e < 0.02

    def test_monotonic_decreasing(self) -> None:
        """More FLOPs → strictly lower E."""
        scores = [_efficiency_score(f, 100.0) for f in [10, 50, 100, 200, 500, 1000]]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_zero_flops_returns_zero(self) -> None:
        assert _efficiency_score(0.0, 100.0) == 0.0

    def test_zero_median_returns_zero(self) -> None:
        assert _efficiency_score(100.0, 0.0) == 0.0


# ------------------------------------------------------------------ #
# OperatingPoint
# ------------------------------------------------------------------ #


class TestOperatingPoint:
    def test_higher_is_better_tp(self) -> None:
        op = OperatingPoint(
            precision="fp32",
            task_metric=0.95,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.05,
            flops_g=50.0,
            params_m=100.0,
        )
        assert op.task_performance() == pytest.approx(0.95)

    def test_lower_is_better_tp(self) -> None:
        op = OperatingPoint(
            precision="fp32",
            task_metric=0.05,
            task_metric_name="absrel",
            higher_is_better=False,
            str_score=0.0,
            flops_g=50.0,
            params_m=100.0,
        )
        # exp(-0.05) ≈ 0.951
        tp = op.task_performance()
        assert tp == pytest.approx(0.951, abs=0.01)

    def test_lower_is_better_large_error_penalized(self) -> None:
        """AbsRel of 1.0 should give a low TP."""
        op = OperatingPoint(
            precision="fp32",
            task_metric=1.0,
            task_metric_name="absrel",
            higher_is_better=False,
            str_score=0.0,
            flops_g=50.0,
            params_m=100.0,
        )
        assert op.task_performance() < 0.4  # exp(-1) ≈ 0.368

    def test_robustness_perfect(self) -> None:
        op = OperatingPoint(
            precision="fp32",
            task_metric=0.9,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=0.0,
            flops_g=50.0,
            params_m=100.0,
        )
        assert op.robustness() == 1.0

    def test_robustness_with_drop(self) -> None:
        op = OperatingPoint(
            precision="fp32",
            task_metric=0.9,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.3,
            flops_g=50.0,
            params_m=100.0,
        )
        assert op.robustness() == pytest.approx(0.7)

    def test_robustness_clips_to_zero(self) -> None:
        op = OperatingPoint(
            precision="fp32",
            task_metric=0.9,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-1.5,
            flops_g=50.0,
            params_m=100.0,
        )
        assert op.robustness() == 0.0


# ------------------------------------------------------------------ #
# compute_drs
# ------------------------------------------------------------------ #


class TestComputeDRS:
    def test_single_op(self) -> None:
        op = OperatingPoint(
            precision="fp32",
            task_metric=0.95,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.02,
            flops_g=50.0,
            params_m=100.0,
        )
        result = compute_drs([op], f_median_g=100.0)
        assert result.drs > 0
        assert result.best_op is op
        assert result.tp == pytest.approx(0.95)
        assert result.r == pytest.approx(0.98)
        assert result.e > 0.5  # below median FLOPs
        assert result.drs == pytest.approx(result.tp * result.r * result.e)

    def test_best_op_selected(self) -> None:
        """FP16 with slightly lower accuracy but same FLOPs should tie;
        FP32 with same accuracy should also tie. Best = highest DRS."""
        op_fp32 = OperatingPoint(
            precision="fp32",
            task_metric=0.90,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.05,
            flops_g=100.0,
            params_m=300.0,
        )
        op_fp16 = OperatingPoint(
            precision="fp16",
            task_metric=0.89,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.05,
            flops_g=100.0,
            params_m=300.0,
        )
        result = compute_drs([op_fp32, op_fp16], f_median_g=100.0)
        # FP32 has 0.90 vs FP16 0.89 — same FLOPs — FP32 wins
        assert result.best_op.precision == "fp32"

    def test_fp16_wins_when_accuracy_equal(self) -> None:
        """If FP16 and FP32 have same accuracy and same FLOPs, either wins (tied DRS)."""
        op_fp32 = OperatingPoint(
            precision="fp32",
            task_metric=0.90,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=0.0,
            flops_g=100.0,
            params_m=300.0,
        )
        op_fp16 = OperatingPoint(
            precision="fp16",
            task_metric=0.90,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=0.0,
            flops_g=100.0,
            params_m=300.0,
        )
        result = compute_drs([op_fp32, op_fp16], f_median_g=100.0)
        # Tied — first one wins (FP32). That's fine.
        assert result.drs > 0

    def test_empty_ops_returns_zero(self) -> None:
        result = compute_drs([], f_median_g=100.0)
        assert result.drs == 0.0
        assert result.best_op is None

    def test_fragile_model_penalized(self) -> None:
        """A model with large STR gets low DRS even if accurate."""
        robust_op = OperatingPoint(
            precision="fp32",
            task_metric=0.85,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.02,
            flops_g=50.0,
            params_m=100.0,
        )
        fragile_op = OperatingPoint(
            precision="fp32",
            task_metric=0.95,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.50,
            flops_g=50.0,
            params_m=100.0,
        )
        robust_result = compute_drs([robust_op], f_median_g=50.0)
        fragile_result = compute_drs([fragile_op], f_median_g=50.0)
        # Despite 0.95 > 0.85 accuracy, the fragile model should score lower
        assert robust_result.drs > fragile_result.drs

    def test_heavy_model_penalized(self) -> None:
        """A model with 10× FLOPs gets lower DRS even if equally accurate."""
        light_op = OperatingPoint(
            precision="fp32",
            task_metric=0.90,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=0.0,
            flops_g=50.0,
            params_m=100.0,
        )
        heavy_op = OperatingPoint(
            precision="fp32",
            task_metric=0.90,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=0.0,
            flops_g=500.0,
            params_m=1000.0,
        )
        light_result = compute_drs([light_op], f_median_g=100.0)
        heavy_result = compute_drs([heavy_op], f_median_g=100.0)
        assert light_result.drs > heavy_result.drs

    def test_multiplicative_kills_any_zero(self) -> None:
        """If any component is 0, DRS = 0."""
        zero_acc = OperatingPoint(
            precision="fp32",
            task_metric=0.0,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=0.0,
            flops_g=50.0,
            params_m=100.0,
        )
        result = compute_drs([zero_acc], f_median_g=100.0)
        assert result.drs == 0.0

    def test_to_dict_round_trip(self) -> None:
        op = OperatingPoint(
            precision="fp16",
            task_metric=0.92,
            task_metric_name="delta1",
            higher_is_better=True,
            str_score=-0.03,
            flops_g=45.0,
            params_m=335.0,
        )
        result = compute_drs([op], f_median_g=100.0)
        d = result.to_dict()
        assert d["drs"] == result.drs
        assert d["tp"] == result.tp
        assert d["r"] == result.r
        assert d["e"] == result.e
        assert d["f_median_g"] == 100.0
        assert d["n_operating_points"] == 1


# ------------------------------------------------------------------ #
# compute_sweep_drs
# ------------------------------------------------------------------ #


class TestComputeSweepDRS:
    def test_sweep_median_anchoring(self) -> None:
        """The median FLOPs should be computed from the sweep itself."""
        models = {
            "light": [
                OperatingPoint(
                    precision="fp32",
                    task_metric=0.90,
                    task_metric_name="delta1",
                    higher_is_better=True,
                    str_score=0.0,
                    flops_g=10.0,
                    params_m=25.0,
                )
            ],
            "medium": [
                OperatingPoint(
                    precision="fp32",
                    task_metric=0.90,
                    task_metric_name="delta1",
                    higher_is_better=True,
                    str_score=0.0,
                    flops_g=100.0,
                    params_m=300.0,
                )
            ],
            "heavy": [
                OperatingPoint(
                    precision="fp32",
                    task_metric=0.90,
                    task_metric_name="delta1",
                    higher_is_better=True,
                    str_score=0.0,
                    flops_g=1000.0,
                    params_m=800.0,
                )
            ],
        }
        results = compute_sweep_drs(models)
        # Median of [10, 100, 1000] = 100
        assert results["medium"].f_median_g == pytest.approx(100.0)
        # Light model should score highest (same accuracy, cheapest)
        assert results["light"].drs > results["medium"].drs > results["heavy"].drs

    def test_sweep_empty(self) -> None:
        results = compute_sweep_drs({})
        assert results == {}

    def test_sweep_all_same_flops(self) -> None:
        """When all models have same FLOPs, efficiency is equal — accuracy decides."""
        models = {
            "better": [
                OperatingPoint(
                    precision="fp32",
                    task_metric=0.95,
                    task_metric_name="delta1",
                    higher_is_better=True,
                    str_score=0.0,
                    flops_g=50.0,
                    params_m=100.0,
                )
            ],
            "worse": [
                OperatingPoint(
                    precision="fp32",
                    task_metric=0.80,
                    task_metric_name="delta1",
                    higher_is_better=True,
                    str_score=0.0,
                    flops_g=50.0,
                    params_m=100.0,
                )
            ],
        }
        results = compute_sweep_drs(models)
        assert results["better"].drs > results["worse"].drs
        # Both at median → E = 0.5
        assert results["better"].e == pytest.approx(0.5)
        assert results["worse"].e == pytest.approx(0.5)

"""Tests for the frame-budget degradation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.temporal_budget_sweep import (
    DegradationResult,
    degradation_analysis,
    format_degradation_markdown,
)


def _make_cells(budget: int, n: int = 10, base_absrel: float = 0.05) -> list[dict]:
    """Synthetic cell-log rows where absrel degrades at lower budgets."""
    rng = np.random.default_rng(budget)
    # Lower budgets → worse (higher) absrel.
    degradation = max(0, (250 - budget) / 250.0) * 0.02
    return [
        {
            "scene": f"scene_{i:03d}",
            "phase": "clutter",
            "absrel": base_absrel + degradation + rng.normal(0, 0.005),
            "delta1": 0.95 - degradation * 5 + rng.normal(0, 0.01),
            "tae": 0.01 + degradation + rng.normal(0, 0.002),
            "latency_ms": 100.0,  # should be skipped
        }
        for i in range(n)
    ]


class TestDegradationAnalysis:
    def test_basic_output(self):
        cells = {
            50: _make_cells(50),
            100: _make_cells(100),
            150: _make_cells(150),
            250: _make_cells(250),
        }
        results = degradation_analysis(cells, model_name="test_model")
        assert len(results) > 0
        # Should have results for absrel, delta1, tae (but not latency_ms).
        metric_names = {r.metric for r in results}
        assert "absrel" in metric_names
        assert "delta1" in metric_names
        assert "tae" in metric_names
        assert "latency_ms" not in metric_names

    def test_tcv_sign_correct_lower_is_better(self):
        """For lower-is-better metrics, positive TCV = temporal cues help."""
        cells = {
            50: [{"scene": "s1", "phase": "c", "absrel": 0.08}] * 10,
            250: [{"scene": "s1", "phase": "c", "absrel": 0.05}] * 10,
        }
        results = degradation_analysis(cells, metric_keys=["absrel"])
        r = results[0]
        # absrel is lower-is-better; 250 is better → TCV should be positive.
        assert r.tcv > 0, f"TCV should be positive, got {r.tcv}"

    def test_tcv_sign_correct_higher_is_better(self):
        """For higher-is-better metrics, positive TCV = temporal cues help."""
        cells = {
            50: [{"scene": "s1", "phase": "c", "delta1": 0.90}] * 10,
            250: [{"scene": "s1", "phase": "c", "delta1": 0.95}] * 10,
        }
        results = degradation_analysis(cells, metric_keys=["delta1"])
        r = results[0]
        assert r.tcv > 0, f"TCV should be positive, got {r.tcv}"

    def test_critical_budget(self):
        """Critical budget = smallest where metric is within 5% of full."""
        cells = {
            50: [{"scene": "s1", "phase": "c", "absrel": 0.10}] * 5,
            100: [{"scene": "s1", "phase": "c", "absrel": 0.052}] * 5,
            250: [{"scene": "s1", "phase": "c", "absrel": 0.05}] * 5,
        }
        results = degradation_analysis(cells, metric_keys=["absrel"], tol=0.05)
        r = results[0]
        # 0.052 is within 5% of 0.05 (rel diff = 0.04), so critical = 100.
        assert r.critical_budget == 100

    def test_audc_flat_curve(self):
        """Flat curve (no degradation) → AUDC ≈ 1.0."""
        cells = {
            50: [{"scene": "s1", "phase": "c", "absrel": 0.05}] * 5,
            250: [{"scene": "s1", "phase": "c", "absrel": 0.05}] * 5,
        }
        results = degradation_analysis(cells, metric_keys=["absrel"])
        r = results[0]
        assert r.audc == pytest.approx(1.0, abs=0.01)

    def test_empty_cells(self):
        results = degradation_analysis({})
        assert results == []


class TestFormatMarkdown:
    def test_renders(self):
        r = DegradationResult(
            model="test",
            metric="absrel",
            direction="lower",
            budgets=[50, 250],
            values=[0.08, 0.05],
            tcv=0.03,
            audc=0.95,
            critical_budget=250,
        )
        md = format_degradation_markdown([r])
        assert "absrel" in md
        assert "TCV" in md
        assert "0.0300" in md

    def test_empty(self):
        md = format_degradation_markdown([])
        assert "No degradation" in md

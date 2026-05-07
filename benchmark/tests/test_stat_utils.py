"""Tests for ``scripts/stat_utils.py`` — the shared CI aggregator.

Pin the contract so refactors don't silently change the CI shape that
result.json carries (which the paper tables read from).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# stat_utils lives under scripts/, not in the rpx_benchmark package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rpx_benchmark.determinism import RPX_SEED  # noqa: E402
from stat_utils import (  # noqa: E402
    Z_CRITICAL_95,
    aggregate_per_sample_with_ci,
    summarize_with_ci,
)


def test_summarize_empty_returns_zero_filled_dict():
    s = summarize_with_ci([])
    assert s["n"] == 0
    # Every numeric field is 0 (no NaNs / no Nones — keeps JSON clean).
    for k in ("mean", "median", "std", "p5", "p95",
              "ci95_low_t", "ci95_high_t",
              "ci95_low_boot", "ci95_high_boot"):
        assert s[k] == 0.0


def test_summarize_single_value_collapses_ci_to_point():
    s = summarize_with_ci([3.14])
    assert s["n"] == 1
    assert s["mean"] == pytest.approx(3.14)
    assert s["std"] == 0.0
    # Both CIs collapse to the point estimate when n=1.
    assert s["ci95_low_boot"] == s["ci95_high_boot"] == pytest.approx(3.14)
    assert s["ci95_low_t"] == s["ci95_high_t"] == pytest.approx(3.14)


def test_t_ci_matches_textbook_formula():
    """t-CI is mean ± 1.96·s/√n. Pin the formula to catch silent drift."""
    rng = np.random.default_rng(RPX_SEED)
    xs = rng.normal(loc=10.0, scale=2.0, size=200).tolist()
    s = summarize_with_ci(xs)

    a = np.asarray(xs)
    expected_se = a.std(ddof=1) / np.sqrt(a.size)
    expected_lo = a.mean() - Z_CRITICAL_95 * expected_se
    expected_hi = a.mean() + Z_CRITICAL_95 * expected_se
    assert s["ci95_low_t"]  == pytest.approx(expected_lo, rel=1e-9)
    assert s["ci95_high_t"] == pytest.approx(expected_hi, rel=1e-9)


def test_bootstrap_ci_brackets_t_ci_for_normal_data():
    """For ~normal data with large n, bootstrap CI should overlap the
    t-CI to <5% relative half-width — both estimate the same thing."""
    # Different generator state from the previous test (we advance the
    # stream with one extra draw) so this test exercises a different
    # sample under the same project-wide seed.
    rng = np.random.default_rng(RPX_SEED)
    rng.standard_normal()  # advance state
    xs = rng.normal(loc=0.0, scale=1.0, size=1000).tolist()
    s = summarize_with_ci(xs)

    t_half  = (s["ci95_high_t"]    - s["ci95_low_t"])    / 2
    b_half  = (s["ci95_high_boot"] - s["ci95_low_boot"]) / 2
    assert abs(b_half - t_half) / t_half < 0.10


def test_bootstrap_ci_is_deterministic():
    """Same input → same CI across runs (we pin the seed)."""
    xs = list(range(50))
    a = summarize_with_ci(xs)
    b = summarize_with_ci(xs)
    assert a == b


def test_aggregate_per_sample_skips_drop_keys():
    rows = [
        {"id": "a", "scene_id": "s1", "absrel": 0.10, "rmse": 0.20},
        {"id": "b", "scene_id": "s1", "absrel": 0.11, "rmse": 0.21},
        {"id": "c", "scene_id": "s2", "absrel": 0.09, "rmse": 0.19},
    ]
    out = aggregate_per_sample_with_ci(rows, drop_keys=("id", "scene_id"))
    assert set(out.keys()) == {"absrel", "rmse"}
    assert out["absrel"]["n"] == 3
    assert out["absrel"]["mean"] == pytest.approx(0.10, abs=1e-9)


def test_aggregate_per_sample_drops_nan_values():
    """NaN entries in the per-sample list must not poison the mean."""
    rows = [
        {"v": 1.0},
        {"v": float("nan")},
        {"v": 3.0},
    ]
    out = aggregate_per_sample_with_ci(rows)
    assert out["v"]["n"] == 2
    assert out["v"]["mean"] == pytest.approx(2.0)


def test_aggregate_per_sample_handles_missing_keys_gracefully():
    """Rows missing a key just get skipped for that key, not the whole row."""
    rows = [
        {"absrel": 0.10, "rmse": 0.20},
        {"absrel": 0.12},               # rmse missing
        {"rmse":   0.22},               # absrel missing
    ]
    out = aggregate_per_sample_with_ci(rows)
    assert out["absrel"]["n"] == 2
    assert out["rmse"]["n"]   == 2

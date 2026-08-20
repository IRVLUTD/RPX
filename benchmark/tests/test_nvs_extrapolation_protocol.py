"""Locks the RPX NVS interpolation/extrapolation trial geometry."""

from __future__ import annotations

import numpy as np

from rpx_benchmark.nvs_pairs import NVSPairGenerator


def test_k2_extrapolation_targets_are_outside_context_span() -> None:
    generator = object.__new__(NVSPairGenerator)
    pairs = generator._select_context_and_targets(
        list(range(250)), n_context=2, n_targets=25, rng=np.random.default_rng(5_062_026)
    )

    interpolation = [pair for pair in pairs if pair[2] == "interpolation"]
    extrapolation = [pair for pair in pairs if pair[2] == "extrapolation"]
    assert len(interpolation) == 12
    assert len(extrapolation) == 13
    assert all(min(context) < target < max(context) for context, target, _ in interpolation)
    assert all(
        target < min(context) or target > max(context)
        for context, target, _ in extrapolation
    )
    assert any(target > max(context) for context, target, _ in extrapolation)
    assert any(target < min(context) for context, target, _ in extrapolation)


def test_extrapolation_has_twenty_percent_guard_band() -> None:
    generator = object.__new__(NVSPairGenerator)
    extrapolation = [
        pair
        for pair in generator._select_context_and_targets(
            list(range(250)), n_context=2, n_targets=25, rng=np.random.default_rng(1)
        )
        if pair[2] == "extrapolation"
    ]
    for context, target, _ in extrapolation:
        separation = min(abs(target - min(context)), abs(target - max(context)))
        assert separation >= 50

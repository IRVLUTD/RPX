"""Tests for :mod:`rpx_benchmark.determinism`."""

from __future__ import annotations

import random

import numpy as np

from rpx_benchmark.determinism import deterministic, seed_all


def test_seed_all_python_and_numpy_are_deterministic() -> None:
    seed_all(42)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5).tolist()

    seed_all(42)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5).tolist()

    assert py_a == py_b
    assert np_a == np_b


def test_deterministic_context_restores_state() -> None:
    """Entering then exiting the context must restore RNG position.

    Strategy: draw three values from the RNG once without the block,
    then redo the setup and draw the same three values with a
    ``deterministic(0)`` block sandwiched in the middle. The two
    sequences must match — the block must have no net effect on the
    outer RNG state.
    """
    random.seed(1234)
    np.random.seed(1234)
    [random.random() for _ in range(3)]
    np.random.rand(3).tolist()
    baseline_py = [random.random() for _ in range(3)]
    baseline_np = np.random.rand(3).tolist()

    random.seed(1234)
    np.random.seed(1234)
    [random.random() for _ in range(3)]
    np.random.rand(3).tolist()
    with deterministic(0):
        [random.random() for _ in range(10)]
        np.random.rand(10).tolist()
    after_py = [random.random() for _ in range(3)]
    after_np = np.random.rand(3).tolist()

    assert after_py == baseline_py
    assert after_np == baseline_np


def test_seed_all_does_not_raise_without_torch(monkeypatch) -> None:
    """seed_all is safe when torch isn't importable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    seed_all(7)   # must not raise

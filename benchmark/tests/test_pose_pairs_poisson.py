"""Tests for ``scripts/generate_pose_pairs.py``.

Pin the two load-bearing pieces:

1. ``poisson_disk_select`` — the discrete Poisson-disk sampler. Verify
   that (a) no two kept points are within radius, (b) seeding makes
   results reproducible, and (c) cap-by-``max_keep`` honours the limit.
2. ``select_pairs_for_phase`` — the per-(scene, phase) integration.
   Synthesise a 50-frame trajectory with known rotation + translation
   per frame, generate pairs, and check the 2-D distance constraint
   holds in the embedding space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_pose_pairs import (  # noqa: E402
    SamplerConfig,
    poisson_disk_select,
    select_pairs_for_phase,
)

# ────────────────────  poisson_disk_select  ───────────────────────────────


def _pairwise_min(points: np.ndarray, kept: np.ndarray) -> float:
    if kept.size < 2:
        return float("inf")
    sub = points[kept]
    diffs = sub[:, None, :] - sub[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    # Ignore the diagonal (self-distance).
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def test_poisson_disk_respects_radius():
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 50, size=(1000, 2))
    kept = poisson_disk_select(points, radius=2.0, seed=123)
    assert _pairwise_min(points, kept) >= 2.0 - 1e-9
    # On uniform 50×50 with r=2, dense packing → on the order of 200+ kept.
    assert kept.size > 50, f"unexpectedly few kept: {kept.size}"


def test_poisson_disk_is_seeded():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 20, size=(500, 2))
    a = poisson_disk_select(points, radius=1.5, seed=99)
    b = poisson_disk_select(points, radius=1.5, seed=99)
    c = poisson_disk_select(points, radius=1.5, seed=100)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "different seeds should give different orderings"


def test_poisson_disk_honours_max_keep():
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 100, size=(2000, 2))
    kept = poisson_disk_select(points, radius=0.1, seed=1, max_keep=42)
    # radius=0.1 is tiny → algorithm would keep nearly everything. The cap
    # should kick in and stop us at 42.
    assert kept.size == 42


def test_poisson_disk_empty_input():
    out = poisson_disk_select(np.empty((0, 2)), radius=1.0)
    assert out.shape == (0,)


# ────────────────────  select_pairs_for_phase  ────────────────────────────


def _rot_z(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    return np.array(
        [
            [np.cos(t), -np.sin(t), 0.0],
            [np.sin(t), np.cos(t), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _synthetic_trajectory(n: int):
    """Camera rotates 1°/frame about Z and translates 5 cm/frame along X.
    Pair (i, j) has rotation |i-j|° and translation 0.05·|i-j| m."""
    poses = []
    for k in range(n):
        R = _rot_z(1.0 * k)
        t = np.array([0.05 * k, 0.0, 0.0])
        poses.append((R, t))
    return list(range(n)), poses


def test_select_pairs_respects_radius_in_embedding():
    """For a 50-frame synthetic walk, every kept pair must satisfy the
    2-D Poisson-disk constraint in (rot_deg, t_m × scale)."""
    frame_idxs, poses = _synthetic_trajectory(50)
    cfg = SamplerConfig(
        radius_deg=3.0,
        scale_deg_per_m=100.0,
        pairs_per_phase=None,  # exhaustive
        max_frame_gap=49,
        min_rotation_deg=0.0,
        min_translation_m=0.0,
        seed=2026,
    )
    pairs = select_pairs_for_phase(frame_idxs, poses, cfg)
    assert pairs, "synthetic trajectory should yield at least some pairs"

    # Re-embed each kept pair and verify pairwise distance ≥ radius.
    embed = np.array([(p[2], p[3] * cfg.scale_deg_per_m) for p in pairs])
    min_d = _pairwise_min(embed, np.arange(len(embed)))
    assert min_d >= cfg.radius_deg - 1e-9, (
        f"min pairwise distance {min_d:.3f}° < radius {cfg.radius_deg}°"
    )


def test_select_pairs_honours_pairs_per_phase():
    """Hard cap is respected even when many more candidates pass the
    radius constraint."""
    frame_idxs, poses = _synthetic_trajectory(50)
    cfg = SamplerConfig(
        radius_deg=0.5,
        scale_deg_per_m=100.0,
        pairs_per_phase=10,
        max_frame_gap=49,
        min_rotation_deg=0.0,
        min_translation_m=0.0,
        seed=2026,
    )
    pairs = select_pairs_for_phase(frame_idxs, poses, cfg)
    assert len(pairs) <= 10


def test_select_pairs_is_deterministic():
    """Same config + same trajectory → identical kept pairs."""
    frame_idxs, poses = _synthetic_trajectory(40)
    cfg = SamplerConfig(
        radius_deg=2.5,
        scale_deg_per_m=100.0,
        pairs_per_phase=None,
        max_frame_gap=39,
        min_rotation_deg=0.0,
        min_translation_m=0.0,
        seed=11,
    )
    a = select_pairs_for_phase(frame_idxs, poses, cfg)
    b = select_pairs_for_phase(frame_idxs, poses, cfg)
    assert a == b


def test_select_pairs_filters_degenerate():
    """A static camera (no rotation, no translation) yields zero pairs
    once the min-motion thresholds are non-zero."""
    poses = [(np.eye(3), np.zeros(3)) for _ in range(20)]
    cfg = SamplerConfig(
        radius_deg=1.0,
        scale_deg_per_m=100.0,
        pairs_per_phase=None,
        max_frame_gap=19,
        min_rotation_deg=0.1,
        min_translation_m=0.005,
        seed=0,
    )
    assert select_pairs_for_phase(list(range(20)), poses, cfg) == []

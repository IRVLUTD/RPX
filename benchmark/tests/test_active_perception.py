"""Tests for the active-perception metrics + baselines (Task #11).

Locks the contracts the proposal at
``benchmark/docs/methods/active_perception.md`` cites:
``pose_geodesic_deg`` / ``translation_l2_m`` / ``pus_active`` /
``evaluate_active_perception`` (three-axis aggregator), plus the two
baselines that don't need an upstream install (``random_pose`` and
``farthest_point``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.active_perception_metrics import (
    evaluate_active_perception,
    evaluate_active_perception_sample,
    pose_distance_se3,
    pose_geodesic_deg,
    pus_active,
    translation_l2_m,
)

# Make ./scripts importable so the baseline registry is reachable.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from active_perception_models import (  # noqa: E402
    MODEL_REGISTRY,
    list_models,
)
from active_perception_models.farthest_point import FarthestPointNBV  # noqa: E402
from active_perception_models.random_pose import RandomPoseNBV  # noqa: E402


def _pose(t=(0.0, 0.0, 0.0), R: np.ndarray | None = None) -> np.ndarray:
    """4×4 SE(3) builder for tests."""
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def _yaw(angle_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# pose_geodesic_deg
# ─────────────────────────────────────────────────────────────────────────────


class TestPoseGeodesicDeg:
    def test_identity_is_zero(self) -> None:
        assert pose_geodesic_deg(np.eye(3), np.eye(3)) == pytest.approx(0.0)

    def test_90_yaw_is_90_deg(self) -> None:
        R = _yaw(90.0)
        assert pose_geodesic_deg(R, np.eye(3)) == pytest.approx(90.0)
        # Symmetric (predicted vs gt swap shouldn't change distance).
        assert pose_geodesic_deg(np.eye(3), R) == pytest.approx(90.0)

    def test_180_yaw_is_180_deg(self) -> None:
        # Geodesic distance is bounded above by 180°.
        R = _yaw(180.0)
        assert pose_geodesic_deg(R, np.eye(3)) == pytest.approx(180.0, abs=1e-6)

    def test_safe_clip_under_fp_noise(self) -> None:
        """A rotation that's identity-up-to-fp should give ~0°, not nan."""
        eps = 1e-12
        R = np.eye(3) + eps * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        # Not perfectly orthogonal — this stresses the clip path.
        out = pose_geodesic_deg(R, np.eye(3))
        assert math.isfinite(out)
        assert out == pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# translation_l2_m + pose_distance_se3
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslation:
    def test_zero_distance(self) -> None:
        assert translation_l2_m([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_unit_axis(self) -> None:
        assert translation_l2_m([1, 0, 0], [0, 0, 0]) == pytest.approx(1.0)

    def test_pose_distance_combined(self) -> None:
        T1 = _pose((0, 0, 0), _yaw(0.0))
        T2 = _pose((3, 4, 0), _yaw(90.0))  # 3-4-5 triangle in translation
        d = pose_distance_se3(T1, T2)
        assert d["pose_geodesic_deg"] == pytest.approx(90.0)
        assert d["translation_l2_m"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# pus_active
# ─────────────────────────────────────────────────────────────────────────────


class TestPusActive:
    def test_higher_is_better_ratio(self) -> None:
        # Model at pred ≈ 80% of oracle's perf → PUS 0.8.
        assert pus_active(0.8, 1.0, higher_is_better=True) == pytest.approx(0.8)

    def test_lower_is_better_inverted(self) -> None:
        # AbsRel: pred 0.2, oracle 0.1 → ratio 0.5 (oracle is half the error).
        assert pus_active(0.2, 0.1, higher_is_better=False) == pytest.approx(0.5)

    def test_clipped_to_unit_interval(self) -> None:
        # Pred outperforming oracle is clipped to 1.0 (paper convention).
        assert pus_active(1.2, 1.0, higher_is_better=True) == pytest.approx(1.0)

    def test_nan_on_degenerate_oracle(self) -> None:
        assert math.isnan(pus_active(0.5, 0.0))
        assert math.isnan(pus_active(float("nan"), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_active_perception (the sweep aggregator)
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateAggregator:
    def _rows(self) -> list[dict]:
        return [
            {
                "sample_id": f"s{i}",
                "sample_type": "interpolation" if i % 2 == 0 else "extrapolation",
                "difficulty": "easy" if i < 2 else "hard",
                "n_context": 2 + (i % 4) * 2,
                "pose_geodesic_deg": 10.0 * (i + 1),
                "translation_l2_m": 0.1 * (i + 1),
                "pus_active": 0.5 + 0.1 * i if i < 3 else float("nan"),
            }
            for i in range(4)
        ]

    def test_top_level_shape(self) -> None:
        result = evaluate_active_perception(self._rows())
        assert result["n_samples"] == 4
        assert "aggregated" in result
        assert "by_sample_type" in result
        assert "by_difficulty" in result
        assert "by_context_count" in result

    def test_aggregated_pose_geodesic_median(self) -> None:
        result = evaluate_active_perception(self._rows())
        # Pose-geodesic values: [10, 20, 30, 40] → median 25, q25 17.5, q75 32.5.
        agg = result["aggregated"]["pose_geodesic_deg"]
        assert agg["median"] == pytest.approx(25.0)
        assert agg["n"] == pytest.approx(4.0)

    def test_pus_drops_nan_rows(self) -> None:
        """Rows with non-finite pus_active should drop from the
        aggregated pus_active stats so they don't poison the mean."""
        result = evaluate_active_perception(self._rows())
        # 3 finite values [0.5, 0.6, 0.7]; 4th is nan → n=3.
        assert result["aggregated"]["pus_active"]["n"] == pytest.approx(3.0)

    def test_empty_returns_empty(self) -> None:
        assert evaluate_active_perception([]) == {}


# ─────────────────────────────────────────────────────────────────────────────
# random_pose baseline
# ─────────────────────────────────────────────────────────────────────────────


class TestRandomPoseBaseline:
    def test_returns_4x4_from_candidate_pool(self) -> None:
        adapter = RandomPoseNBV()
        ctx = [_pose((0, 0, 0)), _pose((1, 0, 0))]
        candidates = [_pose((i, 0, 0)) for i in range(2, 6)]
        T = adapter(ctx, candidate_poses=candidates)
        assert T.shape == (4, 4)
        # Pick should be one of the candidate centres.
        assert any(np.allclose(T[:3, 3], c[:3, 3]) for c in candidates)

    def test_deterministic_under_seed(self) -> None:
        a = RandomPoseNBV(seed=42)
        b = RandomPoseNBV(seed=42)
        ctx = [_pose((0, 0, 0))]
        cands = [_pose((i, 0, 0)) for i in range(10)]
        # Identical seed → identical sequence of picks.
        assert np.allclose(a(ctx, cands), b(ctx, cands))
        # Three more picks each — sequences stay identical.
        for _ in range(3):
            assert np.allclose(a(ctx, cands), b(ctx, cands))

    def test_fallback_returns_last_context_when_no_pool(self) -> None:
        adapter = RandomPoseNBV()
        ctx = [_pose((0, 0, 0)), _pose((5, 5, 5))]
        T = adapter(ctx)
        assert np.allclose(T[:3, 3], [5, 5, 5])

    def test_raises_on_empty_context(self) -> None:
        adapter = RandomPoseNBV()
        with pytest.raises(ValueError, match="zero context"):
            adapter([])


# ─────────────────────────────────────────────────────────────────────────────
# farthest_point baseline
# ─────────────────────────────────────────────────────────────────────────────


class TestFarthestPointBaseline:
    def test_picks_farthest_candidate(self) -> None:
        adapter = FarthestPointNBV()
        # Context centred at origin (centroid = 0).
        ctx = [_pose((-1, 0, 0)), _pose((1, 0, 0))]
        # Candidates with monotone L2 from origin: 2, 5, 10. Should pick 10.
        cands = [_pose((2, 0, 0)), _pose((5, 0, 0)), _pose((10, 0, 0))]
        T = adapter(ctx, candidate_poses=cands)
        assert np.allclose(T[:3, 3], [10, 0, 0])

    def test_deterministic_no_seed_needed(self) -> None:
        adapter = FarthestPointNBV()
        ctx = [_pose((0, 0, 0))]
        cands = [_pose((1, 1, 1)), _pose((2, 2, 2))]
        # Calling twice gives identical pick — no rng.
        T1 = adapter(ctx, candidate_poses=cands)
        T2 = adapter(ctx, candidate_poses=cands)
        assert np.allclose(T1, T2)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_both_baselines_listed(self) -> None:
        keys = list_models()
        assert "random_pose" in keys
        assert "farthest_point" in keys

    def test_builders_construct_on_bare_env(self) -> None:
        # Both baselines have zero external dependencies.
        for name in ("random_pose", "farthest_point"):
            adapter = MODEL_REGISTRY[name](device="cpu")
            assert hasattr(adapter, "name")
            assert getattr(adapter, "torch_module", None) is None


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: baseline → score
# ─────────────────────────────────────────────────────────────────────────────


def test_end_to_end_score_one_sample() -> None:
    """Smoke: build a synthetic (K context + oracle + candidates) tuple,
    run farthest_point, score against the oracle, verify the row has
    both pose-geodesic and translation keys finite."""
    ctx = [_pose((0, 0, 0)), _pose((1, 0, 0))]
    cands = [_pose((2, 0, 0)), _pose((10, 0, 0)), _pose((5, 0, 0))]
    oracle = _pose((10, 0, 0))  # farthest_point should match this

    adapter = FarthestPointNBV()
    T_pred = adapter(ctx, candidate_poses=cands)
    row = evaluate_active_perception_sample(T_pred, oracle)

    # farthest_point picked the right candidate — zero error.
    assert row["pose_geodesic_deg"] == pytest.approx(0.0)
    assert row["translation_l2_m"] == pytest.approx(0.0)

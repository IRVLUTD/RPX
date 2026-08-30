from __future__ import annotations

import numpy as np

from rpx_benchmark.pose_pairs import (
    VALID_PHASES,
    PairConfig,
    PosePairGenerator,
    _SeqPoses,
)


def _rotation_z(degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sequence() -> _SeqPoses:
    frames = list(range(0, 61))
    return _SeqPoses(
        scene="scene001",
        phase=1,
        frame_idxs=frames,
        rotations=[_rotation_z(frame * 2.0) for frame in frames],
        translations=[np.array([frame * 0.01, 0.0, 0.0]) for frame in frames],
    )


def test_rcpe_uses_clutter_and_clean_phases() -> None:
    assert VALID_PHASES == (0, 2)


def test_intra_phase_pairs_are_exactly_five_frames_apart() -> None:
    generator = PosePairGenerator.__new__(PosePairGenerator)
    generator.cfg = PairConfig(intra_pairs_per_bin=1000, frame_gap=5)
    pairs = generator._intra_pairs(_sequence(), np.random.default_rng(7))
    assert pairs
    assert {pair["frame_idx_b"] - pair["frame_idx"] for pair in pairs} == {5}
    assert {pair["phase"] for pair in pairs} == {1}
    assert {pair["phase_b"] for pair in pairs} == {1}


def test_temporal_chain_edges_are_exact_five_frame_hops() -> None:
    generator = PosePairGenerator.__new__(PosePairGenerator)
    generator.cfg = PairConfig(chain_count=2, chain_length=4, chain_stride=5)
    pairs = generator._temporal_chains(_sequence(), np.random.default_rng(11))
    assert len(pairs) == 8
    assert {pair["frame_idx_b"] - pair["frame_idx"] for pair in pairs} == {5}
    by_chain: dict[str, list[dict]] = {}
    for pair in pairs:
        by_chain.setdefault(pair["metadata"]["chain_id"], []).append(pair)
    for chain in by_chain.values():
        ordered = sorted(chain, key=lambda pair: pair["metadata"]["chain_position"])
        assert all(
            previous["frame_idx_b"] == current["frame_idx"]
            for previous, current in zip(ordered, ordered[1:], strict=False)
        )

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from rpx_benchmark.pose_pairs import (
    VALID_PHASES,
    PairConfig,
    PosePairGenerator,
    _load_pose_file,
    _mode_split,
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


def test_cross_phase_is_disabled_because_capture_worlds_are_unrelated() -> None:
    assert PairConfig().cross_pairs_per_bin == 0


def test_scene_split_ignores_null_rows_and_all_null_scenes() -> None:
    import pandas as pd

    assert _mode_split(pd.Series([None, "easy", "easy", "medium"])) == "easy"
    assert _mode_split(pd.Series([None, None])) is None


def test_compact_hf_pose_vector_is_loaded_in_xyz_xyzw_order(tmp_path) -> None:
    path = tmp_path / "00000.npy"
    np.save(path, np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))

    rotation, translation = _load_pose_file(path)

    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, [1.0, 2.0, 3.0])


def test_pair_paths_preserve_parquet_webp_filenames(tmp_path) -> None:
    generator = PosePairGenerator.__new__(PosePairGenerator)
    generator.root = tmp_path
    generator._frame_filenames = {
        ("scene004", 0, 209): "00209.webp",
        ("scene004", 0, 214): "00214.webp",
    }

    pair = generator._make_entry(
        "scene004", 0, 209, 0, 214, 1.0, 0.1, "intra_phase", "easy"
    )

    assert pair["rgb"].endswith("/00209.webp")
    assert pair["rgb_b"].endswith("/00214.webp")
    assert pair["pose_a"].endswith("/00209.npy")


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


def test_missing_shard_download_is_pinned_to_snapshot_revision(
    tmp_path, monkeypatch
) -> None:
    calls = []
    downloaded = tmp_path / "downloaded.tar"
    downloaded.touch()

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )
    generator = PosePairGenerator.__new__(PosePairGenerator)
    generator._repo_id = "anonymous/RPX"
    generator._snapshot_root = tmp_path / "snapshots" / "pinned-revision"

    assert generator._download_shard("scenes/scene001/0/labels/cam_pose/v1.tar") == downloaded
    assert calls == [
        {
            "repo_id": "anonymous/RPX",
            "filename": "scenes/scene001/0/labels/cam_pose/v1.tar",
            "repo_type": "dataset",
            "revision": "pinned-revision",
        }
    ]

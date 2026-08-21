"""Locks the RPX NVS interpolation/extrapolation trial geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpx_benchmark.nvs_pairs import NVSConfig, NVSPairGenerator


def test_manifest_loader_excludes_sos_scenes_with_null_split(monkeypatch) -> None:
    rows = []
    for scene_id, split in (("scene_easy", "easy"), ("object_sos", None)):
        for frame_idx in range(3):
            rows.append(
                {
                    "scene_id": scene_id,
                    "split": split,
                    "phase": 0,
                    "frame_idx": frame_idx,
                    "has_cam_pose": True,
                    "has_rgb": True,
                    "has_depth": True,
                }
            )
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(rows))

    generator = NVSPairGenerator(
        extracted_root="/unused",
        parquet_path="/unused/frames_v1.parquet",
        split="easy",
    )

    assert set(generator._sequences) == {("scene_easy", 0)}


def test_generator_uses_v2_extensions_declared_by_manifest(tmp_path, monkeypatch) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "current.json").write_text(
        '{"modality_extensions":{"rgb":".webp","depth":".png","cam_pose":".npy"}}'
    )
    rows = [
        {
            "scene_id": "scene_easy",
            "split": "easy",
            "phase": 0,
            "frame_idx": frame_idx,
            "has_cam_pose": True,
            "has_rgb": True,
            "has_depth": True,
        }
        for frame_idx in range(10)
    ]
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(rows))
    generator = NVSPairGenerator(
        extracted_root=tmp_path / "extracted",
        parquet_path=manifest_dir / "frames_v1.parquet",
        split="easy",
        config=NVSConfig(
            context_counts=(2,), targets_per_config=2, include_cross_phase=False
        ),
    )

    sample = next(generator.iter_samples())
    assert all(path.endswith(".webp") for path in sample.context_rgb_paths)
    assert all(path.endswith(".png") for path in sample.context_depth_paths)
    assert all(path.endswith(".npy") for path in sample.context_pose_paths)
    assert sample.target_rgb_path.endswith(".webp")
    assert sample.target_pose_path.endswith(".npy")


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

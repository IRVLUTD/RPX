"""Tests for ``rpx_benchmark.dataset_hub.mock``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.mock import (
    MockSpec,
    generate_mock,
    measure_tree,
)


def test_generate_mock_default_spec_has_expected_top_level(tmp_path: Path):
    out = generate_mock(tmp_path / "ds")
    top = {p.name for p in out.iterdir() if p.is_dir()}
    assert {"mos", "sos"} <= top
    assert len(list((out / "mos").iterdir())) == MockSpec().multi_object_scenes
    assert len(list((out / "sos").iterdir())) == MockSpec().single_object_scenes


def test_multi_object_scene_has_three_phases(tmp_path: Path):
    out = generate_mock(tmp_path / "ds")
    scene = next((out / "mos").iterdir())
    phases = sorted(p.name for p in scene.iterdir())
    assert phases == ["0", "1", "2"]


def test_single_object_scene_has_only_phase_zero(tmp_path: Path):
    out = generate_mock(tmp_path / "ds")
    obj = next((out / "sos").iterdir())
    phases = sorted(p.name for p in obj.iterdir())
    assert phases == ["0"]


def test_phase_dir_has_all_modality_subdirs(tmp_path: Path):
    out = generate_mock(tmp_path / "ds")
    phase = next((out / "mos").iterdir()) / "0"
    expected = {"rgb", "depth", "fisheye", "cam_pose", "sam2"}
    assert {p.name for p in phase.iterdir() if p.is_dir()} == expected


def test_sam2_dir_has_essentials_and_aux_subdirs(tmp_path: Path):
    out = generate_mock(tmp_path / "ds")
    sam2 = next((out / "mos").iterdir()) / "0" / "sam2"
    subdirs = {p.name for p in sam2.iterdir() if p.is_dir()}
    assert "masks" in subdirs
    aux = {"bbox_overlay", "contour_gt_masks", "dino_output",
            "masks_contour_with_hidden", "palette", "rgb_and_mask"}
    assert aux <= subdirs

    files = {p.name for p in sam2.iterdir() if p.is_file()}
    assert "mask_to_object.json" in files
    assert "verified_masks.txt" in files
    assert "iter1_faulty.txt" in files
    assert "iter4_faulty.txt" in files


def test_frame_count_matches_spec(tmp_path: Path):
    spec = MockSpec(multi_object_scenes=1, single_object_scenes=0,
                     phases_per_multi=2, frames_per_phase=7)
    out = generate_mock(tmp_path / "ds", spec)
    rgb_dir = next((out / "mos").iterdir()) / "0" / "rgb"
    assert len(list(rgb_dir.glob("*.png"))) == 7


def test_pngs_have_valid_signature(tmp_path: Path):
    out = generate_mock(tmp_path / "ds", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1))
    rgb = next((next((out / "mos").iterdir()) / "0" / "rgb").glob("*.png"))
    sig = rgb.read_bytes()[:8]
    assert sig == b"\x89PNG\r\n\x1a\n"


def test_cam_pose_files_are_valid_json(tmp_path: Path):
    out = generate_mock(tmp_path / "ds", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=2))
    pose = next(
        (next((out / "mos").iterdir()) / "0" / "cam_pose").glob("*.json"),
    )
    payload = json.loads(pose.read_text(encoding="utf-8"))
    assert "pose" in payload and len(payload["pose"]) == 4


def test_generation_is_deterministic(tmp_path: Path):
    """Re-running with the same spec should produce identical bytes."""
    a = generate_mock(tmp_path / "a", MockSpec(
        multi_object_scenes=1, single_object_scenes=1,
        phases_per_multi=1, frames_per_phase=2))
    b = generate_mock(tmp_path / "b", MockSpec(
        multi_object_scenes=1, single_object_scenes=1,
        phases_per_multi=1, frames_per_phase=2))
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), \
            f"{rel} differs between runs"


def test_measure_tree_counts_files_and_bytes(tmp_path: Path):
    out = generate_mock(tmp_path / "ds", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1))
    files, total = measure_tree(out)
    assert files > 0
    assert total > 0

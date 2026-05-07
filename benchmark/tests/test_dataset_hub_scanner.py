"""Tests for ``rpx_benchmark.dataset_hub.scanner``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.recipes import SceneType
from rpx_benchmark.dataset_hub.scanner import scan_capture_root


@pytest.fixture
def small_mock(tmp_path: Path) -> Path:
    return generate_mock(
        tmp_path / "ds",
        MockSpec(
            multi_object_scenes=2,
            single_object_scenes=3,
            phases_per_multi=3,
            frames_per_phase=4,
        ),
    )


def test_scan_classifies_scene_types(small_mock: Path):
    res = scan_capture_root(small_mock)
    multi = res.by_type(SceneType.MULTI_OBJECT)
    single = res.by_type(SceneType.SINGLE_OBJECT)
    assert len(multi) == 2
    assert len(single) == 3
    assert all(s.scene_type is SceneType.MULTI_OBJECT for s in multi)
    assert all(s.scene_type is SceneType.SINGLE_OBJECT for s in single)


def test_multi_object_phases_have_correct_indices(small_mock: Path):
    res = scan_capture_root(small_mock)
    scene = res.by_type(SceneType.MULTI_OBJECT)[0]
    indices = sorted(p.phase_index for p in scene.phases)
    assert indices == [0, 1, 2]


def test_single_object_has_one_phase(small_mock: Path):
    res = scan_capture_root(small_mock)
    obj = res.by_type(SceneType.SINGLE_OBJECT)[0]
    assert len(obj.phases) == 1
    assert obj.phases[0].phase_index == 0


def test_each_phase_has_top_level_modalities(small_mock: Path):
    res = scan_capture_root(small_mock)
    phase = res.scenes[0].phases[0]
    assert "rgb" in phase.modalities
    assert "depth" in phase.modalities
    assert "fisheye" in phase.modalities
    assert "cam_pose" in phase.modalities
    # sam2/ is split
    assert "sam2" not in phase.modalities


def test_sam2_split_into_subdirs(small_mock: Path):
    res = scan_capture_root(small_mock)
    phase = res.scenes[0].phases[0]
    sam2_keys = [k for k in phase.modalities if k.startswith("sam2/")]
    assert "sam2/masks" in sam2_keys
    assert "sam2/_meta" in sam2_keys  # JSON + txts
    aux = {
        "sam2/bbox_overlay",
        "sam2/contour_gt_masks",
        "sam2/dino_output",
        "sam2/masks_contour_with_hidden",
        "sam2/palette",
        "sam2/rgb_and_mask",
    }
    assert aux.issubset(set(sam2_keys))


def test_modality_inventories_have_nonzero_files_and_bytes(small_mock: Path):
    res = scan_capture_root(small_mock)
    phase = res.scenes[0].phases[0]
    for name, inv in phase.modalities.items():
        assert inv.file_count > 0, f"{name} has no files"
        assert inv.total_bytes > 0, f"{name} has zero bytes"


def test_scan_aggregates_match_individual(small_mock: Path):
    res = scan_capture_root(small_mock)
    expected_files = sum(s.file_count for s in res.scenes)
    expected_bytes = sum(s.total_bytes for s in res.scenes)
    assert res.file_count == expected_files
    assert res.total_bytes == expected_bytes


def test_modality_totals_aggregate_correctly(small_mock: Path):
    res = scan_capture_root(small_mock)
    totals = res.modality_totals()
    assert "rgb" in totals
    # rgb should appear in every phase of every scene (multi: 2*3=6 phases,
    # single: 3*1=3 phases) -> 9 phases × 4 frames = 36 rgb files.
    assert totals["rgb"].file_count == 9 * 4


def test_scan_skips_non_scene_directories(tmp_path: Path):
    small = generate_mock(
        tmp_path / "ds",
        MockSpec(
            multi_object_scenes=1, single_object_scenes=1, phases_per_multi=1, frames_per_phase=1
        ),
    )
    (small / "README.md").write_text("docs", encoding="utf-8")
    (small / "trash").mkdir()
    # A stray file inside mos/ should also be skipped (only directories
    # named like real scenes are picked up).
    (small / "mos" / "stray.txt").write_text("oops", encoding="utf-8")
    res = scan_capture_root(small)
    skipped_names = {p.name for p in res.skipped}
    assert "README.md" in skipped_names
    assert "trash" in skipped_names
    assert "stray.txt" in skipped_names
    assert len(res.scenes) == 2  # 1 multi + 1 single


def test_scan_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        scan_capture_root(tmp_path / "does_not_exist")

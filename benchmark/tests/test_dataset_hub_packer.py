"""Tests for ``rpx_benchmark.dataset_hub.packer``."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import (
    PackPlan,
    pack_capture_tree,
)
from rpx_benchmark.dataset_hub.recipes import SceneType
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.exceptions import ConfigError, DatasetError


@pytest.fixture
def packed(tmp_path: Path):
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=2, single_object_scenes=2,
        phases_per_multi=3, frames_per_phase=3,
    ))
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    res = pack_capture_tree(PackPlan(src_root=src, staging_root=staging), scan)
    return src, staging, scan, res


def test_packer_writes_tars_under_scenes_and_objects(packed):
    src, staging, scan, res = packed
    assert (staging / "scenes").is_dir()
    assert (staging / "objects").is_dir()
    multi = list((staging / "scenes").rglob("*.tar"))
    single = list((staging / "objects").rglob("*.tar"))
    assert multi and single


def test_raw_modalities_at_phase_root_label_modalities_under_labels(packed):
    src, staging, scan, res = packed
    sample_phase = next(iter((staging / "scenes").iterdir())) / "0"
    assert (sample_phase / "rgb.tar").is_file()
    assert (sample_phase / "depth.tar").is_file()
    assert (sample_phase / "fisheye.tar").is_file()
    assert (sample_phase / "cam_pose.tar").is_file()
    assert (sample_phase / "labels" / "masks" / "v1.tar").is_file()
    assert (sample_phase / "labels" / "masks_aux" / "v1.tar").is_file()
    assert (sample_phase / "labels" / "sam2_meta" / "v1.tar").is_file()


def test_pack_result_aggregate_matches_filesystem(packed):
    src, staging, scan, res = packed
    on_disk = sum(p.stat().st_size
                   for p in staging.rglob("*.tar") if p.is_file())
    # PackResult.total_bytes is the sum of *source* file bytes (not the
    # tar overhead). The on-disk bytes should be at least the source
    # total minus a tiny amount of trailing zero padding.
    assert res.total_bytes > 0
    assert on_disk >= res.total_bytes


def test_tar_contents_round_trip_to_source_files(packed):
    src, staging, scan, res = packed
    rgb_shard = next(s for s in res.shards if s.modality == "rgb")
    tar_path = staging / rgb_shard.repo_path
    with tarfile.open(tar_path, "r") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile())
    sub = "scenes/" if rgb_shard.repo_path.startswith("scenes/") else "objects/"
    on_disk_sub = "mos" if sub == "scenes/" else "sos"
    src_dir = src / on_disk_sub / rgb_shard.scene_id / str(rgb_shard.phase) / "rgb"
    src_names = sorted(p.name for p in src_dir.iterdir())
    assert [Path(n).name for n in names] == src_names


def test_masks_aux_collapses_six_aux_subdirs(packed):
    src, staging, scan, res = packed
    aux_shard = next(s for s in res.shards
                       if s.modality == "masks_aux"
                       and s.scene_id == res.shards[0].scene_id)
    # 6 aux subdirs * 3 frames each = 18 files in masks_aux.tar.
    assert aux_shard.file_count == 6 * 3


def test_sam2_meta_picks_up_loose_files(packed):
    src, staging, scan, res = packed
    meta_shard = next(s for s in res.shards if s.modality == "sam2_meta")
    # mask_to_object.json + verified_masks.txt + iter1..4_faulty.txt = 6 files
    assert meta_shard.file_count == 6


def test_packer_is_deterministic(tmp_path: Path):
    """Same source + same plan → byte-identical tars."""
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=1, single_object_scenes=1,
        phases_per_multi=1, frames_per_phase=2,
    ))
    a = tmp_path / "a"
    b = tmp_path / "b"
    scan = scan_capture_root(src)
    pack_capture_tree(PackPlan(src, a), scan)
    pack_capture_tree(PackPlan(src, b), scan)
    a_tars = sorted(p.relative_to(a) for p in a.rglob("*.tar"))
    b_tars = sorted(p.relative_to(b) for p in b.rglob("*.tar"))
    assert a_tars == b_tars
    for rel in a_tars:
        assert hashlib.sha256((a / rel).read_bytes()).hexdigest() == \
                hashlib.sha256((b / rel).read_bytes()).hexdigest(), \
                f"shard {rel} not deterministic"


def test_packer_refuses_to_overwrite_by_default(tmp_path: Path):
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1,
    ))
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack_capture_tree(PackPlan(src, staging), scan)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        pack_capture_tree(PackPlan(src, staging), scan)


def test_packer_overwrite_flag_allows_repack(tmp_path: Path):
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1,
    ))
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack_capture_tree(PackPlan(src, staging), scan)
    # Should not raise.
    pack_capture_tree(PackPlan(src, staging, overwrite=True), scan)


def test_packer_rejects_mismatched_src_root(tmp_path: Path):
    src1 = generate_mock(tmp_path / "src1", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1,
    ))
    src2 = generate_mock(tmp_path / "src2", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1,
    ))
    scan = scan_capture_root(src1)
    with pytest.raises(ConfigError, match="does not match"):
        pack_capture_tree(PackPlan(src2, tmp_path / "stage"), scan)


def test_label_version_lands_in_repo_path(tmp_path: Path):
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=1, single_object_scenes=0,
        phases_per_multi=1, frames_per_phase=1,
    ))
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    res = pack_capture_tree(PackPlan(src, staging, label_version="v7"), scan)
    masks = next(s for s in res.shards if s.modality == "masks")
    assert masks.repo_path.endswith("/labels/masks/v7.tar")


def test_shard_sha256_is_64_hex_chars(packed):
    src, staging, scan, res = packed
    for shard in res.shards:
        assert shard.sha256 is not None
        assert len(shard.sha256) == 64
        assert all(c in "0123456789abcdef" for c in shard.sha256)


def test_scene_type_routes_to_correct_top_level(packed):
    src, staging, scan, res = packed
    multi_shards = [s for s in res.shards if s.repo_path.startswith("scenes/")]
    single_shards = [s for s in res.shards if s.repo_path.startswith("objects/")]
    assert multi_shards and single_shards
    multi_scene_ids = {s.scene_id for s in multi_shards}
    single_scene_ids = {s.scene_id for s in single_shards}
    assert multi_scene_ids.isdisjoint(single_scene_ids)
    # Reconcile against the scan: shards in `scenes/` must come from
    # MOS scenes, shards in `objects/` from SOS scenes.
    expected_multi = {s.scene_id for s in scan.scenes
                       if s.scene_type is SceneType.MULTI_OBJECT}
    expected_single = {s.scene_id for s in scan.scenes
                        if s.scene_type is SceneType.SINGLE_OBJECT}
    assert multi_scene_ids == expected_multi
    assert single_scene_ids == expected_single

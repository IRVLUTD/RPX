"""Tests for ``rpx_benchmark.dataset_hub.manifest``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from rpx_benchmark.dataset_hub.manifest import (
    SCHEMA_VERSION,
    build_frame_manifest,
    read_frame_manifest,
)
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.scanner import scan_capture_root


@pytest.fixture
def packed(tmp_path: Path):
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=2, single_object_scenes=3,
        phases_per_multi=3, frames_per_phase=4,
    ))
    staging = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack = pack_capture_tree(PackPlan(src_root=src, staging_root=staging), scan)
    return src, staging, scan, pack


def test_manifest_files_are_written(packed):
    src, staging, scan, pack = packed
    paths = build_frame_manifest(scan, pack, staging)
    assert paths.parquet_path.is_file()
    assert paths.current_json_path.is_file()
    assert paths.parquet_path.name == f"frames_{SCHEMA_VERSION}.parquet"


def test_manifest_row_count_matches_total_frames(packed):
    src, staging, scan, pack = packed
    paths = build_frame_manifest(scan, pack, staging)
    table = read_frame_manifest(paths.parquet_path)
    # 2 multi-object scenes * 3 phases * 4 frames + 3 single * 1 phase * 4 frames
    expected = 2 * 3 * 4 + 3 * 1 * 4
    assert table.num_rows == expected


def test_manifest_schema_columns(packed):
    src, staging, scan, pack = packed
    paths = build_frame_manifest(scan, pack, staging)
    table = read_frame_manifest(paths.parquet_path)
    expected_cols = {
        "scene_id", "scene_type", "phase", "frame_idx", "frame_filename",
        "split",
        "has_rgb", "has_depth", "has_fisheye", "has_cam_pose",
        "has_masks", "has_masks_aux", "has_sam2_meta",
        "shard_rgb", "shard_depth", "shard_fisheye", "shard_cam_pose",
        "shard_masks", "shard_masks_aux", "shard_sam2_meta",
    }
    assert expected_cols == set(table.column_names)


def test_split_assignment_applies_only_to_multi_object(packed):
    src, staging, scan, pack = packed
    splits = {s.scene_id: "easy"
               for s in scan.scenes
               if s.scene_type.value == "multi_object"}
    paths = build_frame_manifest(scan, pack, staging, splits=splits)
    table = read_frame_manifest(paths.parquet_path)
    df = table.to_pandas()
    multi = df[df["scene_type"] == "multi_object"]
    single = df[df["scene_type"] == "single_object"]
    assert (multi["split"] == "easy").all()
    assert single["split"].isna().all()


def test_has_columns_reflect_packed_modalities(packed):
    src, staging, scan, pack = packed
    paths = build_frame_manifest(scan, pack, staging)
    table = read_frame_manifest(paths.parquet_path)
    df = table.to_pandas()
    # All mock frames have every modality.
    for col in ("has_rgb", "has_depth", "has_fisheye",
                 "has_cam_pose", "has_masks", "has_masks_aux", "has_sam2_meta"):
        assert df[col].all(), f"{col} should be True for every mock frame"


def test_shard_columns_point_to_actual_tar_paths(packed):
    src, staging, scan, pack = packed
    paths = build_frame_manifest(scan, pack, staging)
    table = read_frame_manifest(paths.parquet_path)
    df = table.to_pandas()
    for shard_col in ("shard_rgb", "shard_depth", "shard_masks"):
        for repo_path in df[shard_col].dropna().unique():
            assert (staging / repo_path).is_file(), \
                f"shard column {shard_col} points to missing file {repo_path}"


def test_current_json_records_label_versions(packed):
    src, staging, scan, pack = packed
    versions = {"masks": "v1", "masks_aux": "v1", "sam2_meta": "v1"}
    paths = build_frame_manifest(scan, pack, staging,
                                   label_versions=versions)
    payload = json.loads(paths.current_json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["label_versions"] == versions


def test_split_input_accepts_scene_splits_json_shape(packed):
    src, staging, scan, pack = packed
    multi_ids = {s.scene_id for s in scan.scenes
                  if s.scene_type.value == "multi_object"}
    splits_dict = {sid: "hard" for sid in multi_ids}
    paths = build_frame_manifest(scan, pack, staging, splits=splits_dict)
    table = read_frame_manifest(paths.parquet_path)
    multi_rows = [
        (s, sp) for s, sp, st in zip(
            table["scene_id"].to_pylist(),
            table["split"].to_pylist(),
            table["scene_type"].to_pylist(),
        ) if st == "multi_object"
    ]
    assert all(sp == "hard" for _, sp in multi_rows)


def test_frame_indices_are_sequential_per_phase(packed):
    src, staging, scan, pack = packed
    paths = build_frame_manifest(scan, pack, staging)
    table = read_frame_manifest(paths.parquet_path)
    df = table.to_pandas()
    for (sid, phase), grp in df.groupby(["scene_id", "phase"]):
        idxs = sorted(grp["frame_idx"].tolist())
        assert idxs == list(range(len(idxs))), \
            f"frame_idx not sequential for {sid}/{phase}"

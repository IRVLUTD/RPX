"""Tests for ``rpx_benchmark.dataset_hub.downloader``.

The downloader's network paths are mocked: we verify that the right
allow_patterns are produced, that the recipe / split / scene_type rules
are enforced, and that label_versions are resolved through current.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from rpx_benchmark.dataset_hub import downloader
from rpx_benchmark.dataset_hub.downloader import download_for_task
from rpx_benchmark.dataset_hub.manifest import (
    SCHEMA_VERSION,
    build_frame_manifest,
)
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.recipes import SceneType
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.exceptions import ConfigError, DownloadError


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    """Build a packed staging dir with a manifest already on disk.

    Tests then point the downloader at this dir as if it were the HF
    snapshot root.
    """
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=3, single_object_scenes=2,
        phases_per_multi=3, frames_per_phase=2,
    ))
    out = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack = pack_capture_tree(PackPlan(src_root=src, staging_root=out), scan)

    # Assign the 3 multi-object scenes to the 3 tiers, 1 each.
    multi = [s.scene_id for s in scan.scenes
              if s.scene_type is SceneType.MULTI_OBJECT]
    splits = {multi[0]: "easy", multi[1]: "medium", multi[2]: "hard"}
    build_frame_manifest(scan, pack, out, splits=splits)
    return out


def _patch_hf(monkeypatch, staging: Path) -> MagicMock:
    """Make ``downloader._hub()`` return a fake HF module pointing at staging."""
    fake = MagicMock()

    def fake_hub_download(repo_id, repo_type, filename, revision=None,
                           cache_dir=None):
        path = staging / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        return str(path)

    def fake_snapshot(repo_id, repo_type, allow_patterns,
                       revision=None, cache_dir=None, max_workers=8):
        return str(staging)

    fake.hf_hub_download = fake_hub_download
    fake.snapshot_download = fake_snapshot
    monkeypatch.setattr(downloader, "_hub", lambda: fake)
    return fake


def test_segmentation_easy_resolves_to_one_multi_scene(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(task="segmentation", split="easy",
                              repo_id="acme/RPX")
    assert res.scene_type is SceneType.MULTI_OBJECT
    assert len(res.matched_scenes) == 1   # 3 multi-object scenes / 3 tiers


def test_allow_patterns_have_rgb_and_masks_for_segmentation(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(task="segmentation", split="easy",
                              repo_id="acme/RPX")
    pat_strs = " ".join(res.allow_patterns)
    assert "/rgb.tar" in pat_strs
    assert "/labels/masks/v1.tar" in pat_strs
    assert "/depth.tar" not in pat_strs   # not in recipe
    assert "/labels/masks_aux/" not in pat_strs   # opt-in only


def test_extra_modalities_are_added_to_pull(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(
        task="segmentation", split="easy", repo_id="acme/RPX",
        extra_modalities=("masks_aux",),
    )
    assert any("/labels/masks_aux/v1.tar" in p for p in res.allow_patterns)


def test_label_version_override_changes_pattern(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(
        task="segmentation", split="easy", repo_id="acme/RPX",
        label_versions={"masks": "v9"},
    )
    assert any("/labels/masks/v9.tar" in p for p in res.allow_patterns)


def test_single_object_recipe_must_not_get_a_split(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    with pytest.raises(ConfigError, match="have no splits"):
        download_for_task(task="object_templates", split="easy",
                            repo_id="acme/RPX")


def test_multi_object_recipe_requires_a_split(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    with pytest.raises(ConfigError, match="pick a split"):
        download_for_task(task="segmentation", split=None,
                            repo_id="acme/RPX")


def test_single_object_recipe_pulls_single_scenes(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(task="object_templates", split=None,
                              repo_id="acme/RPX")
    assert res.scene_type is SceneType.SINGLE_OBJECT
    assert res.matched_scenes  # at least one scene matched
    assert all(p.startswith("objects/") for p in res.allow_patterns)


def test_unknown_split_returns_no_scenes_and_raises(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    with pytest.raises(DownloadError, match="No scenes matched"):
        download_for_task(task="segmentation", split="impossible",
                            repo_id="acme/RPX")


def test_manifest_table_returned_filtered_to_slice(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(task="segmentation", split="medium",
                              repo_id="acme/RPX")
    table = res.manifest_table
    assert table is not None
    assert table.num_rows > 0
    splits = set(table["split"].to_pylist())
    types = set(table["scene_type"].to_pylist())
    assert splits == {"medium"}
    assert types == {"multi_object"}


def test_return_manifest_false_skips_table(staging, monkeypatch):
    _patch_hf(monkeypatch, staging)
    res = download_for_task(task="segmentation", split="easy",
                              repo_id="acme/RPX", return_manifest=False)
    assert res.manifest_table is None


def test_label_version_falls_back_to_current_json(staging, monkeypatch):
    """If label_versions={} and current.json says masks=v1, we use v1."""
    # Rewrite current.json to declare masks=v3 — patterns should follow.
    current_path = staging / "manifest" / "current.json"
    current_path.write_text(json.dumps(
        {"label_versions": {"masks": "v3", "masks_aux": "v1", "sam2_meta": "v1"},
         "schema_version": SCHEMA_VERSION},
    ), encoding="utf-8")

    _patch_hf(monkeypatch, staging)
    res = download_for_task(task="segmentation", split="easy",
                              repo_id="acme/RPX")
    assert any("/labels/masks/v3.tar" in p for p in res.allow_patterns)


def test_scene_type_kwarg_disambiguates_collision(staging, monkeypatch):
    """If a recipe key existed in both tables, scene_type would resolve it."""
    _patch_hf(monkeypatch, staging)
    # Use a known single-object recipe with explicit scene_type.
    res = download_for_task(
        task="object_templates", split=None, repo_id="acme/RPX",
        scene_type=SceneType.SINGLE_OBJECT,
    )
    assert res.scene_type is SceneType.SINGLE_OBJECT

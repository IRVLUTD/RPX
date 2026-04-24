"""Tests for ``rpx_benchmark.dataset_hub.dataset_card``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.dataset_card import (
    CardSpec,
    _size_category,
    write_dataset_card,
)
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.exceptions import DatasetError


@pytest.fixture
def scan(tmp_path: Path):
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=2, single_object_scenes=3,
        phases_per_multi=3, frames_per_phase=2,
    ))
    return scan_capture_root(src)


def test_card_lands_at_staging_readme(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan)
    assert out == tmp_path / "README.md"
    assert out.is_file()


def test_card_starts_with_yaml_frontmatter(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    # Frontmatter ends with --- followed by newline.
    end_idx = text.index("\n---\n", 4)
    assert end_idx > 0


def test_card_contains_required_yaml_keys(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan)
    text = out.read_text(encoding="utf-8")
    for key in ("license:", "task_categories:", "tags:", "size_categories:"):
        assert key in text, f"missing required frontmatter key: {key}"


def test_card_includes_scene_counts(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan)
    text = out.read_text(encoding="utf-8")
    n_multi  = sum(1 for s in scan.scenes if s.scene_type.value == "multi_object")
    n_single = sum(1 for s in scan.scenes if s.scene_type.value == "single_object")
    assert f"**{n_multi}**" in text
    assert f"**{n_single}**" in text


def test_card_lists_modalities(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan)
    text = out.read_text(encoding="utf-8")
    for modality in ("rgb", "depth", "fisheye"):
        assert f"`{modality}`" in text, f"modality {modality} missing"


def test_card_recipe_table_present_for_both_families(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan)
    text = out.read_text(encoding="utf-8")
    assert "Multi-object" in text
    assert "Single-object" in text
    assert "segmentation" in text
    assert "object_templates" in text


def test_card_quick_start_uses_default_repo_id(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan, spec=CardSpec(repo_id="acme/RPX"))
    text = out.read_text(encoding="utf-8")
    assert "acme/RPX" in text


def test_card_label_versions_block_renders_when_provided(tmp_path: Path, scan):
    out = write_dataset_card(tmp_path, scan,
                               label_versions={"masks": "v1", "cam_pose": "v2"})
    text = out.read_text(encoding="utf-8")
    assert "| `masks` | `v1` |" in text
    assert "| `cam_pose` | `v2` |" in text


def test_card_splits_advertise_tier_counts(tmp_path: Path, scan):
    multi_ids = [s.scene_id for s in scan.scenes
                  if s.scene_type.value == "multi_object"]
    splits = {multi_ids[0]: "easy", multi_ids[1]: "hard"}
    out = write_dataset_card(tmp_path, scan, splits=splits)
    text = out.read_text(encoding="utf-8")
    assert "config_name: multi_object" in text
    assert 'split: "easy"' in text
    assert 'split: "hard"' in text


def test_card_refuses_overwrite_by_default(tmp_path: Path, scan):
    write_dataset_card(tmp_path, scan)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        write_dataset_card(tmp_path, scan)


def test_card_overwrite_flag_allows_re_render(tmp_path: Path, scan):
    write_dataset_card(tmp_path, scan)
    write_dataset_card(tmp_path, scan, overwrite=True)


def test_size_category_buckets():
    assert _size_category(500_000)             == "100M<n<1B"
    assert _size_category(2 * 1024**3)         == "1B<n<10B"
    assert _size_category(50 * 1024**3)        == "10B<n<100B"
    assert _size_category(890 * 1024**3)       == "100B<n<1T"
    assert _size_category(2 * 1024**4)         == "n>1T"

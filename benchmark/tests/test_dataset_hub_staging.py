"""Tests for ``rpx_benchmark.dataset_hub.staging``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.staging import (
    SCENE_SPLITS_JSON,
    SPLIT_TIER_FILES,
    load_scene_splits,
    stage_paths,
    stage_splits,
)
from rpx_benchmark.exceptions import ConfigError, DatasetError


@pytest.fixture
def splits_src(tmp_path: Path) -> Path:
    """Build a fake `benchmark/data/splits/` source tree."""
    src = tmp_path / "splits_src"
    src.mkdir()
    (src / "easy.txt").write_text("scene1\nscene2\n", encoding="utf-8")
    (src / "medium.txt").write_text("scene3\n", encoding="utf-8")
    (src / "hard.txt").write_text("scene4\n", encoding="utf-8")
    (src / SCENE_SPLITS_JSON).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "splits": {
                    "easy": ["scene1", "scene2"],
                    "medium": ["scene3"],
                    "hard": ["scene4"],
                },
            }
        ),
        encoding="utf-8",
    )
    return src


def test_stage_splits_copies_all_four_files(tmp_path: Path, splits_src: Path):
    staging = tmp_path / "stage"
    staged = stage_splits(staging, splits_src=splits_src)
    assert len(staged) == 4
    for name in (SCENE_SPLITS_JSON, *SPLIT_TIER_FILES):
        assert (staging / "splits" / name).is_file()


def test_stage_splits_repo_paths_are_forward_slashed(tmp_path: Path, splits_src: Path):
    staging = tmp_path / "stage"
    staged = stage_splits(staging, splits_src=splits_src)
    for s in staged:
        assert s.repo_path.startswith("splits/")
        assert "\\" not in s.repo_path
        assert s.bytes_ > 0


def test_stage_splits_default_src_is_benchmark_data_splits(tmp_path: Path):
    """Default splits_src should resolve under the installed package."""
    from rpx_benchmark.dataset_hub.staging import _default_splits_src

    default = _default_splits_src()
    assert default.parts[-2:] == ("data", "splits")
    assert default.parts[-3] == "benchmark"


def test_stage_splits_missing_src_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="splits source dir does not exist"):
        stage_splits(tmp_path / "stage", splits_src=tmp_path / "no_such_dir")


def test_stage_splits_missing_files_raises_when_required(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "easy.txt").write_text("scene1\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing splits files"):
        stage_splits(tmp_path / "stage", splits_src=src)


def test_stage_splits_allow_missing_yields_partial_set(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "easy.txt").write_text("scene1\n", encoding="utf-8")
    staged = stage_splits(tmp_path / "stage", splits_src=src, require_all=False)
    assert len(staged) == 1
    assert staged[0].repo_path == "splits/easy.txt"


def test_stage_splits_refuses_overwrite_by_default(tmp_path: Path, splits_src: Path):
    staging = tmp_path / "stage"
    stage_splits(staging, splits_src=splits_src)
    with pytest.raises(DatasetError, match="refusing to overwrite"):
        stage_splits(staging, splits_src=splits_src)


def test_stage_splits_overwrite_flag_allows_re_stage(tmp_path: Path, splits_src: Path):
    staging = tmp_path / "stage"
    stage_splits(staging, splits_src=splits_src)
    # Should not raise.
    stage_splits(staging, splits_src=splits_src, overwrite=True)


def test_load_scene_splits_normalises_canonical_shape(tmp_path: Path, splits_src: Path):
    staging = tmp_path / "stage"
    stage_splits(staging, splits_src=splits_src)
    flat = load_scene_splits(staging)
    assert flat == {"scene1": "easy", "scene2": "easy", "scene3": "medium", "scene4": "hard"}


def test_load_scene_splits_accepts_flat_shape(tmp_path: Path):
    staging = tmp_path / "stage"
    (staging / "splits").mkdir(parents=True)
    (staging / "splits" / SCENE_SPLITS_JSON).write_text(
        json.dumps({"sceneA": "easy", "sceneB": "hard"}),
        encoding="utf-8",
    )
    flat = load_scene_splits(staging)
    assert flat == {"sceneA": "easy", "sceneB": "hard"}


def test_load_scene_splits_missing_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_scene_splits(tmp_path / "empty")


def test_stage_paths_lands_files_at_specified_subdir(tmp_path: Path):
    src = tmp_path / "extras"
    src.mkdir()
    (src / "doc.md").write_text("docs", encoding="utf-8")
    (src / "rpx_croissant.json").write_text("{}", encoding="utf-8")

    staging = tmp_path / "stage"
    staged = stage_paths(
        [src / "doc.md", src / "rpx_croissant.json"],
        staging_root=staging,
        repo_subdir="",
    )
    assert (staging / "doc.md").is_file()
    assert (staging / "rpx_croissant.json").is_file()
    assert {s.repo_path for s in staged} == {"doc.md", "rpx_croissant.json"}


def test_stage_paths_nests_under_repo_subdir(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "preview1.jpg").write_bytes(b"\xff\xd8\xff")

    staging = tmp_path / "stage"
    staged = stage_paths([src / "preview1.jpg"], staging, repo_subdir="preview")
    assert (staging / "preview" / "preview1.jpg").is_file()
    assert staged[0].repo_path == "preview/preview1.jpg"


def test_stage_paths_missing_src_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="source file does not exist"):
        stage_paths([tmp_path / "nope.txt"], tmp_path / "stage")

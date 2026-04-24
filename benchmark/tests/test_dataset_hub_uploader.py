"""Tests for ``rpx_benchmark.dataset_hub.uploader``.

These tests deliberately avoid the network. The uploader's network paths
are exercised only through ``--dry-run`` (which short-circuits before any
HTTP calls) and through monkeypatched ``HfApi`` doubles.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.dataset_hub.uploader import (
    DEFAULT_IGNORE_PATTERNS,
    UploadPlan,
    upload_paths,
    upload_staging,
)
from rpx_benchmark.exceptions import ConfigError


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    src = generate_mock(tmp_path / "src", MockSpec(
        multi_object_scenes=1, single_object_scenes=1,
        phases_per_multi=1, frames_per_phase=2,
    ))
    out = tmp_path / "stage"
    scan = scan_capture_root(src)
    pack_capture_tree(PackPlan(src_root=src, staging_root=out), scan)
    return out


def test_dry_run_returns_file_count_without_network(staging: Path):
    plan = UploadPlan(staging_root=staging, repo_id="acme/RPX",
                       dry_run=True, create_if_missing=False)
    res = upload_staging(plan)
    assert res.repo_id == "acme/RPX"
    assert res.files_planned > 0
    assert res.bytes_planned > 0
    assert res.commit_url is None


def test_dry_run_respects_ignore_patterns(staging: Path):
    # Drop a junk file that should be ignored.
    (staging / ".DS_Store").write_bytes(b"junk")
    (staging / "scratch.tmp").write_bytes(b"also junk")

    plan = UploadPlan(
        staging_root=staging, repo_id="acme/RPX",
        dry_run=True, create_if_missing=False,
        ignore_patterns=DEFAULT_IGNORE_PATTERNS,
    )
    res = upload_staging(plan)
    # Junk files should not contribute to the file count.
    flat = list(staging.rglob("*"))
    file_count = sum(1 for p in flat if p.is_file())
    assert res.files_planned == file_count - 2


def test_missing_staging_dir_raises_config_error(tmp_path: Path):
    plan = UploadPlan(staging_root=tmp_path / "nope", repo_id="acme/RPX",
                       dry_run=True, create_if_missing=False)
    with pytest.raises(ConfigError, match="staging dir does not exist"):
        upload_staging(plan)


def test_real_upload_calls_huggingface_api(staging: Path, monkeypatch):
    """Patch HfApi so we verify the call shape without hitting the network."""
    from rpx_benchmark.dataset_hub import uploader

    fake_api = MagicMock()
    fake_api.upload_large_folder = MagicMock(return_value=None)
    fake_api.create_repo = MagicMock()

    fake_module = MagicMock()
    fake_module.HfApi = MagicMock(return_value=fake_api)
    monkeypatch.setattr(uploader, "_hub", lambda: fake_module)

    plan = UploadPlan(staging_root=staging, repo_id="acme/RPX",
                       dry_run=False, create_if_missing=True,
                       use_large_folder=True)
    res = upload_staging(plan)

    assert res.repo_id == "acme/RPX"
    fake_api.create_repo.assert_called_once()
    fake_api.upload_large_folder.assert_called_once()
    kwargs = fake_api.upload_large_folder.call_args.kwargs
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["folder_path"] == str(staging)


def test_falls_back_to_upload_folder_when_large_folder_unavailable(
    staging: Path, monkeypatch,
):
    """Older huggingface_hub versions without upload_large_folder."""
    from rpx_benchmark.dataset_hub import uploader

    fake_api = MagicMock(spec=["upload_folder", "create_repo"])
    fake_api.upload_folder = MagicMock(return_value=MagicMock(commit_url="https://example/c"))
    fake_api.create_repo = MagicMock()

    fake_module = MagicMock()
    fake_module.HfApi = MagicMock(return_value=fake_api)
    monkeypatch.setattr(uploader, "_hub", lambda: fake_module)

    plan = UploadPlan(staging_root=staging, repo_id="acme/RPX",
                       dry_run=False, use_large_folder=True)
    res = upload_staging(plan)
    fake_api.upload_folder.assert_called_once()
    assert res.commit_url == "https://example/c"


def test_upload_paths_builds_allow_patterns(staging: Path, monkeypatch):
    from rpx_benchmark.dataset_hub import uploader

    fake_api = MagicMock(spec=["upload_folder", "create_repo"])
    fake_api.upload_folder = MagicMock(return_value=MagicMock(commit_url=None))
    fake_api.create_repo = MagicMock()
    fake_module = MagicMock()
    fake_module.HfApi = MagicMock(return_value=fake_api)
    monkeypatch.setattr(uploader, "_hub", lambda: fake_module)

    one_tar = next(staging.rglob("*.tar"))
    res = upload_paths([one_tar], staging_root=staging, repo_id="acme/RPX")
    assert res.files_planned == 1

    kwargs = fake_api.upload_folder.call_args.kwargs
    assert "allow_patterns" in kwargs
    assert one_tar.relative_to(staging).as_posix() in kwargs["allow_patterns"]

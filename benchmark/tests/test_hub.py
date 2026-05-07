"""Unit tests for :mod:`rpx_benchmark.hub`.

Focuses on the pure-Python helpers and error wrapping. Actual
HuggingFace network calls are monkey-patched so the tests run offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpx_benchmark import hub
from rpx_benchmark.api import Difficulty, TaskType
from rpx_benchmark.exceptions import DownloadError, ManifestError

# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_manifest_repo_path_with_enums_and_strings():
    assert (
        hub._manifest_repo_path(TaskType.MONOCULAR_DEPTH, Difficulty.HARD)
        == "manifests/monocular_depth/hard.json"
    )
    assert (
        hub._manifest_repo_path("monocular_depth", "easy") == "manifests/monocular_depth/easy.json"
    )


def test_modalities_for_task_type():
    mods = hub._modalities_for(TaskType.MONOCULAR_DEPTH)
    assert "rgb/*" in mods
    assert "depth/*" in mods


def test_modalities_for_alias_string():
    mods = hub._modalities_for("qa_spatial")
    assert "rgb/*" in mods
    assert "spatial_qa.json" in mods


def test_extract_scene_phase_pairs_from_scenes_field():
    manifest = {
        "scenes": [
            {"scene": "scene_000", "phase": "0"},
            {"scene": "scene_001", "phase": "1"},
            {"scene": "scene_000", "phase": "0"},  # duplicate is deduped
        ]
    }
    pairs = hub._extract_scene_phase_pairs(manifest)
    assert pairs == {("scene_000", "0"), ("scene_001", "1")}


def test_extract_scene_phase_pairs_from_sample_paths():
    manifest = {
        "samples": [
            {"rgb": "scenes/scene_042/2/rgb/0001.png"},
            {"depth": "scenes/scene_007/0/depth/0005.png"},
        ]
    }
    pairs = hub._extract_scene_phase_pairs(manifest)
    assert pairs == {("scene_042", "2"), ("scene_007", "0")}


def test_build_allow_patterns_expands_modalities_per_pair():
    pats = hub._build_allow_patterns(
        modalities=["rgb/*", "depth/*"],
        scene_phase_pairs=[("scene_000", "0"), ("scene_001", "1")],
    )
    assert "scenes/scene_000/0/rgb/*" in pats
    assert "scenes/scene_000/0/depth/*" in pats
    assert "scenes/scene_001/1/rgb/*" in pats
    assert "scenes/scene_001/1/depth/*" in pats
    assert len(pats) == 4


# --------------------------------------------------------------------------- #
# Error wrapping
# --------------------------------------------------------------------------- #


def test_fetch_manifest_wraps_hf_exception_as_download_error(monkeypatch):
    """hf_hub_download failures must surface as DownloadError."""

    class _FakeHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            raise RuntimeError("simulated network blowup")

    monkeypatch.setattr(hub, "_hub", lambda: _FakeHub)

    with pytest.raises(DownloadError, match="simulated network blowup"):
        hub.fetch_manifest(TaskType.MONOCULAR_DEPTH, Difficulty.HARD)


def test_fetch_manifest_bad_json_raises_manifest_error(
    tmp_path: Path,
    monkeypatch,
):
    """Manifest file that exists but is garbage JSON → ManifestError."""
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{not valid")

    class _FakeHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            return str(bogus)

    monkeypatch.setattr(hub, "_hub", lambda: _FakeHub)
    with pytest.raises(ManifestError, match="not valid JSON"):
        hub.fetch_manifest(TaskType.MONOCULAR_DEPTH, Difficulty.HARD)


def test_download_split_raises_when_manifest_has_no_scenes(
    tmp_path: Path,
    monkeypatch,
):
    """Well-formed manifest that references no scenes → ManifestError."""
    empty_manifest = tmp_path / "empty.json"
    empty_manifest.write_text(json.dumps({"samples": []}))

    class _FakeHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            return str(empty_manifest)

        @staticmethod
        def snapshot_download(**kwargs):
            raise AssertionError("snapshot_download should not have been called")

    monkeypatch.setattr(hub, "_hub", lambda: _FakeHub)
    with pytest.raises(ManifestError, match="no scenes"):
        hub.download_split(TaskType.MONOCULAR_DEPTH, Difficulty.HARD)


def test_download_split_wraps_snapshot_download_failure(
    tmp_path: Path,
    monkeypatch,
):
    good_manifest = tmp_path / "good.json"
    good_manifest.write_text(
        json.dumps(
            {
                "scenes": [{"scene": "scene_000", "phase": "0"}],
                "samples": [],
            }
        )
    )

    class _FakeHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            return str(good_manifest)

        @staticmethod
        def snapshot_download(**kwargs):
            raise PermissionError("simulated 403")

    monkeypatch.setattr(hub, "_hub", lambda: _FakeHub)
    with pytest.raises(DownloadError, match="snapshot_download failed"):
        hub.download_split(TaskType.MONOCULAR_DEPTH, Difficulty.HARD)

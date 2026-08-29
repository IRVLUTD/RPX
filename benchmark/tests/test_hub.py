"""Unit tests for :mod:`rpx_benchmark.hub`.

Focuses on the pure-Python helpers and error wrapping. Actual
HuggingFace network calls are monkey-patched so the tests run offline.
"""

from __future__ import annotations

import io
import json
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rpx_benchmark import hub
from rpx_benchmark.api import Difficulty, TaskType
from rpx_benchmark.exceptions import DownloadError, ManifestError

# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_snapshot_extraction_is_serialized(tmp_path: Path, monkeypatch) -> None:
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def fake_extract(root: Path) -> tuple[int, int]:
        nonlocal active, maximum_active
        assert root == tmp_path
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return 0, 0

    monkeypatch.setattr(hub, "_extract_snapshot_tars_unlocked", fake_extract)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(hub._extract_snapshot_tars, [tmp_path] * 3))

    assert results == [(0, 0)] * 3
    assert maximum_active == 1


def test_snapshot_extraction_supports_ego_capture_directory(tmp_path: Path) -> None:
    """Ego shards use ``scenes/<scene>/ego`` rather than a numeric phase."""
    tar_path = tmp_path / "scenes" / "scene004" / "ego" / "rgb.tar"
    tar_path.parent.mkdir(parents=True)
    payload = b"valid-webp-placeholder"
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo("rgb/00000.webp")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    n_new, n_skipped = hub._extract_snapshot_tars(tmp_path)

    extracted = tmp_path / "extracted/scenes/scene004/ego/rgb/00000.webp"
    assert n_new == 1
    assert n_skipped == 0
    assert extracted.read_bytes() == payload


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


def test_video_depth_uses_rgb_depth_and_pose_modalities():
    assert hub._modalities_for(TaskType.VIDEO_DEPTH) == [
        "rgb/*",
        "depth/*",
        "pose/*",
    ]


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


def test_extract_scene_phase_pairs_from_video_entries():
    manifest = {
        "samples": [
            {
                "scene_id": "scene_004",
                "phase": 0,
                "frame_filenames": ["extracted/scenes/scene_004/0/rgb/00000.webp"],
            }
        ]
    }
    assert hub._extract_scene_phase_pairs(manifest) == {("scene_004", "0")}


def test_build_allow_patterns_expands_modalities_per_pair():
    pats = hub._build_allow_patterns(
        modalities=["rgb/*", "depth/*"],
        scene_phase_pairs=[("scene_000", "0"), ("scene_001", "1")],
    )
    assert "scenes/scene_000/0/rgb/*" in pats
    assert "scenes/scene_000/0/depth/*" in pats
    assert "scenes/scene_001/1/rgb/*" in pats
    assert "scenes/scene_001/1/depth/*" in pats
    assert "scenes/scene_000/0/rgb.tar" in pats
    assert "scenes/scene_000/0/depth.tar" in pats
    assert len(pats) == 8


def test_video_allow_patterns_include_lossless_pose_tar():
    pats = hub._build_allow_patterns(
        modalities=hub._modalities_for(TaskType.VIDEO_DEPTH),
        scene_phase_pairs=[("scene_004", "0")],
    )
    assert "scenes/scene_004/0/rgb.tar" in pats
    assert "scenes/scene_004/0/depth.tar" in pats
    assert "scenes/scene_004/0/labels/cam_pose/v1.tar" in pats


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


def test_download_split_limits_pairs_before_bulk_fetch(tmp_path, monkeypatch):
    manifest = {
        "task": "monocular_depth",
        "samples": [
            {
                "id": "a",
                "scene_id": "scene_a",
                "phase": 0,
                "rgb": "extracted/scenes/scene_a/0/rgb/00000.webp",
                "depth": "extracted/scenes/scene_a/0/depth/00000.png",
            },
            {
                "id": "b",
                "scene_id": "scene_b",
                "phase": 0,
                "rgb": "extracted/scenes/scene_b/0/rgb/00000.webp",
                "depth": "extracted/scenes/scene_b/0/depth/00000.png",
            },
        ],
    }
    captured = {}

    class _FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return str(tmp_path / "snapshot")

    (tmp_path / "snapshot").mkdir()
    monkeypatch.setattr(hub, "_hub", lambda: _FakeHub)
    monkeypatch.setattr(hub, "fetch_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(hub, "_extract_snapshot_tars", lambda root: (0, 0))
    monkeypatch.setenv("RPX_CACHE_DIR", str(tmp_path / "resolved"))

    resolved = hub.download_split(
        TaskType.MONOCULAR_DEPTH,
        Difficulty.EASY,
        max_samples=1,
    )
    allow = captured["allow_patterns"]
    assert "scenes/scene_a/0/rgb.tar" in allow
    assert not any("scene_b" in pattern for pattern in allow)
    assert len(json.loads(resolved.read_text())["samples"]) == 1


def test_ego_manifest_uses_physical_ego_directory() -> None:
    manifest = {
        "samples": [
            {
                "scene_id": "scene004",
                "phase": 0,
                "rgb": "extracted/scenes/scene004/ego/rgb/00000.webp",
                "mask": "extracted/scenes/scene004/ego/sam2/masks/00000.png",
            }
        ]
    }

    assert hub._extract_scene_phase_pairs(manifest) == {("scene004", "ego")}


def test_download_split_accepts_alternate_manifest_name(tmp_path, monkeypatch):
    manifest = {
        "task": "object_tracking",
        "samples": [
            {
                "scene_id": "scene004",
                "phase": 0,
                "rgb": "extracted/scenes/scene004/ego/rgb/00000.webp",
                "mask": "extracted/scenes/scene004/ego/sam2/masks/00000.png",
            }
        ],
    }
    captured = {}

    class _FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return str(tmp_path / "snapshot")

    (tmp_path / "snapshot").mkdir()
    monkeypatch.setattr(hub, "_hub", lambda: _FakeHub)
    monkeypatch.setattr(hub, "fetch_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(hub, "_extract_snapshot_tars", lambda root: (0, 0))
    monkeypatch.setenv("RPX_CACHE_DIR", str(tmp_path / "resolved"))

    resolved = hub.download_split(
        TaskType.OBJECT_TRACKING,
        Difficulty.EASY,
        manifest_name="ego_object_tracking",
    )

    assert "manifests/ego_object_tracking/easy.json" in captured["allow_patterns"]
    assert "scenes/scene004/ego/rgb.tar" in captured["allow_patterns"]
    assert "scenes/scene004/ego/labels/masks/v1.tar" in captured["allow_patterns"]
    assert resolved.parent.name == "ego_object_tracking"

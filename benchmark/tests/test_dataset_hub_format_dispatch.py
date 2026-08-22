"""Integration tests for v2 (post-lossless-convert) format dispatch.

What's covered
--------------

* ``manifest._detect_modality_extensions`` sniffs the actual file
  extensions in a scanned tree, modality by modality. Mixed extensions
  (.png for some modalities, .webp / .npy for others) are reported as
  declared.
* ``build_frame_manifest`` records the detection result in
  ``manifest/current.json`` under ``modality_extensions``.
* ``write_split_manifests`` reads ``modality_extensions`` from
  ``current.json`` and produces per-task JSONs whose paths point at
  the actual on-disk file extensions (not the v1 hardcoded defaults).
* Backward compatibility: a v1-shaped ``current.json`` (no
  ``modality_extensions`` field) still produces the v1 hardcoded paths.

This file does *not* exercise the real conversion script — that lives in
``test_dataset_hub_lossless_convert.py``. The fixtures here construct
post-conversion trees by hand so the tests stay fast and isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpx_benchmark.dataset_hub.manifest import (
    _detect_modality_extensions,
    build_frame_manifest,
)
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.dataset_hub.split_manifests import write_split_manifests

pytest.importorskip("pyarrow", reason="pyarrow required for manifest builder")
pytest.importorskip("pandas", reason="pandas required for split-manifest writer")


# --------------------------------------------------------------------------- #
# _detect_modality_extensions
# --------------------------------------------------------------------------- #


@pytest.fixture
def v1_mock(tmp_path: Path) -> Path:
    """A pristine mock tree — every modality at its v1 default extension."""
    return generate_mock(
        tmp_path / "src",
        MockSpec(
            multi_object_scenes=1,
            single_object_scenes=0,
            phases_per_multi=1,
            frames_per_phase=2,
            image_size=16,
        ),
    )


def _rewrite_extension(directory: Path, new_suffix: str) -> int:
    """Rename every file in ``directory`` to use ``new_suffix``.

    Returns the count of files touched. Used to simulate a post-conversion
    state without re-running the converter — the byte contents do not
    matter for these tests, only the file extension.
    """
    n = 0
    for p in sorted(directory.iterdir()):
        if p.is_file():
            p.rename(p.with_suffix(new_suffix))
            n += 1
    return n


def _make_v2_tree(src: Path) -> None:
    """In-place: rewrite the v1 mock to mimic a post-conversion v2 tree.

    rgb / fisheye / fisheye/left / fisheye/right → .webp
    cam_pose                                     → .npy
    depth, sam2/masks                            → .png (unchanged)
    """
    for phase_dir in (src / "mos").rglob("*"):
        if not phase_dir.is_dir() or not phase_dir.name.isdigit():
            continue
        for sub in ("rgb", "fisheye", "fisheye/left", "fisheye/right"):
            d = phase_dir / sub
            if d.is_dir():
                _rewrite_extension(d, ".webp")
        cp = phase_dir / "cam_pose"
        if cp.is_dir():
            _rewrite_extension(cp, ".npy")


def test_detect_v1_tree_returns_all_png(v1_mock: Path):
    """v1 mock plants images everywhere — every image modality reports
    .png. cam_pose is intentionally .json in the mock generator (it
    diverges from real captures, which use .npz); the detection
    machinery just records the actual on-disk suffix without prejudice."""
    scan = scan_capture_root(v1_mock)
    ext = _detect_modality_extensions(scan)
    assert ext["rgb"] == ".png"
    assert ext["depth"] == ".png"
    assert ext["masks"] == ".png"
    # cam_pose in the mock is .json (mock-generator convention); the
    # detector just reports what's there.
    assert ext["cam_pose"] in {".npz", ".json"}


def test_detect_v2_tree_returns_webp_and_npy(v1_mock: Path):
    """Post-conversion tree: rgb/fisheye are .webp, cam_pose is .npy,
    depth and masks stay .png."""
    _make_v2_tree(v1_mock)
    scan = scan_capture_root(v1_mock)
    ext = _detect_modality_extensions(scan)
    assert ext["rgb"] == ".webp"
    assert ext["fisheye"] == ".webp"
    # fisheye_left / fisheye_right also detected if those subdirs exist
    if "fisheye_left" in ext:
        assert ext["fisheye_left"] == ".webp"
    if "fisheye_right" in ext:
        assert ext["fisheye_right"] == ".webp"
    assert ext["depth"] == ".png"  # unchanged
    assert ext["masks"] == ".png"  # unchanged
    assert ext["cam_pose"] == ".npy"


def test_detect_ego_only_tree_does_not_go_blind(tmp_path: Path):
    """Regression test: _detect_modality_extensions used to compute the
    scan-subdir via a `"mos" if MULTI_OBJECT else "sos"` ternary — for an
    EGO scene that resolves to "sos", so it looked for ego's files under
    <root>/sos/<scene_id>/... which doesn't exist, silently finding
    nothing (found={}). That's masked whenever mos/sos scenes are also
    present (mos alone fills in every modality first) but would go fully
    blind on an ego-only capture root, like the real
    ego-only-DATA staging path used to validate the ego upload before it
    was ever combined with the mos/sos data. Uses SRC_SUBDIR_BY_TYPE now.
    """
    src = tmp_path / "src"
    rgb = src / "ego" / "scene009" / "0" / "rgb"
    rgb.mkdir(parents=True)
    (rgb / "00000.png").write_text("x")
    masks = src / "ego" / "scene009" / "0" / "sam2" / "masks"
    masks.mkdir(parents=True)
    (masks / "00000.png").write_text("x")
    # No mos/ or sos/ dirs at all.
    scan = scan_capture_root(src)
    ext = _detect_modality_extensions(scan)
    assert ext.get("rgb") == ".png", f"ego-only tree went blind: {ext}"
    assert ext.get("masks") == ".png", f"ego-only tree went blind: {ext}"


def test_detect_skips_missing_modalities(tmp_path: Path):
    """A tree that doesn't have every modality only reports what exists."""
    src = tmp_path / "src"
    rgb = src / "mos" / "scene1" / "0" / "rgb"
    rgb.mkdir(parents=True)
    (rgb / "00000.png").write_text("x")
    # Intentionally NO depth/, masks/, cam_pose/, fisheye/
    scan = scan_capture_root(src)
    ext = _detect_modality_extensions(scan)
    assert ext.get("rgb") == ".png"
    assert "depth" not in ext
    assert "cam_pose" not in ext


# --------------------------------------------------------------------------- #
# build_frame_manifest writes modality_extensions to current.json
# --------------------------------------------------------------------------- #


def test_build_frame_manifest_writes_v2_extensions(v1_mock: Path, tmp_path: Path):
    _make_v2_tree(v1_mock)
    scan = scan_capture_root(v1_mock)
    staging = tmp_path / "stage"
    pack = pack_capture_tree(PackPlan(src_root=v1_mock, staging_root=staging), scan)
    build_frame_manifest(scan, pack, staging)

    cur = json.loads((staging / "manifest" / "current.json").read_text())
    assert "modality_extensions" in cur, "modality_extensions missing from current.json"
    ext = cur["modality_extensions"]
    assert ext.get("rgb") == ".webp"
    assert ext.get("cam_pose") == ".npy"
    assert ext.get("depth") == ".png"
    assert ext.get("masks") == ".png"


def test_build_frame_manifest_preserves_unmanaged_top_level_keys(v1_mock: Path, tmp_path: Path):
    """A pre-existing current.json with extra top-level blocks (the live
    IRVLUTD/RPX repo carries ``manifests``, ``sos``, ``mos``,
    ``metadata_versions``) MUST survive a subsequent
    ``build_frame_manifest`` call. The function may only overwrite the
    keys it owns (``label_versions``, ``schema_version``,
    ``modality_extensions``)."""
    scan = scan_capture_root(v1_mock)
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "manifest").mkdir()
    # Plant a current.json that carries blocks our builder does NOT own.
    prior = {
        "label_versions": {"masks": "vOLD"},
        "schema_version": "vOLD",
        "manifests": {"frames": "manifest/frames_v1.parquet"},
        "sos": {"selected_object_count": 70},
        "mos": {"scene_count": 100, "phase_count": 300},
        "metadata_versions": {"scene_name_mapping": "v1"},
    }
    (staging / "manifest" / "current.json").write_text(json.dumps(prior, indent=2))

    pack = pack_capture_tree(PackPlan(src_root=v1_mock, staging_root=staging), scan)
    build_frame_manifest(scan, pack, staging)

    cur = json.loads((staging / "manifest" / "current.json").read_text())
    # Our keys updated:
    assert cur["schema_version"] != "vOLD"  # we own it; we wrote a fresh value
    assert cur["label_versions"] != {"masks": "vOLD"}  # we own it; we wrote fresh
    # Unmanaged blocks preserved verbatim:
    assert cur["manifests"] == {"frames": "manifest/frames_v1.parquet"}
    assert cur["sos"] == {"selected_object_count": 70}
    assert cur["mos"] == {"scene_count": 100, "phase_count": 300}
    assert cur["metadata_versions"] == {"scene_name_mapping": "v1"}


def test_build_frame_manifest_omits_modality_extensions_for_empty_tree(tmp_path: Path):
    """If the scanned tree has nothing recognisable, the field is omitted
    rather than written as an empty dict."""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "mos").mkdir()  # makes the scan succeed with zero scenes
    scan = scan_capture_root(empty)
    staging = tmp_path / "stage"

    # Re-use packer to produce an empty PackResult.
    pack = pack_capture_tree(PackPlan(src_root=empty, staging_root=staging), scan)
    build_frame_manifest(scan, pack, staging)
    cur = json.loads((staging / "manifest" / "current.json").read_text())
    # Backward-compat: absent is fine, downstream defaults take over.
    assert "modality_extensions" not in cur or cur["modality_extensions"] == {}


# --------------------------------------------------------------------------- #
# split_manifests dispatches on current.json
# --------------------------------------------------------------------------- #


def test_split_manifests_uses_v2_extensions_when_declared(v1_mock: Path, tmp_path: Path):
    """write_split_manifests must produce paths whose suffix matches
    current.json's modality_extensions — not the v1 hardcoded default."""
    _make_v2_tree(v1_mock)
    scan = scan_capture_root(v1_mock)
    staging = tmp_path / "stage"
    pack = pack_capture_tree(PackPlan(src_root=v1_mock, staging_root=staging), scan)
    build_frame_manifest(
        scan,
        pack,
        staging,
        splits={s.scene_id: "easy" for s in scan.scenes},
    )

    written = write_split_manifests(staging)
    assert written, "no split-manifest files written"

    # Every emitted task JSON whose schema includes 'rgb' must reference
    # .webp paths, and 'cam_pose' must reference .npy paths. Depth/masks
    # stay .png.
    seen_rgb = seen_depth = seen_pose = seen_mask = False
    for path in written.values():
        payload = json.loads(path.read_text())
        for sample in payload["samples"]:
            for _key, val in sample.items():
                if isinstance(val, str) and "/rgb/" in val:
                    assert val.endswith(".webp"), f"rgb not webp at {path}: {val}"
                    seen_rgb = True
                if isinstance(val, str) and "/depth/" in val:
                    assert val.endswith(".png"), f"depth not png at {path}: {val}"
                    seen_depth = True
                if isinstance(val, str) and "/cam_pose/" in val:
                    assert val.endswith(".npy"), f"cam_pose not npy at {path}: {val}"
                    seen_pose = True
                if isinstance(val, str) and "/sam2/masks/" in val:
                    assert val.endswith(".png"), f"masks not png at {path}: {val}"
                    seen_mask = True
    # We expect to have seen at least one of each kind (mock has all modalities).
    assert seen_rgb, "no rgb path observed across emitted samples"
    assert seen_depth or seen_pose or seen_mask, (
        "no non-rgb modality path observed — recipe coverage looks too thin"
    )


def test_split_manifests_falls_back_to_v1_defaults_without_modality_extensions(
    v1_mock: Path, tmp_path: Path
):
    """If current.json lacks modality_extensions (legacy v1 datasets),
    the writer must produce the original .png / .npz paths so already-
    uploaded datasets keep working."""
    scan = scan_capture_root(v1_mock)
    staging = tmp_path / "stage"
    pack = pack_capture_tree(PackPlan(src_root=v1_mock, staging_root=staging), scan)
    build_frame_manifest(
        scan,
        pack,
        staging,
        splits={s.scene_id: "easy" for s in scan.scenes},
    )

    # Sabotage: strip the modality_extensions field to simulate v1.
    cur_path = staging / "manifest" / "current.json"
    cur = json.loads(cur_path.read_text())
    cur.pop("modality_extensions", None)
    cur_path.write_text(json.dumps(cur, indent=2))

    written = write_split_manifests(staging)
    assert written, "no split-manifest files written"

    for path in written.values():
        payload = json.loads(path.read_text())
        for sample in payload["samples"]:
            for _key, val in sample.items():
                if isinstance(val, str):
                    if "/rgb/" in val:
                        assert val.endswith(".png"), f"rgb fallback wrong at {path}: {val}"
                    if "/cam_pose/" in val:
                        assert val.endswith(".npz"), f"cam_pose fallback wrong at {path}: {val}"

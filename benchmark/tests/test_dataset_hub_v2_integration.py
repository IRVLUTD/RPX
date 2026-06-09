"""End-to-end integration test for the v2 (lossless-convert) format.

The test exercises the **full HF publication pipeline** against a
synthetic capture tree that has been converted to v2 format. It is the
strongest available check that the upload + download flow round-trips
correctly when the on-disk modality formats are no longer the v1
defaults.

Pipeline under test
-------------------

1. ``generate_mock`` → synthetic v1 capture tree (rgb/depth/fisheye/
   masks as ``.png``, cam_pose as ``.json``).
2. **Plant a per-frame ``.npz``** under ``cam_pose/`` so the lossless
   converter has something realistic to consolidate (the mock writes
   JSON, but the converter targets the .npz format real captures use).
3. **Run** ``lossless_convert.convert_capture_tree`` against the v1
   tree to produce a v2 tree (rgb/fisheye → ``.webp``, cam_pose →
   ``.npy``, depth/masks → re-encoded ``.png``).
4. **Pack** the v2 tree into staging tars via ``pack_capture_tree``.
5. **Build the frames Parquet + ``current.json``** — must record
   ``modality_extensions`` with ``.webp`` and ``.npy``.
6. **Write the per-task per-split JSONs** — must consult
   ``modality_extensions`` and emit paths whose suffix matches the v2
   tree.
7. **Extract** the tar shards via ``_extract_snapshot_tars`` — the
   same code path a downloaded HF dataset takes locally.
8. **Open** every emitted manifest through ``RPXDataset.from_manifest``
   and pull a sample. Every modality path must resolve to a real file
   under ``extracted/...``.
9. **Bit-identity assertion**: for the same (scene, phase, frame), the
   v2-path loader and a hand-rolled decode of the source v1 file
   must produce ``numpy.array_equal`` arrays. The format on disk
   changed; the array the model sees did not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="pyarrow required for manifest builder")
pytest.importorskip("pandas", reason="pandas required for split-manifest writer")
pytest.importorskip("PIL.Image", reason="Pillow required for image round-trip")

import numpy as np
from PIL import Image

from rpx_benchmark.dataset_hub.cli import _rehydrate_pack_result
from rpx_benchmark.dataset_hub.lossless_convert import (
    ConvertSpec,
    convert_capture_tree,
)
from rpx_benchmark.dataset_hub.manifest import build_frame_manifest
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.scanner import scan_capture_root
from rpx_benchmark.dataset_hub.split_manifests import write_split_manifests
from rpx_benchmark.hub import _extract_snapshot_tars
from rpx_benchmark.loader import RPXDataset

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _plant_npz_cam_pose(src_root: Path) -> None:
    """The mock generator writes cam_pose as ``.json``. Real captures use
    per-frame ``.npz`` with ``position`` and ``orientation`` keys, which
    is what the lossless converter consumes. Replace the mock's JSON
    files with valid .npz files so the conversion path is realistic."""
    rng = np.random.default_rng(0)
    for cp_dir in src_root.rglob("cam_pose"):
        if not cp_dir.is_dir():
            continue
        # Delete the mock's JSON files; replace with .npz of matching stem.
        for j in sorted(cp_dir.glob("*.json")):
            stem = j.stem
            j.unlink()
            pos = rng.standard_normal(3).astype(np.float64)
            quat = rng.standard_normal(4).astype(np.float64)
            quat /= np.linalg.norm(quat)  # unit quaternion
            np.savez(cp_dir / f"{stem}.npz", position=pos, orientation=quat)


@pytest.fixture
def v2_pipeline(tmp_path: Path):
    """Run the full pipeline once and yield every artefact downstream
    tests need to inspect. Heavy fixture (one converter run + one packer
    run + one extractor run), so we share it across the assertion-only
    tests below."""
    # 1. v1 mock tree
    v1 = tmp_path / "v1"
    generate_mock(
        v1,
        MockSpec(
            multi_object_scenes=2,
            single_object_scenes=0,
            phases_per_multi=1,
            frames_per_phase=3,
            image_size=16,
        ),
    )
    _plant_npz_cam_pose(v1)

    # 2. Convert to v2 (rgb/fisheye → .webp, cam_pose → .npy)
    v2 = tmp_path / "v2"
    res = convert_capture_tree(ConvertSpec(src_root=v1, out_root=v2, workers=1, verify=True))
    assert res.failures == []

    # 3. Pack v2 into staging
    staging = tmp_path / "staging"
    scan = scan_capture_root(v2)
    pack_capture_tree(
        PackPlan(
            src_root=v2,
            staging_root=staging,
            label_version="v1",
            overwrite=True,
        ),
        scan,
    )

    # 4. Manifest (writes modality_extensions to current.json)
    pack = _rehydrate_pack_result(scan, staging)
    splits = {s.scene_id: ("easy" if i % 2 == 0 else "hard") for i, s in enumerate(scan.scenes)}
    build_frame_manifest(scan, pack, staging, splits=splits)

    # 5. Per-task per-split JSONs
    written = write_split_manifests(staging)
    assert written, "writer produced no manifests"

    # 6. Extract tars (the local mirror of what HF download produces)
    n_new, _ = _extract_snapshot_tars(staging)
    assert n_new > 0, "extraction produced zero files"

    return {
        "v1": v1,
        "v2": v2,
        "staging": staging,
        "written": written,
    }


# --------------------------------------------------------------------------- #
# Test 1 — current.json declares the right v2 extensions
# --------------------------------------------------------------------------- #


def test_current_json_declares_v2_modality_extensions(v2_pipeline):
    staging: Path = v2_pipeline["staging"]
    cur = json.loads((staging / "manifest" / "current.json").read_text())
    assert "modality_extensions" in cur
    ext = cur["modality_extensions"]
    assert ext["rgb"] == ".webp"
    assert ext["cam_pose"] == ".npy"
    assert ext["depth"] == ".png"
    assert ext["masks"] == ".png"


# --------------------------------------------------------------------------- #
# Test 2 — split manifests point at v2 file extensions, and those files
# exist after extraction
# --------------------------------------------------------------------------- #


def test_extracted_files_match_manifest_paths(v2_pipeline):
    staging: Path = v2_pipeline["staging"]
    written: dict = v2_pipeline["written"]

    # monocular_depth/easy is the simplest schema (rgb + depth) and
    # exercises both the WebP path and the still-PNG path.
    if ("monocular_depth", "easy") not in written:
        pytest.skip("no monocular_depth/easy in this mock")
    payload = json.loads(written[("monocular_depth", "easy")].read_text())

    rgb_count = depth_count = 0
    for sample in payload["samples"]:
        rgb = staging / sample["rgb"]
        depth = staging / sample["depth"]
        assert rgb.suffix == ".webp", f"rgb not .webp in manifest: {sample['rgb']}"
        assert depth.suffix == ".png", f"depth not .png in manifest: {sample['depth']}"
        assert rgb.is_file(), f"missing extracted rgb:   {rgb}"
        assert depth.is_file(), f"missing extracted depth: {depth}"
        rgb_count += 1
        depth_count += 1
    assert rgb_count > 0
    assert depth_count > 0


# --------------------------------------------------------------------------- #
# Test 3 — RPXDataset.from_manifest opens the v2 manifests and decodes
# every sample
# --------------------------------------------------------------------------- #


def _resolve_manifest_root(manifest_path: Path, root: Path) -> Path:
    """Inject `root` into the manifest (the writer leaves it null)."""
    payload = json.loads(manifest_path.read_text())
    payload["root"] = str(root)
    out = manifest_path.parent / f"{manifest_path.stem}_resolved.json"
    out.write_text(json.dumps(payload))
    return out


@pytest.mark.parametrize(
    "recipe_key,split",
    [
        ("monocular_depth", "easy"),
        ("segmentation", "easy"),
        ("rgbd_segmentation", "easy"),
        ("stereo_depth", "easy"),
    ],
)
def test_loader_opens_and_decodes_v2_samples(v2_pipeline, recipe_key, split):
    staging: Path = v2_pipeline["staging"]
    written: dict = v2_pipeline["written"]
    if (recipe_key, split) not in written:
        pytest.skip(f"no manifest for ({recipe_key}, {split}) in this mock")

    resolved = _resolve_manifest_root(written[(recipe_key, split)], staging)
    ds = RPXDataset.from_manifest(resolved, batch_size=1)
    assert len(ds) > 0, f"{recipe_key}/{split}: empty dataset"

    batch = next(iter(ds))
    assert batch
    sample = batch[0]
    # RGB came from a .webp, but the loader returns the same uint8 H×W×3.
    assert sample.rgb is not None and sample.rgb.size > 0
    assert sample.rgb.dtype == np.uint8
    assert sample.rgb.ndim == 3 and sample.rgb.shape[-1] == 3


# --------------------------------------------------------------------------- #
# Test 4 — bit-identity: v2-pipeline output equals v1-source decode
# --------------------------------------------------------------------------- #


def test_v2_loaded_rgb_is_bit_identical_to_v1_source(v2_pipeline):
    """For every (scene, phase, frame) loaded through the v2 pipeline,
    the uint8 RGB array must equal what we'd get by decoding the v1
    source PNG directly. Proves the format change is invisible.
    """
    v1: Path = v2_pipeline["v1"]
    staging: Path = v2_pipeline["staging"]
    written: dict = v2_pipeline["written"]
    if ("monocular_depth", "easy") not in written:
        pytest.skip("no monocular_depth/easy in this mock")

    resolved = _resolve_manifest_root(written[("monocular_depth", "easy")], staging)
    ds = RPXDataset.from_manifest(resolved, batch_size=1)

    checked = 0
    for batch in ds:
        for sample in batch:
            # Find the corresponding v1 source PNG. The sample's rgb path
            # is `extracted/scenes/<scene>/<phase>/rgb/<stem>.webp`;
            # the v1 source is `<v1>/mos/<scene>/<phase>/rgb/<stem>.png`.
            meta = sample.metadata
            scene = meta["scene_id"]
            phase = meta["phase_idx"]
            stem = meta["frame"]
            v1_rgb_path = v1 / "mos" / scene / str(phase) / "rgb" / f"{stem}.png"
            assert v1_rgb_path.is_file(), f"missing v1 source: {v1_rgb_path}"
            v1_rgb = np.asarray(Image.open(v1_rgb_path).convert("RGB"), dtype=np.uint8)
            assert np.array_equal(sample.rgb, v1_rgb), (
                f"bit-identity broken at {scene}/{phase}/{stem}: "
                f"v2 shape={sample.rgb.shape}, v1 shape={v1_rgb.shape}"
            )
            checked += 1
    assert checked > 0, "no frames compared — fixture too small?"


def test_v2_loaded_depth_is_bit_identical_to_v1_source(v2_pipeline):
    """Same property for 16-bit depth: v1 PNG and v2 PNG re-encoded at
    compress_level=9 must produce identical pixel arrays."""
    v1: Path = v2_pipeline["v1"]
    staging: Path = v2_pipeline["staging"]
    written: dict = v2_pipeline["written"]
    if ("monocular_depth", "easy") not in written:
        pytest.skip("no monocular_depth/easy in this mock")

    resolved = _resolve_manifest_root(written[("monocular_depth", "easy")], staging)
    ds = RPXDataset.from_manifest(resolved, batch_size=1)

    checked = 0
    for batch in ds:
        for sample in batch:
            meta = sample.metadata or {}
            scene = meta["scene_id"]
            phase = meta["phase_idx"]
            stem = meta["frame"]
            v1_depth_path = v1 / "mos" / scene / str(phase) / "depth" / f"{stem}.png"
            assert v1_depth_path.is_file(), f"missing v1 source: {v1_depth_path}"
            # For monocular_depth, depth GT is wrapped in DepthGroundTruth.
            assert sample.ground_truth is not None, "ground_truth missing for monocular_depth"
            v2_depth = sample.ground_truth.depth_map
            # _load_depth produces float32 metres (uint16 mm / 1000),
            # so reproduce that exactly to compare apples to apples.
            v1_depth_mm = np.array(Image.open(v1_depth_path), dtype=np.float32)
            v1_depth_m = v1_depth_mm / 1000.0
            v1_depth_m[v1_depth_mm == 0] = 0.0
            assert np.array_equal(v2_depth, v1_depth_m), (
                f"depth bit-identity broken at {scene}/{phase}/{stem}"
            )
            checked += 1
    assert checked > 0


def test_v2_loaded_pose_is_bit_identical_to_v1_npz(v2_pipeline):
    """The cam_pose .npz → .npy round-trip must produce the same 4×4
    SE(3) matrix the v1 npz-loaded pose would."""
    v1: Path = v2_pipeline["v1"]
    staging: Path = v2_pipeline["staging"]
    written: dict = v2_pipeline["written"]
    if ("rgbd_relative_pose", "easy") not in written:
        pytest.skip("no rgbd_relative_pose/easy manifest in this mock")

    resolved = _resolve_manifest_root(written[("rgbd_relative_pose", "easy")], staging)
    ds = RPXDataset.from_manifest(resolved, batch_size=1)
    # Build a v1-rooted dataset over the same manifest layout so we can
    # ask it to load the source .npz instead.
    v1_loader = RPXDataset(samples=[], task=ds.task, root=v1)

    checked = 0
    for batch in ds:
        for sample in batch:
            meta = sample.metadata or {}
            scene = meta.get("scene_id")
            phase = meta.get("phase_idx")
            stem = meta.get("frame") or meta.get("frame_a")
            if not (scene and phase is not None and stem):
                continue
            v1_pose_path = f"mos/{scene}/{phase}/cam_pose/{stem}.npz"
            v1_T = v1_loader._load_pose(v1_pose_path)
            v2_T = sample.camera_pose
            if v2_T is None:
                continue  # this task doesn't expose camera_pose
            assert np.array_equal(v1_T, v2_T), f"pose mismatch at {scene}/{phase}/{stem}"
            checked += 1
            if checked >= 3:
                return
    if checked == 0:
        pytest.skip("no pose-task samples reached this test — fixture too small?")


# --------------------------------------------------------------------------- #
# Test 5 — re-extraction is idempotent on v2 too (no duplicated work)
# --------------------------------------------------------------------------- #


def test_v2_re_extraction_is_idempotent(v2_pipeline):
    staging: Path = v2_pipeline["staging"]
    n_new, _ = _extract_snapshot_tars(staging)
    assert n_new == 0, f"second extraction wrote {n_new} files"

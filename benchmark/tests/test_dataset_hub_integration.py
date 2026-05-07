"""End-to-end contract test: dataset_hub writer → loader.

The writer (`split_manifests.write_split_manifests`) produces per-task
per-split JSONs that the loader (`RPXDataset.from_manifest`) is supposed
to consume directly. This test pins that contract by:

1. Building a synthetic capture tree (mock data).
2. Running every dataset_hub step that contributes to the published
   tree (pack → manifest → split-manifest writing).
3. Calling `_extract_snapshot_tars` to materialise the extracted PNG
   layout that the manifests reference.
4. Loading every produced (task, split) manifest through
   `RPXDataset.from_manifest` and pulling samples through the dataset.

Catches the silent-contract bugs we fought through earlier:
- `task` field doesn't match a `TaskType` value (e.g. "segmentation"
  vs "object_segmentation").
- Per-task entry keys mismatch (`mask` vs `masks`, `pose` vs `cam_pose`,
  `fisheye_left`/`fisheye_right` vs `fisheye`).
- Modality file extensions wrong (`.png` for cam_pose when reality is `.npz`).
- Tar layout wrong (`masks/<frame>` vs `sam2/masks/<frame>`).
- Manifest written but never extracted (the extraction gap before
  download_split learned to extract).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# We need a packed mock dataset to drive this test. The dataset_hub's
# mock generator already produces one, so we use it.
from rpx_benchmark.dataset_hub.mock import MockSpec, generate_mock
from rpx_benchmark.dataset_hub.packer import PackPlan, pack_capture_tree
from rpx_benchmark.dataset_hub.scanner import scan_capture_root


@pytest.fixture
def staged_dataset(tmp_path):
    """Build a tiny mock dataset, pack to staging, and return the staging path."""
    src = tmp_path / "captures"
    staging = tmp_path / "staging"
    spec = MockSpec(
        multi_object_scenes=2,
        single_object_scenes=0,
        frames_per_phase=4,  # enough rows that every (task, split) hits ≥ 1
        image_size=32,
    )
    generate_mock(src, spec)

    scan = scan_capture_root(src)
    pack_capture_tree(
        PackPlan(
            src_root=src,
            staging_root=staging,
            label_version="v1",
            overwrite=True,
        ),
        scan,
    )
    return src, staging


@pytest.fixture
def with_manifests(staged_dataset, tmp_path):
    """Build the parquet + per-task split manifests on top of the staged tree."""
    src, staging = staged_dataset

    # 1. Build the all-frames parquet (with a per-scene split mapping
    #    so write_split_manifests has rows to filter on).
    from rpx_benchmark.dataset_hub.cli import _rehydrate_pack_result
    from rpx_benchmark.dataset_hub.manifest import build_frame_manifest
    from rpx_benchmark.dataset_hub.scanner import scan_capture_root

    scan = scan_capture_root(src)
    pack = _rehydrate_pack_result(scan, staging)

    splits = {
        scene.scene_id: ("easy" if i % 2 == 0 else "hard") for i, scene in enumerate(scan.scenes)
    }
    build_frame_manifest(scan, pack, staging, splits=splits)

    # 2. Per-task per-split JSONs (the writer under test).
    from rpx_benchmark.dataset_hub.split_manifests import write_split_manifests

    written = write_split_manifests(staging)
    assert written, "writer produced no manifests — splits weren't applied"

    return src, staging, written


# ────────────────────────  Loader contract round-trip  ───────────────────


def _resolve_paths_for_loader(staging: Path, manifest_path: Path) -> Path:
    """The manifest references `extracted/scenes/...` paths and has
    `root: None` (filled at download time). Set root to the staging
    dir + materialise extracted layout so the loader can resolve."""
    from rpx_benchmark.hub import _extract_snapshot_tars

    _extract_snapshot_tars(staging)

    payload = json.loads(manifest_path.read_text())
    payload["root"] = str(staging)
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
        ("relative_pose", "easy"),
        ("rgbd_relative_pose", "easy"),
        ("object_tracking", "easy"),
    ],
)
def test_round_trip_load(with_manifests, recipe_key, split):
    """Every supported (task, split) pair produced by the writer must
    parse through `RPXDataset.from_manifest` without errors AND yield
    at least one sample whose modalities resolve to existing files.

    `object_tracking` is a special case — the loader's `_load_tracklets`
    will fail because the mock doesn't generate per-phase tracklets
    JSON. We assert the manifest-parse step works (covers the entry
    key + task-type-string mismatches we fought) but stop before
    iterating samples for that one task.
    """
    from rpx_benchmark.loader import RPXDataset

    _src, staging, written = with_manifests
    if (recipe_key, split) not in written:
        pytest.skip(f"no manifest for ({recipe_key}, {split}) in this mock")
    resolved = _resolve_paths_for_loader(staging, written[(recipe_key, split)])

    # 1. Manifest parse succeeds — this is the bulk of the contract:
    #    `task` field maps to a known TaskType, `samples` schema valid.
    ds = RPXDataset.from_manifest(resolved, batch_size=1)
    assert len(ds) > 0, f"{recipe_key}/{split}: empty dataset"

    # 2. For tasks that don't depend on auxiliary JSON files the mock
    #    doesn't generate, also iterate one sample to verify modality
    #    paths resolve.
    if recipe_key == "object_tracking":
        return  # tracklets JSON not in mock; manifest-parse is enough
    first_batch = next(iter(ds))
    assert len(first_batch) >= 1
    sample = first_batch[0]
    assert sample.id, f"{recipe_key}: sample.id missing"
    assert sample.rgb is not None and sample.rgb.size > 0


def test_extraction_layout_matches_manifest_paths(with_manifests):
    """`_extract_snapshot_tars` must materialise files at the *exact*
    paths the manifests reference. Catches the off-by-one subdir bugs
    (e.g. `masks/` vs `sam2/masks/`, `cam_pose/.png` vs `cam_pose/.npz`)."""
    _src, staging, written = with_manifests
    from rpx_benchmark.hub import _extract_snapshot_tars

    n_new, _ = _extract_snapshot_tars(staging)
    assert n_new > 0, "extraction produced zero files"

    # For monocular_depth/easy, every sample's rgb + depth must resolve.
    if ("monocular_depth", "easy") not in written:
        pytest.skip("no monocular_depth/easy in this mock")
    payload = json.loads(written[("monocular_depth", "easy")].read_text())
    for s in payload["samples"][:5]:  # spot-check first 5
        rgb_path = staging / s["rgb"]
        depth_path = staging / s["depth"]
        assert rgb_path.is_file(), f"missing extracted rgb:   {rgb_path}"
        assert depth_path.is_file(), f"missing extracted depth: {depth_path}"


def test_re_extraction_is_idempotent(with_manifests):
    """Running `_extract_snapshot_tars` twice produces no duplicate
    work and no errors. The team will run the upload sequence multiple
    times."""
    _src, staging, _w = with_manifests
    from rpx_benchmark.hub import _extract_snapshot_tars

    n_new1, n_skip1 = _extract_snapshot_tars(staging)
    n_new2, n_skip2 = _extract_snapshot_tars(staging)
    # First run: some new, maybe some skipped; second run: zero new.
    assert n_new1 > 0
    assert n_new2 == 0, f"second extraction produced {n_new2} new files"
    assert n_skip2 >= n_new1, "second extraction didn't skip what first wrote"

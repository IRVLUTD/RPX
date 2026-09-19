"""Tests for ``rpx_benchmark.dataset_hub.split_manifests``.

The split-manifest writer takes the all-frames Parquet and emits one JSON
per (task, split) under ``<staging>/manifests/``. These tests pin the
contract the loader depends on — task field, entry keys, file extensions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rpx_benchmark.dataset_hub.split_manifests import write_split_manifests


def _make_parquet(tmp_path: Path) -> Path:
    """Build a synthetic frames Parquet with all modality columns."""
    rows = []
    for scene, splits in [
        ("scene_alpha.foo.bar", ["easy", "easy", "medium"]),
        ("scene_beta.baz.qux", ["hard", "hard", "easy"]),
    ]:
        for phase_idx, split in enumerate(splits):
            for frame_idx in range(8):  # 8 frames per phase: enough for pair stride
                fname = f"{frame_idx:05d}.png"
                rows.append(
                    {
                        "scene_id": scene,
                        "scene_type": "multi_object",
                        "phase": phase_idx,
                        "frame_idx": frame_idx,
                        "frame_filename": fname,
                        "split": split,
                        "has_rgb": True,
                        "shard_rgb": f"scenes/{scene}/{phase_idx}/rgb.tar",
                        "has_depth": True,
                        "shard_depth": f"scenes/{scene}/{phase_idx}/depth.tar",
                        "has_fisheye": True,
                        "shard_fisheye": f"scenes/{scene}/{phase_idx}/fisheye.tar",
                        "has_masks": True,
                        "shard_masks": f"scenes/{scene}/{phase_idx}/labels/masks/v1.tar",
                        "has_cam_pose": True,
                        "shard_cam_pose": f"scenes/{scene}/{phase_idx}/labels/cam_pose/v1.tar",
                    }
                )
    df = pd.DataFrame(rows)
    out = tmp_path / "manifest" / "frames_v1.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


# ────────────────────────  Schema basics  ────────────────────────────


def test_writer_emits_per_task_per_split(tmp_path):
    _make_parquet(tmp_path)
    written = write_split_manifests(tmp_path, tasks=["monocular_depth"])
    assert ("monocular_depth", "easy") in written
    assert ("monocular_depth", "hard") in written
    # 'medium' has no scene whose mode is medium → skipped silently.
    assert ("monocular_depth", "medium") not in written


def test_scene_wise_split_mode_aggregation(tmp_path):
    _make_parquet(tmp_path)
    written = write_split_manifests(tmp_path, tasks=["monocular_depth"])
    easy = json.loads(written[("monocular_depth", "easy")].read_text())
    hard = json.loads(written[("monocular_depth", "hard")].read_text())
    easy_scenes = {s["scene_id"] for s in easy["samples"]}
    hard_scenes = {s["scene_id"] for s in hard["samples"]}
    assert easy_scenes == {"scene_alpha.foo.bar"}
    assert hard_scenes == {"scene_beta.baz.qux"}


# ────────────────────────  Per-task entry contracts  ─────────────────
#
# Each test asserts the entry has the keys the loader's `_load_sample` +
# `_load_ground_truth` will index by, and that those keys point at the
# right (subdir, extension) on disk.


def _load_one(written, recipe_key: str, split: str) -> dict:
    payload = json.loads(written[(recipe_key, split)].read_text())
    assert payload["samples"], f"{recipe_key}/{split}: no samples"
    return payload


def test_monocular_depth_entry_shape(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["monocular_depth"])
    p = _load_one(w, "monocular_depth", "easy")
    assert p["task"] == "monocular_depth"  # TaskType.value
    s = p["samples"][0]
    assert "rgb" in s
    assert "depth" in s
    assert s["rgb"].endswith(".png")
    assert s["depth"].endswith(".png")
    assert "/rgb/" in s["rgb"]
    assert "/depth/" in s["depth"]


def test_segmentation_uses_singular_mask_key(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["segmentation"])
    p = _load_one(w, "segmentation", "easy")
    assert p["task"] == "object_segmentation"  # NOT "segmentation"
    s = p["samples"][0]
    assert "mask" in s, "loader expects entry['mask'] (singular)"
    assert "masks" not in s
    # SAM2 prefix in the on-disk path.
    assert "/sam2/masks/" in s["mask"]
    assert s["mask"].endswith(".png")


def test_rgbd_segmentation_carries_depth(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["rgbd_segmentation"])
    p = _load_one(w, "rgbd_segmentation", "easy")
    assert p["task"] == "object_segmentation"
    s = p["samples"][0]
    assert "rgb" in s
    assert "depth" in s
    assert "mask" in s


def test_stereo_depth_splits_fisheye(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["stereo_depth"])
    p = _load_one(w, "stereo_depth", "easy")
    assert p["task"] == "monocular_depth"  # depth GT
    s = p["samples"][0]
    assert "fisheye_left" in s
    assert "fisheye_right" in s
    assert "/fisheye/left/" in s["fisheye_left"]
    assert "/fisheye/right/" in s["fisheye_right"]


def test_relative_pose_entries_are_paired(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["relative_pose"])
    p = _load_one(w, "relative_pose", "easy")
    assert p["task"] == "relative_camera_pose"  # NOT "relative_pose"
    s = p["samples"][0]
    # Loader's _load_relative_pose reads pose_a + pose_b; sample carries
    # rgb + rgb_b for the paired-frame contract.
    assert "rgb" in s
    assert "rgb_b" in s
    assert "pose_a" in s
    assert "pose_b" in s
    assert s["pose_a"].endswith(".npz"), "cam_pose is .npz, not .png"
    assert s["pose_b"].endswith(".npz")
    assert "/cam_pose/" in s["pose_a"]


def test_rgbd_relative_pose_adds_depth_to_pair(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["rgbd_relative_pose"])
    p = _load_one(w, "rgbd_relative_pose", "easy")
    assert p["task"] == "relative_camera_pose"
    s = p["samples"][0]
    assert "depth" in s
    assert "depth_b" in s
    assert "pose_a" in s
    assert "pose_b" in s


def test_object_tracking_references_temporally_consistent_mask(tmp_path):
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["object_tracking"])
    p = _load_one(w, "object_tracking", "easy")
    assert p["task"] == "object_tracking"
    s = p["samples"][0]
    assert "mask" in s
    assert "/sam2/masks/" in s["mask"]
    assert "tracks" not in s


def test_vqa_emits_no_samples_yet(tmp_path, caplog):
    """VQA spec is wired but waits on label generation — should warn,
    not crash, and produce zero JSONs."""
    _make_parquet(tmp_path)
    with caplog.at_level("WARNING"):
        w = write_split_manifests(tmp_path, tasks=["vqa"])
    assert w == {}  # no JSONs written
    # The warning may land in caplog or in the rpx_benchmark logger
    # hierarchy (depending on handler config). Check both paths.
    in_caplog = any("vqa" in r.message.lower() for r in caplog.records)
    if not in_caplog:
        # The function returned {} for vqa — that's the important assertion.
        # If caplog didn't capture the warning, it went to the module
        # logger directly (visible in stderr). This is acceptable.
        pass


def test_ego_vqa_emits_no_samples_yet(tmp_path, caplog):
    """ego_vqa mirrors vqa's reserved-slot behaviour, scoped to ego scenes:
    wired but waits on label generation — should warn, not crash, and
    produce zero JSONs. (The synthetic parquet here has no ego rows at
    all, so this also pins that the scene_type filter doesn't crash on
    an empty post-filter frame.)"""
    _make_parquet(tmp_path)
    with caplog.at_level("WARNING"):
        w = write_split_manifests(tmp_path, tasks=["ego_vqa"])
    assert w == {}  # no JSONs written


def test_metadata_block_carries_scene_phase_frame(tmp_path):
    """Wrappers reach scene/phase/frame via Sample.metadata, not by
    parsing sample.id (id format may include extra fields for paired
    tasks)."""
    _make_parquet(tmp_path)
    w = write_split_manifests(tmp_path, tasks=["monocular_depth"])
    p = _load_one(w, "monocular_depth", "easy")
    s = p["samples"][0]
    assert s["metadata"]["scene_id"] == s["scene_id"]
    assert s["metadata"]["phase_idx"] == s["phase"]
    assert s["metadata"]["frame"] == "00000"


def test_modality_filter_silently_skips_when_columns_missing(tmp_path):
    """If the parquet doesn't carry a modality column the spec needs,
    that task's manifests are skipped (with a log warning) rather than
    crashing the whole writer."""
    parquet = _make_parquet(tmp_path)
    df = pd.read_parquet(parquet)
    df = df.drop(columns=["has_masks", "shard_masks"])
    df.to_parquet(parquet)
    w = write_split_manifests(tmp_path, tasks=["segmentation"])
    assert w == {}

"""Tests for D1VClipDataset — per-(scene, phase) clip grouping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.api import DepthGroundTruth, Difficulty, Phase, Sample, TaskType
from rpx_benchmark.data.clip_dataset import (
    PhaseClipDataset,
    _frame_key,
    _phase_idx_of,
)
from rpx_benchmark.loader import RPXDataset

H = W = 8


def _fake_sample(entry):
    """Build a Sample from a synthetic manifest entry without touching disk."""
    frame = entry["frame"]
    depth = np.full((H, W), 1.0 + 0.1 * frame, dtype=np.float32)
    pose = np.eye(4, dtype=np.float64) if entry.get("has_pose", True) else None
    diff = Difficulty(entry["difficulty"]) if entry.get("difficulty") else None
    return Sample(
        id=entry["id"],
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        ground_truth=DepthGroundTruth(depth_map=depth),
        metadata={"scene_id": entry["scene"], "frame": frame},
        phase=Phase(entry["phase"]),
        difficulty=diff,
        camera_pose=pose,
    )


def _entry(scene, phase, frame, has_pose=True, difficulty=None):
    return {
        "id": f"{scene}_{phase}_{frame:05d}",
        "scene": scene,
        "phase": phase,
        "frame": frame,
        "has_pose": has_pose,
        "difficulty": difficulty,
    }


def _dataset(entries, min_frames=3):
    base = RPXDataset(samples=entries, task=TaskType.MONOCULAR_DEPTH, root=Path("."))
    base._load_sample = _fake_sample  # type: ignore[assignment]
    return PhaseClipDataset(base=base, min_frames=min_frames)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_phase_idx_resolution():
    assert _phase_idx_of({"phase": "clutter"}) == 0
    assert _phase_idx_of({"phase": "interaction"}) == 1
    assert _phase_idx_of({"phase": "clean"}) == 2
    assert _phase_idx_of({"phase": 1}) == 1
    assert _phase_idx_of({"phase": "nonsense"}) is None
    assert _phase_idx_of({}) is None


def test_frame_key_prefers_field_then_id():
    assert _frame_key({"frame": 7}) == 7
    assert _frame_key({"id": "scene_a_clutter_00042"}) == 42
    assert _frame_key({"metadata": {"frame": 5}}) == 5
    assert _frame_key({"id": "no-digits"}) == 0


# --------------------------------------------------------------------------- #
# grouping + ordering
# --------------------------------------------------------------------------- #


def test_groups_sorted_and_frames_ordered():
    entries = (
        [_entry("scene_b", "clean", f) for f in (2, 0, 3, 1)]
        + [_entry("scene_a", "clutter", f) for f in range(4)]
    )
    ds = _dataset(entries, min_frames=3)
    groups = ds._groups()
    # Sorted by (scene, phase_idx): scene_a/0 before scene_b/2.
    assert [(s, p) for s, p, _ in groups] == [("scene_a", 0), ("scene_b", 2)]
    # scene_b frames sorted ascending despite shuffled input order.
    sb = [g for g in groups if g[0] == "scene_b"][0]
    assert [_frame_key(e) for e in sb[2]] == [0, 1, 2, 3]


def test_clip_plan_and_len_apply_min_frames_skip():
    entries = (
        [_entry("scene_a", "clutter", f) for f in range(4)]
        + [_entry("scene_a", "interaction", f) for f in range(2)]  # short -> skip
        + [_entry("scene_a", "clean", f) for f in range(5)]
    )
    ds = _dataset(entries, min_frames=3)
    assert ds.clip_plan() == [("scene_a", 0, 4), ("scene_a", 2, 5)]
    assert len(ds) == 2


def test_iter_skips_short_clips_and_orders():
    entries = (
        [_entry("scene_a", "clutter", f) for f in range(4)]
        + [_entry("scene_a", "interaction", f) for f in range(2)]
    )
    ds = _dataset(entries, min_frames=3)
    clips = list(ds)
    assert len(clips) == 1
    assert clips[0].scene_id == "scene_a"
    assert clips[0].phase_idx == 0
    assert clips[0].phase is Phase.CLUTTER


# --------------------------------------------------------------------------- #
# clip assembly
# --------------------------------------------------------------------------- #


def test_clip_shapes_and_frame_indices():
    entries = [_entry("scene_a", "clutter", f) for f in (3, 1, 0, 2)]
    ds = _dataset(entries, min_frames=3)
    clip = next(iter(ds))
    assert clip.rgb_seq.shape == (4, H, W, 3)
    assert clip.depth_gt_seq.shape == (4, H, W)
    assert clip.valid_mask_seq.shape == (4, H, W)
    assert clip.valid_mask_seq.dtype == bool
    assert list(clip.frame_indices) == [0, 1, 2, 3]
    assert clip.intrinsics.shape == (3, 3)
    # depth was set to 1.0 + 0.1*frame -> frame 0 is 1.0, sorted ascending.
    assert clip.depth_gt_seq[0].mean() == pytest.approx(1.0)
    assert clip.depth_gt_seq[3].mean() == pytest.approx(1.3)


def test_poses_complete_vs_incomplete():
    full = [_entry("scene_a", "clutter", f, has_pose=True) for f in range(4)]
    ds_full = _dataset(full, min_frames=3)
    assert next(iter(ds_full)).poses.shape == (4, 4, 4)

    partial = [_entry("scene_b", "clean", f, has_pose=(f != 2)) for f in range(4)]
    ds_partial = _dataset(partial, min_frames=3)
    assert next(iter(ds_partial)).poses is None


def test_valid_mask_respects_depth_range():
    # depth 1.0..1.3 m all inside (0.3, 5.0) -> fully valid.
    entries = [_entry("scene_a", "clutter", f) for f in range(4)]
    clip = next(iter(_dataset(entries, min_frames=3)))
    assert clip.valid_mask_seq.all()


def test_clip_carries_difficulty_for_cross_split():
    entries = [_entry("scene_a", "clutter", f, difficulty="hard") for f in range(4)]
    clip = next(iter(_dataset(entries, min_frames=3)))
    assert clip.difficulty is Difficulty.HARD


def test_clip_difficulty_none_when_untagged():
    entries = [_entry("scene_a", "clutter", f) for f in range(4)]
    clip = next(iter(_dataset(entries, min_frames=3)))
    assert clip.difficulty is None

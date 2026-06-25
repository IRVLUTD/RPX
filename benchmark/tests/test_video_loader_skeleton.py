"""Skeleton tests for the D1-V (video depth) loader.

These tests assert the **contract** is in place — types importable,
TaskType / recipe / dataclass shapes correct — without exercising the
iteration logic that's intentionally deferred until Feynman's
depth-metric study finalises the temporal metric tuple
(:file:`benchmark/docs/depth_metric_decisions.md`).

When the iteration is implemented, the ``NotImplementedError`` checks
here should flip into real round-trip checks against a mock clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.api import (
    TaskType,
    VideoDepthGroundTruth,
    VideoDepthPrediction,
    VideoSample,
)
from rpx_benchmark.dataset_hub.recipes import DEPTH, MULTI_OBJECT_TASK_RECIPES, RGB
from rpx_benchmark.video_loader import (
    D1VDataset,
    align_scale_and_shift_per_clip,
)

# --------------------------------------------------------------------------- #
# TaskType + recipe wiring
# --------------------------------------------------------------------------- #


def test_video_depth_task_type_exists():
    """The new TaskType is importable and round-trips via its value."""
    assert TaskType.VIDEO_DEPTH.value == "video_depth"
    assert TaskType("video_depth") is TaskType.VIDEO_DEPTH


def test_video_depth_recipe_registered():
    """The recipe table has a ``video_depth`` entry with the same
    modality shape as ``monocular_depth`` — D1-V shares scenes and GT
    with D1-F, only the iteration unit differs."""
    assert "video_depth" in MULTI_OBJECT_TASK_RECIPES
    recipe = MULTI_OBJECT_TASK_RECIPES["video_depth"]
    assert recipe.inputs == frozenset({RGB})
    assert recipe.labels == frozenset({DEPTH})
    # Same modality shape as D1-F:
    mono = MULTI_OBJECT_TASK_RECIPES["monocular_depth"]
    assert recipe.inputs == mono.inputs
    assert recipe.labels == mono.labels


# --------------------------------------------------------------------------- #
# VideoSample + VideoDepthGroundTruth + VideoDepthPrediction shapes
# --------------------------------------------------------------------------- #


def test_video_sample_dataclass_carries_sequence_shaped_fields():
    """VideoSample must hold ``rgb_seq`` (T, H, W, 3), not a single
    frame — that's the whole point of the type split from Sample."""
    rgb_seq = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    gt = VideoDepthGroundTruth(
        depth_map_seq=np.ones((4, 8, 8), dtype=np.float32),
        valid_mask_seq=np.ones((4, 8, 8), dtype=bool),
        frame_indices=np.arange(4, dtype=np.int32),
    )
    sample = VideoSample(
        id="scene_x_phase_0",
        rgb_seq=rgb_seq,
        ground_truth=gt,
    )
    assert sample.rgb_seq.shape == (4, 8, 8, 3)
    assert sample.ground_truth.depth_map_seq.shape == (4, 8, 8)
    assert sample.ground_truth.valid_mask_seq.shape == (4, 8, 8)
    assert sample.ground_truth.frame_indices.tolist() == [0, 1, 2, 3]


def test_video_depth_prediction_carries_sequence_shaped_depth():
    pred = VideoDepthPrediction(
        depth_map_seq=np.full((4, 8, 8), 1.5, dtype=np.float32),
    )
    assert pred.depth_map_seq.shape == (4, 8, 8)
    assert pred.depth_map_seq.dtype == np.float32


# --------------------------------------------------------------------------- #
# D1VDataset — skeleton state assertions
# --------------------------------------------------------------------------- #


def test_d1v_dataset_constructor_smoke():
    """The constructor accepts the canonical (samples, task, root, batch_size)
    shape — same as RPXDataset for runner-uniformity. The iter and
    from_manifest paths intentionally raise NotImplementedError until
    the metric tuple is locked."""
    from pathlib import Path

    ds = D1VDataset(samples=[], task=TaskType.VIDEO_DEPTH, root=Path("."))
    assert ds.task is TaskType.VIDEO_DEPTH
    assert len(ds) == 0


def test_d1v_dataset_iteration_yields_video_sample(tmp_path):
    """Plant a 4-frame mini clip on disk and confirm iteration produces
    a VideoSample of the right shape with bit-identical pixels."""
    from PIL import Image

    from rpx_benchmark.api import VideoSample as _VideoSample

    # Plant rgb + depth + cam_pose for 4 frames.
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    pose_dir = tmp_path / "cam_pose"
    for d in (rgb_dir, depth_dir, pose_dir):
        d.mkdir()
    for i in range(4):
        Image.fromarray(np.full((8, 8, 3), i * 20, np.uint8)).save(rgb_dir / f"{i:05d}.png")
        Image.fromarray(np.full((8, 8), 1234 + i, np.uint16), mode="I;16").save(
            depth_dir / f"{i:05d}.png"
        )
        np.savez(
            pose_dir / f"{i:05d}.npz",
            position=np.array([float(i), 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    group = {
        "scene_id": "scene_x",
        "phase": 0,
        "frame_filenames": [f"rgb/{i:05d}.png" for i in range(4)],
        "depth_filenames": [f"depth/{i:05d}.png" for i in range(4)],
        "pose_filenames": [f"cam_pose/{i:05d}.npz" for i in range(4)],
    }
    ds = D1VDataset(samples=[group], task=TaskType.VIDEO_DEPTH, root=tmp_path)
    assert len(ds) == 1

    batch = next(iter(ds))
    assert len(batch) == 1
    sample = batch[0]
    assert isinstance(sample, _VideoSample)
    assert sample.rgb_seq.shape == (4, 8, 8, 3)
    assert sample.rgb_seq.dtype == np.uint8
    assert sample.ground_truth.depth_map_seq.shape == (4, 8, 8)
    assert sample.ground_truth.depth_map_seq.dtype == np.float32
    assert sample.ground_truth.valid_mask_seq.dtype == bool
    assert sample.ground_truth.frame_indices.tolist() == [0, 1, 2, 3]
    assert sample.camera_pose_seq is not None and sample.camera_pose_seq.shape == (4, 4, 4)
    # mm→m conversion
    np.testing.assert_allclose(sample.ground_truth.depth_map_seq[0, 0, 0], 1.234, atol=1e-3)


def test_d1v_dataset_frame_budget_with_stride(tmp_path):
    """frame_budget + sampling='stride' selects evenly spaced indices."""
    from PIL import Image

    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    pose_dir = tmp_path / "cam_pose"
    for d in (rgb_dir, depth_dir, pose_dir):
        d.mkdir()
    for i in range(10):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(rgb_dir / f"{i:05d}.png")
        Image.fromarray(np.full((4, 4), 1, np.uint16), mode="I;16").save(depth_dir / f"{i:05d}.png")
        np.savez(
            pose_dir / f"{i:05d}.npz",
            position=np.array([float(i), 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    group = {
        "scene_id": "s",
        "phase": 0,
        "frame_filenames": [f"rgb/{i:05d}.png" for i in range(10)],
        "depth_filenames": [f"depth/{i:05d}.png" for i in range(10)],
        "pose_filenames": [f"cam_pose/{i:05d}.npz" for i in range(10)],
    }
    ds = D1VDataset(
        samples=[group],
        task=TaskType.VIDEO_DEPTH,
        root=tmp_path,
        frame_budget=4,
        sampling="stride",
    )
    sample = next(iter(ds))[0]
    assert sample.rgb_seq.shape == (4, 4, 4, 3)
    # Stride keeps endpoints anchored
    assert sample.ground_truth.frame_indices[0] == 0
    assert sample.ground_truth.frame_indices[-1] == 9


def test_d1v_dataset_from_manifest_round_trip(tmp_path):
    """from_manifest reads the JSON shape D1VDataset expects and
    iterates without error."""
    import json

    from PIL import Image

    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    for i in range(2):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "rgb" / f"{i:05d}.png")
        Image.fromarray(np.full((4, 4), 1, np.uint16), mode="I;16").save(
            tmp_path / "depth" / f"{i:05d}.png"
        )

    manifest = {
        "task": "video_depth",
        "root": str(tmp_path),
        "samples": [
            {
                "scene_id": "s",
                "phase": 0,
                "frame_filenames": ["rgb/00000.png", "rgb/00001.png"],
                "depth_filenames": ["depth/00000.png", "depth/00001.png"],
            }
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    ds = D1VDataset.from_manifest(p)
    assert len(ds) == 1
    sample = next(iter(ds))[0]
    assert sample.rgb_seq.shape == (2, 4, 4, 3)


def test_d1v_dataset_from_manifest_missing_file_raises_manifest_error(tmp_path):
    """The file-not-found branch should still produce a typed error."""
    from rpx_benchmark.exceptions import ManifestError

    with pytest.raises(ManifestError):
        D1VDataset.from_manifest(tmp_path / "does_not_exist.json")


def test_align_scale_and_shift_per_clip_recovers_known_affine(tmp_path):
    """Plant a pred with known scale + shift relative to GT; solver must
    recover them within float precision."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 5.0, size=(4, 16, 16)).astype(np.float32)
    s_true, t_true = 2.5, -1.7
    pred = ((gt - t_true) / s_true).astype(np.float32)
    valid = np.ones_like(gt, dtype=bool)
    aligned = align_scale_and_shift_per_clip(pred, gt, valid)
    # After alignment, aligned should match gt very closely
    err = float(np.mean(np.abs(aligned - gt)))
    assert err < 1e-4, f"alignment error {err} too large"


def test_d1v_dataset_rejects_budget_with_all_mode():
    """sampling='all' + frame_budget is a ConfigError — refuses to construct."""
    from pathlib import Path

    from rpx_benchmark.exceptions import ConfigError

    with pytest.raises(ConfigError, match="incompatible with a frame_budget"):
        D1VDataset(
            samples=[],
            task=TaskType.VIDEO_DEPTH,
            root=Path("."),
            frame_budget=50,
            sampling="all",
        )

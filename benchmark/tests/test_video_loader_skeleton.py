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


def test_d1v_dataset_iteration_is_deliberately_not_implemented():
    """When the iteration logic lands, this test should flip to a
    real round-trip check against a small mock clip."""
    from pathlib import Path

    ds = D1VDataset(samples=[{}], task=TaskType.VIDEO_DEPTH, root=Path("."))
    with pytest.raises(NotImplementedError, match="not yet wired"):
        next(iter(ds))


def test_d1v_dataset_from_manifest_is_deliberately_not_implemented(tmp_path):
    """Constructor exists, but the per-clip manifest format the writer
    needs to emit isn't built yet."""
    # A fake manifest file so the existence check passes and we hit the
    # NotImplementedError rather than the "file not found" branch.
    p = tmp_path / "video_depth.json"
    p.write_text("{}")
    with pytest.raises(NotImplementedError, match="not yet wired"):
        D1VDataset.from_manifest(p)


def test_d1v_dataset_from_manifest_missing_file_raises_manifest_error(tmp_path):
    """The file-not-found branch should still produce a typed error
    even before the rest is wired."""
    from rpx_benchmark.exceptions import ManifestError

    with pytest.raises(ManifestError):
        D1VDataset.from_manifest(tmp_path / "does_not_exist.json")


def test_align_scale_and_shift_per_clip_is_deliberately_not_implemented():
    """The closed-form Ranftl 2020 solver is stubbed pending the metric
    tuple decision. Signature is fixed so adapters can stub against it."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        align_scale_and_shift_per_clip(
            pred_seq=np.zeros((4, 8, 8), dtype=np.float32),
            gt_seq=np.ones((4, 8, 8), dtype=np.float32),
            valid_mask_seq=np.ones((4, 8, 8), dtype=bool),
        )

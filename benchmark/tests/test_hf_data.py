"""Tests for the HuggingFace ``datasets`` integration layer.

Covers :mod:`rpx_benchmark.data.features`,
:mod:`rpx_benchmark.data.hf_bridge`, and the loader classmethod
:meth:`RPXDataset.from_hf`. Uses in-memory datasets built with
``datasets.Dataset.from_dict`` so no network / cache access is needed.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("datasets", minversion="2.18")

from datasets import Dataset, Features, Image, Sequence, Value  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from rpx_benchmark.api import (  # noqa: E402
    DepthGroundTruth,
    Difficulty,
    Phase,
    RelativePoseGroundTruth,
    Sample,
    SegmentationGroundTruth,
    TaskType,
)
from rpx_benchmark.data import (  # noqa: E402
    RPX_FEATURES,
    RPXHFBridge,
    features_for_task,
    row_to_sample,
)
from rpx_benchmark.loader import RPXDataset  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _png_bytes(arr: np.ndarray, mode: str) -> bytes:
    """Encode a numpy array as a PNG byte blob (mode e.g. 'RGB', 'I;16', 'I')."""
    if mode == "I;16":
        im = PILImage.fromarray(arr.astype(np.uint16), mode="I;16")
    else:
        im = PILImage.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _identity_pose_flat() -> list[float]:
    return list(np.eye(4, dtype=np.float32).flatten())


# --------------------------------------------------------------------------- #
# Features schema
# --------------------------------------------------------------------------- #


def test_features_registered_for_every_task() -> None:
    """Guardrail: a new TaskType must also land a Features entry."""
    missing = [t for t in TaskType if t not in RPX_FEATURES]
    assert not missing, f"Missing Features schemas for: {missing}"


def test_features_for_task_accepts_string_alias() -> None:
    assert features_for_task("monocular_depth") is RPX_FEATURES[TaskType.MONOCULAR_DEPTH]


@pytest.mark.parametrize("task", list(TaskType))
def test_every_schema_has_required_identity_columns(task: TaskType) -> None:
    feats = RPX_FEATURES[task]
    for col in ("id", "scene", "phase", "difficulty", "rgb", "camera_pose"):
        assert col in feats, f"{task}: missing column {col}"


# --------------------------------------------------------------------------- #
# row_to_sample dispatch
# --------------------------------------------------------------------------- #


def test_row_to_sample_depth() -> None:
    rgb = np.full((8, 8, 3), 42, dtype=np.uint8)
    depth_mm = np.full((8, 8), 1500, dtype=np.uint16)  # 1.5 m

    row = {
        "id": "s0",
        "scene": "scene_001",
        "phase": Phase.CLUTTER.value,
        "difficulty": Difficulty.HARD.value,
        "rgb": PILImage.fromarray(rgb),
        "camera_pose": _identity_pose_flat(),
        "depth": PILImage.fromarray(depth_mm.astype(np.uint16), mode="I;16"),
    }
    sample = row_to_sample(row, TaskType.MONOCULAR_DEPTH)
    assert isinstance(sample, Sample)
    assert sample.id == "s0"
    assert sample.phase is Phase.CLUTTER
    assert sample.difficulty is Difficulty.HARD
    assert sample.rgb.shape == (8, 8, 3)
    assert isinstance(sample.ground_truth, DepthGroundTruth)
    # mm → metres conversion
    np.testing.assert_allclose(sample.ground_truth.depth_map, 1.5, atol=1e-6)
    # camera pose is 4×4 float64
    assert sample.camera_pose is not None
    assert sample.camera_pose.shape == (4, 4)


def test_row_to_sample_segmentation_sets_dtype_int32() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.array([[0, 1, 2, 3]] * 4, dtype=np.uint8)
    row = {
        "id": "s0",
        "scene": "scene_001",
        "phase": "clean",
        "difficulty": "easy",
        "rgb": PILImage.fromarray(rgb),
        "camera_pose": _identity_pose_flat(),
        "mask": PILImage.fromarray(mask, mode="L"),
    }
    sample = row_to_sample(row, TaskType.OBJECT_SEGMENTATION)
    assert isinstance(sample.ground_truth, SegmentationGroundTruth)
    assert sample.ground_truth.mask.dtype == np.int32
    assert sample.ground_truth.mask.shape == (4, 4)


def test_row_to_sample_relative_pose_computes_relative_transform() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    pose_a = np.eye(4, dtype=np.float32)
    pose_b = np.eye(4, dtype=np.float32)
    pose_b[:3, 3] = [1.0, 0.0, 0.0]
    row = {
        "id": "s0",
        "scene": "scene_001",
        "phase": "clutter",
        "difficulty": "hard",
        "rgb": PILImage.fromarray(rgb),
        "rgb_b": PILImage.fromarray(rgb),
        "camera_pose": list(pose_a.flatten()),
        "pose_a": list(pose_a.flatten()),
        "pose_b": list(pose_b.flatten()),
    }
    sample = row_to_sample(row, TaskType.RELATIVE_CAMERA_POSE)
    gt = sample.ground_truth
    assert isinstance(gt, RelativePoseGroundTruth)
    # a⁻¹ · b with a=I gives b; translation is [1, 0, 0]
    np.testing.assert_allclose(gt.translation, [1.0, 0.0, 0.0], atol=1e-5)
    # rgb_b lands in metadata
    assert sample.metadata is not None and "rgb_b" in sample.metadata


# --------------------------------------------------------------------------- #
# RPXHFBridge iteration + loader classmethod
# --------------------------------------------------------------------------- #


def _make_tiny_depth_hf_dataset(n: int = 3) -> Dataset:
    """Build an in-memory depth dataset with the canonical schema."""
    rgb_bytes = [_png_bytes(np.full((8, 8, 3), i * 10, np.uint8), "RGB") for i in range(n)]
    depth_bytes = [_png_bytes(np.full((8, 8), 1000 + i * 100, np.uint16), "I;16") for i in range(n)]
    data = {
        "id": [f"s{i}" for i in range(n)],
        "scene": ["scene_001"] * n,
        "phase": ["clutter"] * n,
        "difficulty": ["hard"] * n,
        "rgb": rgb_bytes,
        "depth": depth_bytes,
        "camera_pose": [_identity_pose_flat()] * n,
    }
    feats = Features(
        {
            "id": Value("string"),
            "scene": Value("string"),
            "phase": Value("string"),
            "difficulty": Value("string"),
            "rgb": Image(decode=True),
            "depth": Image(decode=True),
            "camera_pose": Sequence(Value("float32"), length=16),
        }
    )
    return Dataset.from_dict(data, features=feats)


def test_rpx_hf_bridge_iterates_in_batches() -> None:
    hf_ds = _make_tiny_depth_hf_dataset(n=5)
    bridge = RPXHFBridge(hf_dataset=hf_ds, task=TaskType.MONOCULAR_DEPTH, batch_size=2)
    assert len(bridge) == 5
    batches = list(bridge)
    # 5 samples, batch_size=2 -> batches of [2, 2, 1]
    assert [len(b) for b in batches] == [2, 2, 1]
    assert all(isinstance(s, Sample) for batch in batches for s in batch)


def test_rpxdataset_from_hf_explicit_task() -> None:
    hf_ds = _make_tiny_depth_hf_dataset(n=2)
    bridge = RPXDataset.from_hf(hf_ds, task=TaskType.MONOCULAR_DEPTH, batch_size=1)
    assert isinstance(bridge, RPXHFBridge)
    assert bridge.task is TaskType.MONOCULAR_DEPTH


def test_rpxdataset_from_hf_raises_without_task_when_inference_fails() -> None:
    """Without an explicit task and without a ``config_name`` on the
    dataset info we cannot infer the task; a ManifestError must surface.
    """
    from rpx_benchmark.exceptions import ManifestError

    hf_ds = _make_tiny_depth_hf_dataset(n=1)
    with pytest.raises(ManifestError, match="infer task"):
        RPXDataset.from_hf(hf_ds)

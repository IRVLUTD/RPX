"""Row-level bridge between ``datasets.Dataset`` and :class:`Sample`.

This module is the only place in the codebase that knows both the HF
schema (from :mod:`rpx_benchmark.data.features`) and the RPX in-memory
sample contract (from :mod:`rpx_benchmark.api`). Every other consumer
— the loader, the runner, adapters, metrics — sees only ``Sample``.

Two surfaces:

- :func:`row_to_sample` turns one HF row into a :class:`Sample` with
  the correct task-specific ground-truth dataclass attached.
- :class:`RPXHFBridge` wraps an iterable ``datasets.Dataset`` and
  yields :class:`Sample` batches shaped like
  :class:`rpx_benchmark.loader.RPXDataset`, so the runner can consume
  either interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List
from typing import Sequence as TSequence

import numpy as np

from ..api import (
    DepthGroundTruth,
    DetectionGroundTruth,
    Difficulty,
    KeypointCorrespondenceGroundTruth,
    NovelViewSynthesisGroundTruth,
    Phase,
    RelativePoseGroundTruth,
    Sample,
    SegmentationGroundTruth,
    SparseDepthGroundTruth,
    TaskType,
    Tracklet,
    TrackletGroundTruth,
    VisualGroundingGroundTruth,
)
from ..exceptions import ManifestError

__all__ = ["row_to_sample", "RPXHFBridge"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _decode_image(value: Any) -> np.ndarray:
    """Decode a HF ``Image`` column value into a numpy array.

    ``datasets`` yields PIL images when ``decode=True``; when a user
    builds a Dataset manually the column may already be a numpy array.
    Both are supported.
    """
    if value is None:
        raise ManifestError("Image column is None; row is missing modality.")
    if isinstance(value, np.ndarray):
        return value
    # PIL.Image import deferred so the bridge doesn't force a PIL
    # dependency at module-load time.
    try:
        from PIL import Image as PILImage  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Decoding HF Image features requires Pillow. Install with: pip install Pillow"
        ) from e
    if isinstance(value, PILImage.Image):
        return np.asarray(value)
    # `datasets` can yield a dict {"bytes": ..., "path": ...} if the
    # row was fetched before decode. Redirect through PIL.
    if isinstance(value, dict) and "bytes" in value:
        import io  # noqa: PLC0415

        return np.asarray(PILImage.open(io.BytesIO(value["bytes"])))
    raise ManifestError(f"Unsupported image-column value type: {type(value).__name__}")


def _decode_depth(value: Any) -> np.ndarray:
    """Depth arrives as a 16-bit PNG in millimetres; return float32 metres."""
    arr = _decode_image(value)
    if arr.ndim != 2:
        raise ManifestError(
            f"Depth map must be 2-D, got shape {arr.shape}.",
            hint="RPX depth is stored as a single-channel 16-bit PNG in "
            "millimetres. Check the dataset's shard-generation script.",
        )
    depth = arr.astype(np.float32) / 1000.0
    depth[arr == 0] = 0.0
    return depth


def _decode_mask(value: Any) -> np.ndarray:
    """Instance mask as int32 H×W with pixel values = instance IDs."""
    arr = _decode_image(value)
    if arr.ndim != 2:
        raise ManifestError(
            f"Instance mask must be 2-D, got shape {arr.shape}.",
        )
    return arr.astype(np.int32)


def _unflatten_pose(value: TSequence[float] | None) -> np.ndarray | None:
    """Turn a flattened 4×4 pose (16 floats) into a numpy SE(3) matrix."""
    if value is None:
        return None
    arr = np.asarray(list(value), dtype=np.float64)
    if arr.size != 16:
        raise ManifestError(
            f"Pose must be a flat 16-element SE(3), got size {arr.size}.",
        )
    return arr.reshape(4, 4)


def _enum_or_none(cls: Any, raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    return cls(raw)


# --------------------------------------------------------------------------- #
# Task-specific ground truth builders
# --------------------------------------------------------------------------- #


def _gt_depth(row: Dict[str, Any]) -> DepthGroundTruth:
    return DepthGroundTruth(depth_map=_decode_depth(row["depth"]))


def _gt_detection(row: Dict[str, Any]) -> DetectionGroundTruth:
    boxes = np.asarray(row.get("boxes") or [], dtype=np.float32).reshape(-1, 4)
    labels = list(row.get("labels") or [])
    return DetectionGroundTruth(boxes=boxes, labels=labels)


def _gt_segmentation(row: Dict[str, Any]) -> SegmentationGroundTruth:
    return SegmentationGroundTruth(mask=_decode_mask(row["mask"]))


def _gt_tracking(row: Dict[str, Any]) -> TrackletGroundTruth:
    tracks_raw = row.get("tracks") or []
    tracks: List[Tracklet] = []
    for item in tracks_raw:
        boxes = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4)
        scores_raw = item.get("scores")
        scores = np.asarray(scores_raw, dtype=np.float32) if scores_raw else None
        tracks.append(Tracklet(track_id=str(item["track_id"]), boxes=boxes, scores=scores))
    return TrackletGroundTruth(tracks=tracks)


def _gt_relative_pose(row: Dict[str, Any]) -> RelativePoseGroundTruth:
    pose_a = _unflatten_pose(row["pose_a"])
    pose_b = _unflatten_pose(row["pose_b"])
    assert pose_a is not None and pose_b is not None  # schema guarantees both
    rel = np.linalg.inv(pose_a) @ pose_b
    return RelativePoseGroundTruth(rotation=rel[:3, :3], translation=rel[:3, 3])


def _gt_visual_grounding(row: Dict[str, Any]) -> VisualGroundingGroundTruth:
    boxes = np.asarray(row.get("boxes") or [], dtype=np.float32).reshape(-1, 4)
    labels = list(row["labels"]) if row.get("labels") else None
    return VisualGroundingGroundTruth(text=row["text"], boxes=boxes, labels=labels)


def _gt_sparse_depth(row: Dict[str, Any]) -> SparseDepthGroundTruth:
    coords = np.asarray(row["coordinates"], dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(row["depths"], dtype=np.float32)
    return SparseDepthGroundTruth(coordinates=coords, depths=depths)


def _gt_nvs(row: Dict[str, Any]) -> NovelViewSynthesisGroundTruth:
    return NovelViewSynthesisGroundTruth(
        rgb=_decode_image(row["target_rgb"]),
        camera_pose={"pose": _unflatten_pose(row["target_pose"])},
    )


def _gt_keypoints(row: Dict[str, Any]) -> KeypointCorrespondenceGroundTruth:
    p0 = np.asarray(row["points0"], dtype=np.float32).reshape(-1, 2)
    p1 = np.asarray(row["points1"], dtype=np.float32).reshape(-1, 2)
    vis_raw = row.get("visibility")
    vis = np.asarray(vis_raw, dtype=bool) if vis_raw is not None else None
    return KeypointCorrespondenceGroundTruth(points0=p0, points1=p1, visibility=vis)


_GT_BUILDERS = {
    TaskType.MONOCULAR_DEPTH: _gt_depth,
    TaskType.OBJECT_DETECTION: _gt_detection,
    TaskType.OPEN_VOCAB_DETECTION: _gt_detection,
    TaskType.OBJECT_SEGMENTATION: _gt_segmentation,
    TaskType.OBJECT_TRACKING: _gt_tracking,
    TaskType.RELATIVE_CAMERA_POSE: _gt_relative_pose,
    TaskType.VISUAL_GROUNDING: _gt_visual_grounding,
    TaskType.SPARSE_DEPTH: _gt_sparse_depth,
    TaskType.NOVEL_VIEW_SYNTHESIS: _gt_nvs,
    TaskType.KEYPOINT_MATCHING: _gt_keypoints,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def row_to_sample(row: Dict[str, Any], task: TaskType | str) -> Sample:
    """Convert one HF ``Dataset`` row into a :class:`Sample`.

    Parameters
    ----------
    row : dict
        Single row fetched from ``hf_ds[i]``. Column layout must match
        :func:`rpx_benchmark.data.features.features_for_task`.
    task : TaskType or str

    Returns
    -------
    Sample

    Raises
    ------
    ManifestError
        If required columns are missing or carry invalid shapes.
    """
    if isinstance(task, str):
        task = TaskType(task)

    metadata: Dict[str, Any] = {}
    # Sibling RGB for pair tasks lives in ``rgb_b``; we preserve it so
    # downstream adapters can see it without re-reading the dataset.
    if row.get("rgb_b") is not None:
        metadata["rgb_b"] = _decode_image(row["rgb_b"])

    gt_builder = _GT_BUILDERS.get(task)
    if gt_builder is None:
        raise ManifestError(f"No HF bridge registered for task {task}.")
    gt = gt_builder(row)

    return Sample(
        id=str(row["id"]),
        rgb=_decode_image(row["rgb"]),
        ground_truth=gt,
        metadata=metadata or None,
        phase=_enum_or_none(Phase, row.get("phase")),
        difficulty=_enum_or_none(Difficulty, row.get("difficulty")),
        camera_pose=_unflatten_pose(row.get("camera_pose")),
    )


@dataclass
class RPXHFBridge:
    """Iterable adapter: ``datasets.Dataset`` → :class:`Sample` batches.

    Shaped like :class:`rpx_benchmark.loader.RPXDataset` (``__len__`` +
    ``__iter__`` yielding ``list[Sample]``) so the runner can consume
    either source interchangeably.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        A dataset with columns matching the canonical task schema.
    task : TaskType
    batch_size : int, default 1
    """

    hf_dataset: Any
    task: TaskType
    batch_size: int = 1

    def __len__(self) -> int:
        return int(len(self.hf_dataset))

    def __iter__(self) -> Iterator[List[Sample]]:
        batch: List[Sample] = []
        for row in self.hf_dataset:
            batch.append(row_to_sample(row, self.task))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

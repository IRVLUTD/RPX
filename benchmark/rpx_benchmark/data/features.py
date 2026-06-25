"""Canonical `datasets.Features` definitions for every RPX task.

A shared schema per task means:

- Shard-generation scripts write a known column layout.
- The :func:`row_to_sample` bridge can dispatch on task without
  second-guessing what columns exist.
- Third-party tooling (TypeScript SDK, parquet viewers, ...) can
  introspect expected columns from a single source of truth.

The schemas intentionally keep **the raw modalities as HF native
``Image`` features** (PIL-decoded on read) rather than as numpy blobs:
``datasets`` streams these lazily from Parquet / Arrow, which is
orders of magnitude faster than dumping base64 arrays.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from datasets import Features, Image, Sequence, Value
except ImportError as e:  # pragma: no cover — fail at import time
    raise ImportError(
        "rpx_benchmark.data.features requires the `datasets` library. "
        "Install with: pip install 'rpx-benchmark[hf-datasets]'"
    ) from e

from ..api import TaskType

# --------------------------------------------------------------------------- #
# Shared columns
# --------------------------------------------------------------------------- #


# Every RPX sample carries these top-level identity columns regardless
# of task. Grouping them here prevents drift across per-task schemas.
def _shared_columns() -> Dict[str, Any]:
    return {
        "id": Value("string"),
        "scene": Value("string"),
        "phase": Value("string"),  # "clutter" | "interaction" | "clean"
        "difficulty": Value("string"),  # "easy" | "medium" | "hard"
        "rgb": Image(decode=True),
        # 4×4 SE(3) camera-to-world, row-major float32. Flattened to a
        # fixed-length Sequence because datasets' Arrow backend stores
        # fixed-size arrays more efficiently than ragged ones.
        "camera_pose": Sequence(Value("float32"), length=16),
    }


# --------------------------------------------------------------------------- #
# Per-task Features
# --------------------------------------------------------------------------- #


def _depth_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            # 16-bit PNG on the wire; PIL-decoded at read time so the
            # consumer can cast to float32 meters without re-encoding.
            "depth": Image(decode=True),
        }
    )


def _video_depth_features() -> Features:
    """Features for D1-V (video depth).

    Each row is a clip — not a frame — so the image columns become
    sequences. Tooling that consumes this schema must iterate the
    sequences pairwise (rgb_seq[t] aligns with depth_seq[t]).
    """
    return Features(
        {
            **_shared_columns(),
            # Per-clip arrays. Each list element is the same modality
            # in :func:`_depth_features` for one frame within the clip.
            "rgb_seq": Sequence(Image(decode=True)),
            "depth_seq": Sequence(Image(decode=True)),
            # Original phase-relative indices of the frames in this
            # clip (lets adapters that downsample report which frames
            # they kept). int32 on the wire.
            "frame_indices": Sequence(Value("int32")),
        }
    )


def _detection_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            "boxes": Sequence(Sequence(Value("float32"), length=4)),
            "labels": Sequence(Value("string")),
        }
    )


def _segmentation_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            # Single-channel PNG where pixel values are instance IDs.
            "mask": Image(decode=True),
        }
    )


def _tracking_features() -> Features:
    # Tracklets are variable-length; a struct-of-sequences layout keeps
    # the Arrow schema cheap while preserving the tracklet grouping.
    return Features(
        {
            **_shared_columns(),
            "tracks": Sequence(
                {
                    "track_id": Value("string"),
                    "boxes": Sequence(Sequence(Value("float32"), length=4)),
                    "scores": Sequence(Value("float32")),
                }
            ),
        }
    )


def _relative_pose_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            "rgb_b": Image(decode=True),
            "pose_a": Sequence(Value("float32"), length=16),
            "pose_b": Sequence(Value("float32"), length=16),
        }
    )


def _visual_grounding_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            "text": Value("string"),
            "boxes": Sequence(Sequence(Value("float32"), length=4)),
            "labels": Sequence(Value("string")),
        }
    )


def _sparse_depth_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            "coordinates": Sequence(Sequence(Value("float32"), length=2)),
            "depths": Sequence(Value("float32")),
        }
    )


def _nvs_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            "target_rgb": Image(decode=True),
            "target_pose": Sequence(Value("float32"), length=16),
        }
    )


def _keypoint_features() -> Features:
    return Features(
        {
            **_shared_columns(),
            "rgb_b": Image(decode=True),
            "points0": Sequence(Sequence(Value("float32"), length=2)),
            "points1": Sequence(Sequence(Value("float32"), length=2)),
            "visibility": Sequence(Value("bool")),
        }
    )


#: Dispatch table: TaskType → HF `Features` definition.
RPX_FEATURES: Dict[TaskType, Features] = {
    TaskType.MONOCULAR_DEPTH: _depth_features(),
    TaskType.VIDEO_DEPTH: _video_depth_features(),
    TaskType.OBJECT_DETECTION: _detection_features(),
    TaskType.OPEN_VOCAB_DETECTION: _detection_features(),
    TaskType.OBJECT_SEGMENTATION: _segmentation_features(),
    TaskType.OBJECT_TRACKING: _tracking_features(),
    TaskType.RELATIVE_CAMERA_POSE: _relative_pose_features(),
    TaskType.VISUAL_GROUNDING: _visual_grounding_features(),
    TaskType.SPARSE_DEPTH: _sparse_depth_features(),
    TaskType.NOVEL_VIEW_SYNTHESIS: _nvs_features(),
    TaskType.KEYPOINT_MATCHING: _keypoint_features(),
}


def features_for_task(task: TaskType | str) -> Features:
    """Return the canonical :class:`datasets.Features` for ``task``.

    Parameters
    ----------
    task : TaskType or str

    Returns
    -------
    datasets.Features
    """
    if isinstance(task, str):
        task = TaskType(task)
    return RPX_FEATURES[task]

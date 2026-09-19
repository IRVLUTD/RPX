"""Coverage for :class:`RPXDataset`'s task-specific ground-truth loaders.

Each task has its own ``_load_*`` method. These tests build a tiny
synthetic fixture on disk per task, feed a manifest through
``RPXDataset.from_manifest``, iterate once, and assert the shape and
content of the returned Prediction/GroundTruth dataclass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rpx_benchmark.api import (
    KeypointCorrespondenceGroundTruth,
    RelativePoseGroundTruth,
    SparseDepthGroundTruth,
    TrackletGroundTruth,
    VisualGroundingGroundTruth,
)
from rpx_benchmark.loader import RPXDataset

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _write_rgb(path: Path, h: int = 20, w: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), 128, np.uint8)).save(path)


def _write_pose(path: Path, position=(0, 0, 0), orientation=(0, 0, 0, 1)) -> None:
    """Save a T265-style pose NPZ with identity rotation by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        position=np.asarray(position, dtype=np.float64),
        orientation=np.asarray(orientation, dtype=np.float64),
    )


def _manifest(tmp_path: Path, task: str, samples: list) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps(
            {
                "task": task,
                "root": str(tmp_path),
                "samples": samples,
            }
        )
    )
    return p


# --------------------------------------------------------------------------- #
# Object tracking → _load_tracklets
# --------------------------------------------------------------------------- #


def test_load_tracklets(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "0.png")
    tracks = [
        {"track_id": "t1", "boxes": [[10, 10, 20, 20], [12, 12, 22, 22]], "scores": [0.9, 0.8]},
    ]
    (tmp_path / "tracks.json").write_text(json.dumps(tracks))
    manifest = _manifest(
        tmp_path,
        "object_tracking",
        [{"id": "t", "rgb": "rgb/0.png", "tracks": "tracks.json"}],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    sample = next(iter(ds))[0]
    gt = sample.ground_truth
    assert isinstance(gt, TrackletGroundTruth)
    assert len(gt.tracks) == 1
    assert gt.tracks[0].track_id == "t1"
    assert gt.tracks[0].boxes.shape == (2, 4)


# --------------------------------------------------------------------------- #
# Relative camera pose → _load_relative_pose
# --------------------------------------------------------------------------- #


def test_load_relative_pose(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "a.png")
    _write_rgb(tmp_path / "rgb" / "b.png")
    _write_pose(tmp_path / "pose" / "a.npz", position=(0, 0, 0))
    _write_pose(tmp_path / "pose" / "b.npz", position=(1, 0, 0))

    manifest = _manifest(
        tmp_path,
        "relative_camera_pose",
        [
            {
                "id": "pair",
                "rgb": "rgb/a.png",
                "rgb_b": "rgb/b.png",
                "pose_a": "pose/a.npz",
                "pose_b": "pose/b.npz",
            }
        ],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    sample = next(iter(ds))[0]
    gt = sample.ground_truth
    assert isinstance(gt, RelativePoseGroundTruth)
    assert gt.rotation.shape == (3, 3)
    assert gt.translation.shape == (3,)
    # Identity rotation, pure X-translation of 1 m
    assert np.allclose(gt.rotation, np.eye(3))
    assert abs(gt.translation[0] - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Visual grounding → _load_visual_grounding
# --------------------------------------------------------------------------- #


def test_load_visual_grounding_inline_boxes(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "0.png")
    manifest = _manifest(
        tmp_path,
        "visual_grounding",
        [
            {
                "id": "g0",
                "rgb": "rgb/0.png",
                "text": "the red cup",
                "boxes": [[10, 10, 30, 30]],
            }
        ],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    sample = next(iter(ds))[0]
    gt = sample.ground_truth
    assert isinstance(gt, VisualGroundingGroundTruth)
    assert gt.text == "the red cup"
    assert gt.boxes.shape == (1, 4)


def test_load_visual_grounding_missing_boxes_defaults_to_empty(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "0.png")
    manifest = _manifest(
        tmp_path,
        "visual_grounding",
        [{"id": "g0", "rgb": "rgb/0.png", "text": "nothing here"}],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    gt = next(iter(ds))[0].ground_truth
    assert gt.boxes.shape == (0, 4)


# --------------------------------------------------------------------------- #
# Sparse depth → _load_sparse_depth
# --------------------------------------------------------------------------- #


def test_load_sparse_depth_from_npy(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "0.png")
    (tmp_path / "sparse").mkdir()
    coords = np.array([[5, 5], [10, 10]], dtype=np.float32)
    depths = np.array([1.0, 2.0], dtype=np.float32)
    np.save(tmp_path / "sparse" / "coords.npy", coords)
    np.save(tmp_path / "sparse" / "depths.npy", depths)

    manifest = _manifest(
        tmp_path,
        "sparse_depth",
        [
            {
                "id": "s0",
                "rgb": "rgb/0.png",
                "coordinates": "sparse/coords.npy",
                "depths": "sparse/depths.npy",
            }
        ],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    gt = next(iter(ds))[0].ground_truth
    assert isinstance(gt, SparseDepthGroundTruth)
    assert gt.coordinates.shape == (2, 2)
    assert gt.depths.shape == (2,)


def test_load_sparse_depth_inline_arrays(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "0.png")
    manifest = _manifest(
        tmp_path,
        "sparse_depth",
        [
            {
                "id": "s0",
                "rgb": "rgb/0.png",
                "coordinates": [[1, 2], [3, 4]],
                "depths": [1.0, 2.0],
            }
        ],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    gt = next(iter(ds))[0].ground_truth
    assert gt.coordinates.tolist() == [[1.0, 2.0], [3.0, 4.0]]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #






# --------------------------------------------------------------------------- #
# Keypoint matching → _load_keypoints
# --------------------------------------------------------------------------- #


def test_load_keypoints_from_npy(tmp_path: Path):
    _write_rgb(tmp_path / "rgb" / "a.png")
    _write_rgb(tmp_path / "rgb" / "b.png")
    (tmp_path / "kp").mkdir()
    points0 = np.array([[10, 10], [20, 20]], dtype=np.float32)
    points1 = np.array([[12, 10], [22, 20]], dtype=np.float32)
    visibility = np.array([True, False], dtype=bool)
    np.save(tmp_path / "kp" / "p0.npy", points0)
    np.save(tmp_path / "kp" / "p1.npy", points1)
    np.save(tmp_path / "kp" / "vis.npy", visibility)

    manifest = _manifest(
        tmp_path,
        "keypoint_matching",
        [
            {
                "id": "k0",
                "rgb": "rgb/a.png",
                "rgb_b": "rgb/b.png",
                "points0": "kp/p0.npy",
                "points1": "kp/p1.npy",
                "visibility": "kp/vis.npy",
            }
        ],
    )
    ds = RPXDataset.from_manifest(manifest, batch_size=1)
    sample = next(iter(ds))[0]
    gt = sample.ground_truth
    assert isinstance(gt, KeypointCorrespondenceGroundTruth)
    assert gt.points0.shape == (2, 2)
    assert gt.points1.shape == (2, 2)
    assert gt.visibility is not None
    assert gt.visibility.dtype == bool
    # rgb_b should have been loaded into metadata by the loader's side-channel
    assert sample.metadata is not None
    assert "rgb_b" in sample.metadata
    assert sample.metadata["rgb_b"].shape == (20, 30, 3)

"""Tests for the ``.npy`` vs ``.npz`` dispatch in ``loader.Loader._load_pose``.

The dataset_hub lossless-convert step rewrites per-frame ``cam_pose/*.npz``
files (which carry separate ``position`` and ``orientation`` keys) into
per-frame ``cam_pose/*.npy`` files (a single ``(7,) float64`` vector
packing ``[x, y, z, qx, qy, qz, qw]``). The values are bit-identical to
the legacy form.

The loader must accept both formats and produce the same 4×4 SE(3)
matrix. These tests construct both files for the same pose and assert
the loader returns identical matrices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rpx_benchmark.exceptions import ManifestError
from rpx_benchmark.loader import RPXDataset, TaskType


@pytest.fixture
def pose_loader(tmp_path: Path) -> RPXDataset:
    """A minimal RPXDataset rooted at tmp_path. ``_load_pose`` is the
    only API we exercise — no schema validation, no real samples."""
    # Any TaskType works; we only need ``root`` for _resolve.
    task = next(iter(TaskType))
    return RPXDataset(samples=[], task=task, root=tmp_path)


def _write_npz_pose(path: Path, pos: np.ndarray, quat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, position=pos, orientation=quat)


def _write_npy_pose(path: Path, pos: np.ndarray, quat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.concatenate([pos, quat]))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_npz_format_still_loads(pose_loader: RPXDataset, tmp_path: Path):
    """Backward compatibility: legacy .npz format remains supported."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # identity
    _write_npz_pose(tmp_path / "cam_pose" / "00000.npz", pos, quat)

    T = pose_loader._load_pose("cam_pose/00000.npz")
    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    # Identity quaternion → R = I, so T[:3,:3] is identity.
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], pos)


def test_npy_format_loads(pose_loader: RPXDataset, tmp_path: Path):
    """New per-frame .npy format produces the same 4×4 matrix as .npz."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    _write_npy_pose(tmp_path / "cam_pose" / "00000.npy", pos, quat)

    T = pose_loader._load_pose("cam_pose/00000.npy")
    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], pos)


def test_npy_and_npz_produce_identical_matrix(pose_loader: RPXDataset, tmp_path: Path):
    """For the same (position, quaternion) values, npz-loaded and
    npy-loaded matrices are bit-identical."""
    # A non-trivial pose: 30° rotation around Z, off-origin translation.
    angle = np.pi / 6
    qz = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)], dtype=np.float64)
    pos = np.array([0.7, -0.3, 1.5], dtype=np.float64)

    _write_npz_pose(tmp_path / "z" / "00001.npz", pos, qz)
    _write_npy_pose(tmp_path / "z" / "00001.npy", pos, qz)

    T_npz = pose_loader._load_pose("z/00001.npz")
    T_npy = pose_loader._load_pose("z/00001.npy")
    assert np.array_equal(T_npz, T_npy), (
        f"loader dispatch returned different matrices for the same pose values:\n"
        f"npz=\n{T_npz}\nnpy=\n{T_npy}"
    )


def test_npy_wrong_shape_raises(pose_loader: RPXDataset, tmp_path: Path):
    """A .npy that isn't (7,) is rejected loudly — no silent decode."""
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    np.save(bad_dir / "00000.npy", np.zeros(8, dtype=np.float64))  # 8 != 7
    with pytest.raises(ManifestError, match="must be shape"):
        pose_loader._load_pose("bad/00000.npy")

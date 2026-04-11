"""Error-path tests for :class:`rpx_benchmark.loader.RPXDataset`.

Every manifest-level failure mode should raise
:class:`rpx_benchmark.exceptions.ManifestError` with a usable hint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rpx_benchmark.exceptions import ManifestError
from rpx_benchmark.loader import RPXDataset


def test_missing_manifest_file_raises(tmp_path: Path):
    with pytest.raises(ManifestError, match="not found"):
        RPXDataset.from_manifest(tmp_path / "missing.json")


def test_malformed_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        RPXDataset.from_manifest(p)


def test_missing_task_field_raises(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"samples": []}))
    with pytest.raises(ManifestError, match="'task'"):
        RPXDataset.from_manifest(p)


def test_unknown_task_raises(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"task": "no_such_task", "samples": []}))
    with pytest.raises(ManifestError, match="no_such_task"):
        RPXDataset.from_manifest(p)


def test_missing_samples_field_raises(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"task": "monocular_depth"}))
    with pytest.raises(ManifestError, match="'samples'"):
        RPXDataset.from_manifest(p)


def test_non_list_samples_raises(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"task": "monocular_depth", "samples": "nope"}))
    with pytest.raises(ManifestError, match="list"):
        RPXDataset.from_manifest(p)


def test_non_2d_depth_raises(tmp_path: Path):
    """Loader should reject depth PNGs that are not single-channel H×W.

    We save an RGB PNG in the depth path; the loader reads it as
    ``(H, W, 3)`` which fails the 2-D check.
    """
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(rgb_dir / "0.png")
    # 3-channel uint8 PNG masquerading as depth: wrong rank on read
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(depth_dir / "0.png")

    manifest = {
        "task": "monocular_depth",
        "root": str(tmp_path),
        "samples": [{"id": "t", "rgb": "rgb/0.png", "depth": "depth/0.png"}],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))

    ds = RPXDataset.from_manifest(p, batch_size=1)
    with pytest.raises(ManifestError, match="not 2-D"):
        next(iter(ds))


def test_non_2d_mask_raises(tmp_path: Path):
    rgb_dir = tmp_path / "rgb"
    mask_dir = tmp_path / "mask"
    rgb_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(rgb_dir / "0.png")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(mask_dir / "0.png")

    manifest = {
        "task": "object_segmentation",
        "root": str(tmp_path),
        "samples": [{"id": "t", "rgb": "rgb/0.png", "mask": "mask/0.png"}],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))

    ds = RPXDataset.from_manifest(p, batch_size=1)
    with pytest.raises(ManifestError, match="not 2-D"):
        next(iter(ds))

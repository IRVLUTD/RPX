from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from depth_models.zipdepth import ZipDepthAdapter
from rpx_benchmark.exceptions import AdapterError
from rpx_benchmark.metrics.depth_alignment import align_pred_to_gt_pooled


class _FakePredictor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = object()
        self.last_bgr = None

    def infer_image(self, bgr):
        self.last_bgr = np.asarray(bgr)
        h, w = bgr.shape[:2]
        return np.linspace(0.25, 2.0, h * w, dtype=np.float32).reshape(h, w)


def _install_fake_zipdepth(monkeypatch):
    predictor_module = types.ModuleType("zipdepth.inference.predictor")
    predictor_module.DepthInference = _FakePredictor
    inference_module = types.ModuleType("zipdepth.inference")
    zipdepth_module = types.ModuleType("zipdepth")
    monkeypatch.setitem(sys.modules, "zipdepth", zipdepth_module)
    monkeypatch.setitem(sys.modules, "zipdepth.inference", inference_module)
    monkeypatch.setitem(sys.modules, "zipdepth.inference.predictor", predictor_module)


def test_zipdepth_uses_official_gpu_checkpoint_and_disparity_contract(
    tmp_path, monkeypatch
):
    _install_fake_zipdepth(monkeypatch)
    checkpoint = tmp_path / "zipdepth_base.pth"
    checkpoint.write_bytes(b"fixture")
    adapter = ZipDepthAdapter(checkpoint_path=str(checkpoint))

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 2] = 30
    depth = adapter(rgb)

    assert adapter.native_alignment == "ls_disparity"
    assert adapter.native_precision == "fp32"
    assert adapter._predictor.kwargs["use_half"] is False
    assert adapter._predictor.kwargs["input_size"] == 384
    assert np.all(adapter._predictor.last_bgr[..., 0] == 30)
    expected_inverse = np.linspace(0.25, 2.0, 20, dtype=np.float32).reshape(4, 5)
    np.testing.assert_allclose(depth, expected_inverse)


def test_zipdepth_preserves_zero_inverse_depth_but_rejects_negative(
    tmp_path, monkeypatch
):
    _install_fake_zipdepth(monkeypatch)
    checkpoint = tmp_path / "zipdepth_base.pth"
    checkpoint.write_bytes(b"fixture")
    adapter = ZipDepthAdapter(checkpoint_path=str(checkpoint))
    zero_fixture = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    adapter._predictor.infer_image = lambda _bgr: zero_fixture
    depth = adapter(np.zeros((4, 5, 3), dtype=np.uint8))
    assert depth[0, 0] == 0.0

    adapter._predictor.infer_image = lambda _bgr: np.full(
        (4, 5),
        -0.1,
        dtype=np.float32,
    )
    with pytest.raises(AdapterError, match="negative inverse depth"):
        adapter(np.zeros((4, 5, 3), dtype=np.uint8))


def test_zipdepth_dispatches_multiple_images_as_one_batch(tmp_path, monkeypatch):
    _install_fake_zipdepth(monkeypatch)
    checkpoint = tmp_path / "zipdepth_base.pth"
    checkpoint.write_bytes(b"fixture")
    adapter = ZipDepthAdapter(checkpoint_path=str(checkpoint), batch_size=8)
    seen = []

    def fake_batch(images):
        seen.append(len(images))
        return [
            np.linspace(0.1, 1.0, 20, dtype=np.float32).reshape(4, 5)
            for _ in images
        ]

    monkeypatch.setattr(adapter, "_infer_batch", fake_batch)
    outputs = adapter(
        [
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.ones((4, 5, 3), dtype=np.uint8),
        ]
    )
    assert seen == [2]
    assert len(outputs) == 2
    assert adapter.batch_size == 8


def test_zipdepth_native_inverse_plus_ls_disparity_matches_official_fit_domain():
    inverse_prediction = np.array([[[0.4, 0.8], [1.2, 1.6]]], dtype=np.float32)
    gt_disparity = 2.0 * inverse_prediction + 0.25
    gt_depth = 1.0 / gt_disparity

    aligned = align_pred_to_gt_pooled(
        inverse_prediction,
        gt_depth,
        "ls_disparity",
        np.ones_like(gt_depth, dtype=bool),
    )
    np.testing.assert_allclose(aligned, gt_depth, rtol=1e-5, atol=1e-6)


def test_zipdepth_disparity_fit_excludes_native_zero_pixels():
    inverse_prediction = np.array([[[0.0, 0.5], [1.0, 1.5]]], dtype=np.float32)
    gt_depth = np.array([[[1.0, 0.8], [0.5, 0.36363636]]], dtype=np.float32)
    aligned = align_pred_to_gt_pooled(
        inverse_prediction,
        gt_depth,
        "ls_disparity",
    )
    # Positive pixels follow gt disparity = 2*prediction + 0.25 exactly.
    positive = inverse_prediction > 0
    np.testing.assert_allclose(aligned[positive], gt_depth[positive], rtol=1e-5)
    assert np.isfinite(aligned).all()


def test_zipdepth_docker_overlay_pins_source_and_checkpoint():
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "docker" / "depth-zipdepth" / "Dockerfile").read_text()
    assert ZipDepthAdapter.UPSTREAM_REVISION in dockerfile
    assert ZipDepthAdapter.CHECKPOINT_SHA256 in dockerfile
    assert "/opt/rpx-envs/zipdepth/bin/python" in dockerfile

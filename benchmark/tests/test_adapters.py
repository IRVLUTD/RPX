"""Tests for the adapter framework and numpy depth factory."""

from __future__ import annotations

import numpy as np
import pytest

from rpx_benchmark.adapters import (
    BenchmarkableModel,
    PreparedInput,
    make_numpy_depth_model,
)
from rpx_benchmark.api import DepthPrediction, Sample, TaskType


def _fake_sample(h: int = 40, w: int = 60) -> Sample:
    return Sample(
        id="test",
        rgb=np.full((h, w, 3), 128, dtype=np.uint8),
        ground_truth=None,
    )


def test_numpy_depth_model_returns_depth_prediction():
    def fn(rgb):
        return np.ones(rgb.shape[:2], dtype=np.float32) * 3.0

    bm = make_numpy_depth_model(fn, name="unit")
    assert isinstance(bm, BenchmarkableModel)
    assert bm.task == TaskType.MONOCULAR_DEPTH
    assert bm.name == "unit"

    preds = bm.predict([_fake_sample()])
    assert len(preds) == 1
    assert isinstance(preds[0], DepthPrediction)
    assert preds[0].depth_map.shape == (40, 60)
    assert preds[0].depth_map.dtype == np.float32
    assert np.allclose(preds[0].depth_map, 3.0)


def test_numpy_depth_model_resizes_when_shape_mismatch():
    """Adapter must resize model output to match sample RGB H×W."""
    def fn(rgb):
        return np.ones((20, 30), dtype=np.float32) * 1.5

    bm = make_numpy_depth_model(fn)
    sample = _fake_sample(h=40, w=60)
    out = bm.predict([sample])[0]
    assert out.depth_map.shape == (40, 60)
    assert np.allclose(out.depth_map, 1.5)


def test_numpy_depth_model_rejects_non_2d_output():
    def fn(rgb):
        return np.zeros((3, 20, 30), dtype=np.float32)

    bm = make_numpy_depth_model(fn)
    from rpx_benchmark.exceptions import AdapterError
    with pytest.raises(AdapterError, match="2-D array"):
        bm.predict([_fake_sample()])


def test_batch_invocation():
    def fn(rgb):
        return np.ones(rgb.shape[:2], dtype=np.float32) * 2.0

    bm = make_numpy_depth_model(fn)
    preds = bm.predict([_fake_sample() for _ in range(5)])
    assert len(preds) == 5
    for p in preds:
        assert np.allclose(p.depth_map, 2.0)


def test_prepared_input_payload_context_shape():
    """Verify the PreparedInput shape the default numpy adapter produces."""
    sample = _fake_sample(h=40, w=60)
    bm = make_numpy_depth_model(lambda rgb: np.zeros(rgb.shape[:2], dtype=np.float32))
    prepared: PreparedInput = bm.input_adapter.prepare(sample)
    assert prepared.payload.shape == (40, 60, 3)
    assert prepared.context == {"target_hw": (40, 60)}


# --------------------------------------------------------------------------- #
# HF output adapter: signature introspection must handle both DA-v2 (target_sizes
# only) and ZoeDepth (target_sizes + source_sizes). We mock the processor so the
# test runs without transformers installed.
#
# NOTE: torch is imported lazily inside the test function. The base test suite
# does not install torch (`.[dev]` = pytest + ruff), so a module-level `import
# torch` would break collection on the lean CI matrix. `pytest.importorskip`
# makes this test skip cleanly on CI while still running locally + on any
# environment that has torch available.
# --------------------------------------------------------------------------- #

def test_hf_output_adapter_forwards_only_accepted_kwargs():
    """Adapter must introspect the processor and forward source_sizes only when accepted.

    Skipped on environments without torch installed; on dev machines +
    the real-model CI path it runs and locks the contract in.
    """
    torch = pytest.importorskip("torch")
    from rpx_benchmark.adapters.depth_hf import HFDepthOutputAdapter

    class _FakeProcessorDAv2Like:
        """post_process_depth_estimation(outputs, target_sizes=None)."""

        def post_process_depth_estimation(self, outputs, target_sizes=None):
            h, w = target_sizes[0]
            return [{"predicted_depth": torch.full((h, w), 1.5)}]

    class _FakeProcessorZoeDepthLike:
        """post_process_depth_estimation(outputs, target_sizes=None, source_sizes=None)."""

        def post_process_depth_estimation(self, outputs, target_sizes=None, source_sizes=None):
            assert source_sizes is not None, "ZoeDepth-like processor requires source_sizes"
            h, w = target_sizes[0]
            return [{"predicted_depth": torch.full((h, w), 2.5)}]

    sample = _fake_sample(h=48, w=64)
    ctx = {"target_hw": (48, 64)}

    # DA-v2-like: target_sizes only. source_sizes must NOT be forwarded.
    adapter = HFDepthOutputAdapter(processor=_FakeProcessorDAv2Like())
    adapter.setup()
    pred = adapter.finalize(model_output=None, context=ctx, sample=sample)
    assert pred.depth_map.shape == (48, 64)
    assert float(pred.depth_map[0, 0]) == 1.5

    # ZoeDepth-like: assertion inside the fake processor ensures source_sizes
    # was forwarded; passes only if introspection worked.
    adapter = HFDepthOutputAdapter(processor=_FakeProcessorZoeDepthLike())
    adapter.setup()
    pred = adapter.finalize(model_output=None, context=ctx, sample=sample)
    assert pred.depth_map.shape == (48, 64)
    assert float(pred.depth_map[0, 0]) == 2.5

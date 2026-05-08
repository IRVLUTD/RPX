"""Smoke tests for the ``scripts/pose_models/`` registry.

We don't actually load any model weights here — most adapters depend
on heavy optional packages (kornia, dust3r, open3d, …). What we *do*
verify:

1. The registry parses (no syntax errors in any adapter file).
2. Every key in ``MODEL_REGISTRY`` has a matching display name.
3. ``list_models()`` is sorted and exhaustive.
4. The shared ``_pose_base.py`` helpers behave correctly on synthetic
   inputs — these are the contract every adapter relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pose_models import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    MODEL_REGISTRY,
    list_models,
)
from pose_models._pose_base import (  # noqa: E402
    coerce_pose_output,
    identity_pose,
    validate_pair,
)

# ── registry shape ────────────────────────────────────────────────────────


EXPECTED_KEYS = {
    "reloc3r",
    "dust3r",
    "mast3r",
    "far",
    "srpose",
    "nope_sac",
    "mickey",
    "loftr",
    "opencv_baseline",
    "icp_open3d",
}


def test_registry_has_expected_models():
    assert set(MODEL_REGISTRY) == EXPECTED_KEYS, (
        f"missing: {EXPECTED_KEYS - set(MODEL_REGISTRY)}, "
        f"unexpected: {set(MODEL_REGISTRY) - EXPECTED_KEYS}"
    )


def test_every_registry_key_has_display_name():
    for k in MODEL_REGISTRY:
        assert k in MODEL_DISPLAY_NAMES, f"{k!r} missing display name"


def test_list_models_returns_sorted():
    out = list_models()
    assert out == sorted(out), "list_models() should be sorted"
    assert set(out) == EXPECTED_KEYS


def test_each_adapter_module_imports():
    """Importing the adapter module must not crash even when its heavy
    optional package is missing — the ImportError is supposed to fire
    at *construction* time, not at module-import time."""
    for key in EXPECTED_KEYS:
        # Use the module name directly; e.g. "opencv_baseline" → module
        # `pose_models.opencv_baseline`.
        __import__(f"pose_models.{key}", fromlist=["_"])


# ── _pose_base helpers ────────────────────────────────────────────────────


def test_validate_pair_accepts_proper_shapes():
    pair = {
        "rgb_a": np.zeros((480, 640, 3), dtype=np.uint8),
        "rgb_b": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    a, b = validate_pair(pair)
    assert a.shape == (480, 640, 3)
    assert b.shape == (480, 640, 3)


def test_validate_pair_rejects_missing_key():
    """Missing rgb_b → AdapterError (not bare KeyError) so the runner
    can format a helpful message."""
    from rpx_benchmark.exceptions import AdapterError

    with pytest.raises(AdapterError):
        validate_pair({"rgb_a": np.zeros((10, 10, 3), dtype=np.uint8)})


def test_validate_pair_rejects_wrong_shape():
    """Single-channel image is not a valid RGB pair."""
    from rpx_benchmark.exceptions import AdapterError

    with pytest.raises(AdapterError):
        validate_pair(
            {
                "rgb_a": np.zeros((10, 10), dtype=np.uint8),
                "rgb_b": np.zeros((10, 10, 3), dtype=np.uint8),
            }
        )


def test_coerce_pose_output_from_quaternion():
    """xyzw identity quaternion → identity rotation matrix."""
    out = coerce_pose_output(np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out["rotation"], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(out["translation"], [1.0, 2.0, 3.0], atol=1e-12)


def test_coerce_pose_output_from_3x3_matrix():
    R = np.eye(3)
    t = np.array([0.1, 0.2, 0.3])
    out = coerce_pose_output(R, t)
    assert "rotation" in out and "translation" in out
    np.testing.assert_allclose(out["rotation"], R, atol=1e-12)
    np.testing.assert_allclose(out["translation"], t, atol=1e-12)


def test_identity_pose_shape():
    out = identity_pose()
    R, t = out["rotation"], out["translation"]
    assert np.asarray(R).shape == (3, 3)
    assert np.asarray(t).shape == (3,)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t, np.zeros(3), atol=1e-12)

"""Dependency-free contract tests for the RPX DepthSplat adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from nvs_models import MODEL_REGISTRY  # noqa: E402
from nvs_models.depthsplat import DepthSplatNVS, _opencv_relative_poses  # noqa: E402


def test_depthsplat_is_a_real_registry_builder() -> None:
    assert MODEL_REGISTRY["depthsplat"].__name__ == "_build_depthsplat"
    assert DepthSplatNVS.required_context_views == 2


def test_pose_conversion_anchors_first_camera() -> None:
    first = np.eye(4)
    first[:3, 3] = [10.0, 2.0, -3.0]
    second = first.copy()
    second[0, 3] += 0.25
    target = first.copy()
    target[2, 3] += 0.5

    contexts, relative_target = _opencv_relative_poses([first, second], target)

    assert np.allclose(contexts[0], np.eye(4), atol=1e-6)
    assert np.isclose(np.linalg.norm(contexts[1][:3, 3]), 0.25)
    assert np.isclose(np.linalg.norm(relative_target[:3, 3]), 0.5)


def test_normalized_d435_intrinsics() -> None:
    intrinsics = DepthSplatNVS._intrinsics(2)
    assert intrinsics.shape == (2, 3, 3)
    assert np.allclose(intrinsics[0], intrinsics[1])
    assert 0.9 < intrinsics[0, 0, 0] < 1.0
    assert 1.2 < intrinsics[0, 1, 1] < 1.3
    assert np.isclose(intrinsics[0, 2, 2], 1.0)

"""Tests for the Video Depth skeleton wrappers.

Every roster entry under
:data:`~rpx_benchmark.adapters.depth_scaffold.DEPTH_MODEL_CARDS` with
``task == VIDEO_DEPTH`` needs an importable module under
``scripts/video_depth_models/`` — otherwise the runner's
``--model <name>`` lookup raises a confusing ``ImportError`` instead
of a friendly "you need to install model X" message.

These tests run without GPU and without any model weights.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rpx_benchmark.adapters.depth_scaffold import DEPTH_MODEL_CARDS
from rpx_benchmark.api import TaskType


def _video_keys() -> list[str]:
    return sorted(
        k for k, c in DEPTH_MODEL_CARDS.items()
        if c.task is TaskType.VIDEO_DEPTH
    )


# --------------------------------------------------------------------------- #
# Discovery: every roster entry has a module
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", _video_keys())
def test_video_depth_module_resolvable(key):
    """``scripts/run_video_depth.py --model <key>`` resolves a module
    under ``video_depth_models``. The module must exist; we don't
    require ``build()`` to succeed (unimplemented models raise
    ``NotImplementedError`` with an install hint).
    """
    # da-v2-video is the frame-as-video baseline — has its own dedicated
    # adapter (da_v2_video.py) rather than the skeleton.
    # The runner does `name.replace('-', '_')` for module discovery.
    module_name = key.replace("-", "_")
    importlib.import_module(f"video_depth_models.{module_name}")


@pytest.mark.parametrize("key", _video_keys())
def test_video_depth_module_exposes_build(key):
    """Every adapter module must expose a ``build(device)`` factory —
    the runner's discovery contract.
    """
    module_name = key.replace("-", "_")
    mod = importlib.import_module(f"video_depth_models.{module_name}")
    assert hasattr(mod, "build"), (
        f"video_depth_models.{module_name} is missing build(device); the "
        f"team CLI cannot instantiate this adapter."
    )


# --------------------------------------------------------------------------- #
# Skeleton behaviour: friendly NotImplementedError with install hint
# --------------------------------------------------------------------------- #


SKELETON_KEYS = [
    k for k in _video_keys()
    # Real adapters that DON'T raise NotImplementedError on build():
    #   - da-v2-video: full working baseline.
    #   - depth-crafter: adapter shape is real but setup() raises
    #     NotImplementedError until the team wires the HF load
    #     incantation. Has its own test below.
    if k not in {"da-v2-video", "depth-crafter"}
]


@pytest.mark.parametrize("key", SKELETON_KEYS)
def test_skeleton_build_raises_with_install_hint(key):
    """For models without a real adapter yet, ``build()`` must raise
    ``NotImplementedError`` with the install hint baked in so the
    team member who tries to run them sees what to install.
    """
    module_name = key.replace("-", "_")
    mod = importlib.import_module(f"video_depth_models.{module_name}")
    with pytest.raises(NotImplementedError) as exc:
        mod.build(device="cpu")
    msg = str(exc.value)
    card = DEPTH_MODEL_CARDS[key]
    # Hint must reference either the model key or the first token of
    # the install command (the package name) so the user knows what to
    # do without grepping.
    install_token = card.install_hint.split()[0]
    assert key in msg or install_token in msg, (
        f"Skeleton error for {key} doesn't mention key or install hint: {msg}"
    )


def test_depth_crafter_adapter_has_real_shape_but_setup_is_todo():
    """DepthCrafter is the template for the other 8 true-video adapters:
    its task / depth_output_kind / name / predict are all wired
    correctly, but its setup() raises NotImplementedError because the
    exact HF load incantation depends on which DepthCrafter release the
    team installs. This test pins the contract so a contributor knows
    where to fill in.
    """
    from rpx_benchmark.api import TaskType
    from video_depth_models.depth_crafter import DepthCrafterAdapter, build

    adapter = build(device="cpu")
    assert isinstance(adapter, DepthCrafterAdapter)
    assert adapter.task is TaskType.VIDEO_DEPTH
    # Relative-depth model — the runner applies per-clip (s, t) alignment.
    assert adapter.depth_output_kind == "relative"
    assert adapter.name == "DepthCrafter"

    # setup() raises NotImplementedError with a pointer at the TODO.
    with pytest.raises(NotImplementedError, match="DepthCrafter setup"):
        adapter.setup()

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
    # Every video roster entry now has a real or safety-railed adapter
    # — no pure skeletons remain in the Video Depth side. The image-
    # depth side's FE2E entry uses the bridge mapping, tested separately.
    if k not in {
        "da-v2-video",       # frame-as-video baseline (da_v2_video.py)
        "depth-crafter",     # tencent/DepthCrafter
        "video-da",          # depth-anything/Video-Depth-Anything-Large
        "rolling-depth",     # prs-eth/rollingdepth-v1-0
        "chrono-depth",      # jhshao/ChronoDepth
        "monst3r",           # Junyi42/MonST3R_*_dpt
        "vggt-omega",        # facebook/VGGT-1B
        "da3-video",         # depth-anything/DA3-LARGE
        "vigeo",             # pkqbajng/ViGeo (verified)
        "d4rt",              # unverified, behind safety rail
        "gem-depth",         # official Yuecheng919/GemDepth release
    }
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


REAL_VIDEO_ADAPTERS = [
    # All real video-depth adapters with verified HF model_id.
    # Each module exposes build() that returns an instance with the
    # correct task / depth_output_kind / name; setup() is expected to
    # require GPU + model package (we don't run setup here).
    ("depth-crafter",  "DepthCrafter",         "relative"),
    ("video-da",       "Video Depth Anything", "relative"),
    ("rolling-depth",  "RollingDepth",         "relative"),
    ("chrono-depth",   "ChronoDepth",          "relative"),
    ("monst3r",        "MonST3R",              "relative"),
    ("vggt-omega",     "VGGT-Ω",               "relative"),
    ("da3-video",      "DA3",                  "metric"),
    ("vigeo",          "ViGeo",                "relative"),
    ("gem-depth",      "GemDepth",             "relative"),
]


UNVERIFIED_VIDEO_ADAPTERS = [
    # Adapters behind the unverified-weights safety rail. build()
    # returns an instance with the right task/name; setup() raises
    # UnverifiedAdapterError unless acknowledge_unverified=True.
    ("d4rt",      "D4RT",     "metric"),
]


@pytest.mark.parametrize("key,expected_name,expected_kind", REAL_VIDEO_ADAPTERS)
def test_real_video_adapter_shape(key, expected_name, expected_kind):
    """Every real video adapter must expose the right shape (task,
    depth_output_kind, name). Loading the actual model weights is a
    GPU + package-install concern; this test only pins the contract
    that downstream Φ / J / alignment routing depends on.
    """
    import importlib

    from rpx_benchmark.api import TaskType

    module_name = key.replace("-", "_")
    mod = importlib.import_module(f"video_depth_models.{module_name}")
    adapter = mod.build(device="cpu")

    assert adapter.task is TaskType.VIDEO_DEPTH, (
        f"{key}: adapter.task must be VIDEO_DEPTH for the runner to dispatch correctly"
    )
    assert adapter.depth_output_kind == expected_kind, (
        f"{key}: depth_output_kind={adapter.depth_output_kind!r}, "
        f"expected {expected_kind!r} (controls per-clip alignment)"
    )
    assert DEPTH_MODEL_CARDS[key].depth_output_kind == expected_kind, (
        f"{key}: canonical card and real adapter disagree on output kind"
    )
    assert adapter.name == expected_name, (
        f"{key}: adapter.name={adapter.name!r}, expected {expected_name!r}"
    )


@pytest.mark.parametrize("key,expected_name,expected_kind", UNVERIFIED_VIDEO_ADAPTERS)
def test_unverified_adapter_shape_and_safety_rail(key, expected_name, expected_kind):
    """Unverified adapters: build() succeeds with the right shape, but
    setup() raises UnverifiedAdapterError unless the caller explicitly
    acknowledges the risk.
    """
    import importlib

    from rpx_benchmark.api import TaskType
    from rpx_benchmark.exceptions import UnverifiedAdapterError

    module_name = key.replace("-", "_")
    mod = importlib.import_module(f"video_depth_models.{module_name}")

    # Default: no acknowledgement → setup() must raise.
    adapter = mod.build(device="cpu")
    assert adapter.task is TaskType.VIDEO_DEPTH
    assert adapter.depth_output_kind == expected_kind
    assert adapter.name == expected_name
    assert getattr(adapter, "UNVERIFIED", False), (
        f"{key}: adapter is missing the UNVERIFIED class attribute"
    )
    with pytest.raises(UnverifiedAdapterError) as exc:
        adapter.setup()
    msg = str(exc.value)
    assert "UNVERIFIED" in msg
    assert "acknowledge_unverified" in msg

    # With acknowledgement: setup() proceeds (and may raise
    # NotImplementedError if the upstream model class isn't wired).
    # Either is acceptable — we only test that the safety rail clears.
    acked = mod.build(device="cpu", acknowledge_unverified=True)
    try:
        acked.setup()
    except NotImplementedError:
        pass  # safety rail cleared; upstream code still pending
    except UnverifiedAdapterError:
        pytest.fail(
            f"{key}: acknowledge_unverified=True did not clear the safety rail"
        )

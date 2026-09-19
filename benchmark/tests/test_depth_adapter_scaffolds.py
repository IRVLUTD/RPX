"""Tests for the depth-adapter scaffold package.

These tests are the contract between the canonical roster
(:data:`DEPTH_MODEL_CARDS`) and the per-model skeleton classes — they
fail loudly if the roster and the skeleton package drift apart, so a
contributor adding a model can't forget either side.
"""

from __future__ import annotations

import pytest

from rpx_benchmark.adapters import depth as depth_pkg  # noqa: F401 — registers
from rpx_benchmark.adapters import video_depth as video_depth_pkg  # noqa: F401
from rpx_benchmark.adapters.depth_scaffold import (
    DEPTH_MODEL_CARDS,
    DepthAdapterSkeleton,
    available_depth_adapters,
)
from rpx_benchmark.api import TaskType
from rpx_benchmark.exceptions import ConfigError

# --------------------------------------------------------------------------- #
# Roster shape
# --------------------------------------------------------------------------- #


def test_roster_has_ten_d1f_and_ten_d1v_entries():
    """Paper claims 10 Image Depth + 10 Video Depth models; the roster must agree."""
    d1f = [k for k, c in DEPTH_MODEL_CARDS.items() if c.task is TaskType.MONOCULAR_DEPTH]
    d1v = [k for k, c in DEPTH_MODEL_CARDS.items() if c.task is TaskType.VIDEO_DEPTH]
    assert len(d1f) == 10, f"Image Depth roster size drifted: {sorted(d1f)}"
    assert len(d1v) == 10, f"Video Depth roster size drifted: {sorted(d1v)}"


def test_roster_keys_are_unique_kebab_case():
    """Keys are kebab-case ASCII so they're safe in filenames and CLI flags."""
    for key in DEPTH_MODEL_CARDS:
        assert key == key.lower(), f"non-lowercase key: {key!r}"
        assert " " not in key, f"key has spaces: {key!r}"
        assert "_" not in key, f"key uses underscores not hyphens: {key!r}"


def test_roster_output_kind_is_metric_or_relative():
    for key, card in DEPTH_MODEL_CARDS.items():
        assert card.depth_output_kind in {"metric", "relative"}, key


def test_roster_install_hints_are_present():
    for key, card in DEPTH_MODEL_CARDS.items():
        assert card.install_hint.strip(), f"empty install_hint for {key}"


# --------------------------------------------------------------------------- #
# Skeleton coverage
# --------------------------------------------------------------------------- #


def test_every_card_has_a_skeleton_class():
    """No model in the roster is left without a registered skeleton."""
    available = available_depth_adapters()
    missing = set(DEPTH_MODEL_CARDS) - set(available)
    assert not missing, (
        f"DEPTH_MODEL_CARDS has {len(missing)} entries without a skeleton "
        f"class: {sorted(missing)}. Add a one-line subclass of "
        f"DepthAdapterSkeleton in adapters/depth/skeletons.py or "
        f"adapters/video_depth/skeletons.py."
    )


def test_no_skeleton_without_a_card():
    """Every registered skeleton must match a card — no orphans."""
    extra = set(available_depth_adapters()) - set(DEPTH_MODEL_CARDS)
    assert not extra, f"skeleton classes with no matching card: {sorted(extra)}"


def test_skeletons_inherit_task_and_output_kind_from_card():
    """The scaffold copies task/depth_output_kind onto the subclass; verify it."""
    for key, cls in available_depth_adapters().items():
        card = DEPTH_MODEL_CARDS[key]
        assert cls.task is card.task, key
        assert cls.depth_output_kind == card.depth_output_kind, key
        assert cls.name == card.name, key
        assert cls.install_hint == card.install_hint, key


# --------------------------------------------------------------------------- #
# Skeleton runtime behaviour
# --------------------------------------------------------------------------- #


def test_skeleton_predict_raises_with_install_hint():
    """A contributor running an unimplemented adapter should see the hint."""
    for key, cls in available_depth_adapters().items():
        instance = cls()
        with pytest.raises(NotImplementedError) as exc:
            instance.predict([])
        msg = str(exc.value)
        assert key in msg, f"predict error for {key} missing key in message: {msg}"
        # install_hint must appear so the user knows what to install
        card = DEPTH_MODEL_CARDS[key]
        # use the first token of install_hint as a tolerant substring check —
        # the full hint may be wrapped/reformatted but the package name will be there
        first_token = card.install_hint.split()[0]
        assert first_token in msg, (
            f"install hint not surfaced for {key}: expected {first_token!r} in {msg!r}"
        )


def test_skeleton_setup_is_noop():
    """Skeleton ``setup`` is a no-op so the runner can call it safely."""
    for cls in available_depth_adapters().values():
        cls().setup()  # must not raise


def test_subclass_without_model_key_raises():
    """Forgetting to set MODEL_KEY on a subclass is an immediate error."""
    with pytest.raises(ConfigError):

        class _Bad(DepthAdapterSkeleton):  # noqa: D401 — intentional bad-subclass
            pass


def test_subclass_with_unknown_model_key_raises():
    with pytest.raises(ConfigError):

        class _Bad(DepthAdapterSkeleton):
            MODEL_KEY = "not-a-real-model"


# --------------------------------------------------------------------------- #
# Specific Image Depth / Video Depth invariants the paper relies on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key",
    # Affine-invariant Image Depth models. DepthLM was originally
    # listed here but the verified HF model card (facebook/DepthLM)
    # describes it as a metric VLM — moved to the metric group below.
    ["lotus-2", "fe2e"],
)
def test_known_relative_models_marked_relative(key):
    """Paper §3.3: these Image Depth models are affine-invariant."""
    assert DEPTH_MODEL_CARDS[key].depth_output_kind == "relative"


@pytest.mark.parametrize(
    "key",
    [
        "da3-metric-l",
        "da-v2-large",
        "depth-pro",
        "unidepth-v2",
        "metric3d-v2",
        "moge-2-vit-l",  # metric scale per official Microsoft MoGe-2 release
        "hyden",  # canonical roster selects HyDen-MoGeV2 metric point depth
        "depthlm",  # metric per huggingface.co/facebook/DepthLM
    ],
)
def test_known_metric_models_marked_metric(key):
    """Image Depth models that emit metric depth directly (no per-scene
    alignment in the runner). Verified against upstream model cards.
    """
    assert DEPTH_MODEL_CARDS[key].depth_output_kind == "metric"


def test_da3_appears_in_both_rosters():
    """DA3 is the shared model across Image Depth and Video Depth — same weights, different feed."""
    assert DEPTH_MODEL_CARDS["da3-metric-l"].task is TaskType.MONOCULAR_DEPTH
    assert DEPTH_MODEL_CARDS["da3-video"].task is TaskType.VIDEO_DEPTH

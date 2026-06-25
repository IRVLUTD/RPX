"""Tests for the canonical-roster bridge in scripts/depth_models/.

The bridge maps paper-roster kebab-case keys (``da-v2-large``) to the
legacy snake_case registry (``da_v2_metric_indoor``) so the team can
call adapters by their paper-table names via
``scripts/run_depth.py --model <canonical>``.

These tests run without GPU and without any model weights — they
exercise only the name-resolution surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from depth_models import (  # noqa: E402
    CANONICAL_TO_LEGACY,
    MODEL_REGISTRY,
    resolve_model_key,
)
from rpx_benchmark.adapters.depth_scaffold import DEPTH_MODEL_CARDS
from rpx_benchmark.api import TaskType


# --------------------------------------------------------------------------- #
# Bridge shape
# --------------------------------------------------------------------------- #


def test_every_image_depth_card_is_in_canonical_to_legacy():
    """Every Image Depth roster entry must have a bridge entry
    (mapping to a real registry adapter or to None if not implemented).
    """
    image_keys = {
        k for k, card in DEPTH_MODEL_CARDS.items()
        if card.task is TaskType.MONOCULAR_DEPTH
    }
    missing = image_keys - set(CANONICAL_TO_LEGACY)
    assert not missing, (
        f"Image Depth roster entries missing from CANONICAL_TO_LEGACY: "
        f"{sorted(missing)}. Add them to the bridge in "
        f"scripts/depth_models/__init__.py."
    )


def test_no_orphan_bridge_entries():
    """Every CANONICAL_TO_LEGACY key must be in the roster — no
    bridge entries for models the paper doesn't list.
    """
    extra = set(CANONICAL_TO_LEGACY) - set(DEPTH_MODEL_CARDS)
    assert not extra, (
        f"CANONICAL_TO_LEGACY has bridge entries for models that "
        f"aren't in DEPTH_MODEL_CARDS: {sorted(extra)}."
    )


def test_non_null_bridge_targets_resolve_to_real_adapters():
    """Every non-None bridge target must be a real key in
    MODEL_REGISTRY. A typo here would surface only at runtime; the
    test catches it at PR time.
    """
    for canonical, legacy in CANONICAL_TO_LEGACY.items():
        if legacy is None:
            continue
        assert legacy in MODEL_REGISTRY, (
            f"Bridge target {canonical!r} → {legacy!r} not in "
            f"MODEL_REGISTRY. Either fix the bridge or add the "
            f"adapter to scripts/depth_models/."
        )


# --------------------------------------------------------------------------- #
# resolve_model_key contract
# --------------------------------------------------------------------------- #


def test_resolve_canonical_returns_legacy_key():
    """The canonical → legacy mapping must round-trip via resolve."""
    assert resolve_model_key("da-v2-large") == "da_v2_metric_indoor"
    assert resolve_model_key("depth-pro") == "depth_pro"
    assert resolve_model_key("lotus-2") == "lotus_2"
    assert resolve_model_key("moge-2-vit-l") == "moge_2"


def test_resolve_legacy_key_passes_through():
    """Legacy snake_case keys must continue to work."""
    assert resolve_model_key("da_v2_metric_indoor") == "da_v2_metric_indoor"
    assert resolve_model_key("zoedepth") == "zoedepth"


def test_resolve_unknown_raises_with_options():
    """An unknown name must list both canonical and legacy options
    so the user can correct their typo without grepping.
    """
    with pytest.raises(SystemExit) as exc:
        resolve_model_key("definitely-not-a-real-model")
    msg = str(exc.value).lower()
    assert "canonical" in msg, "Error should mention canonical roster"
    assert "legacy" in msg, "Error should mention legacy registry"


def test_fe2e_now_resolves_via_safety_rail_adapter():
    """FE2E previously mapped to None (no adapter). It now has a real
    safety-rail adapter (scripts/depth_models/fe2e.py) — resolve_model_key
    should route to it. Actually invoking the adapter without the
    --acknowledge-unverified flag raises UnverifiedAdapterError; that
    layer is tested separately.
    """
    legacy = resolve_model_key("fe2e")
    assert legacy == "fe2e", (
        f"fe2e should resolve to MODEL_REGISTRY['fe2e'], got {legacy!r}"
    )
    assert "fe2e" in MODEL_REGISTRY


def test_fe2e_image_adapter_safety_rail():
    """FE2E adapter is gated by the unverified-weights safety rail.

    Building it without acknowledge_unverified=True raises
    UnverifiedAdapterError with the candidate HF path and the
    instructions for clearing the rail. Building with the flag clears
    the rail but raises NotImplementedError because no upstream model
    class is wired yet (the candidate HF repo has no code).
    """
    from depth_models import MODEL_REGISTRY
    from rpx_benchmark.exceptions import UnverifiedAdapterError

    factory = MODEL_REGISTRY["fe2e"]

    # Default: no acknowledgement → UnverifiedAdapterError.
    with pytest.raises(UnverifiedAdapterError) as exc:
        factory(device="cpu", batch_size=1)
    msg = str(exc.value)
    assert "UNVERIFIED" in msg
    assert "acknowledge_unverified" in msg
    assert "exander/FE2E" in msg  # candidate HF path surfaced

    # With acknowledgement: rail clears, but constructor still raises
    # NotImplementedError because no upstream model class is wired.
    with pytest.raises(NotImplementedError):
        factory(device="cpu", batch_size=1, acknowledge_unverified=True)

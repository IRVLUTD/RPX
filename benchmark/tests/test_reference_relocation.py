"""Tests for the `rpx_benchmark.reference` subpackage relocation.

- New canonical import paths work.
- Old paths still resolve the same objects.
- Old paths emit a ``DeprecationWarning`` when imported directly.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        (
            "rpx_benchmark.adapters.depth_hf",
            "rpx_benchmark.reference.adapters.depth_hf",
            "make_hf_depth_model",
        ),
        (
            "rpx_benchmark.adapters.depth_unidepth",
            "rpx_benchmark.reference.adapters.depth_unidepth",
            "make_unidepth_v2_model",
        ),
        (
            "rpx_benchmark.adapters.depth_metric3d",
            "rpx_benchmark.reference.adapters.depth_metric3d",
            "make_metric3d_v2_model",
        ),
        (
            "rpx_benchmark.adapters.seg_hf",
            "rpx_benchmark.reference.adapters.seg_hf",
            "make_hf_instance_seg_model",
        ),
    ],
)
def test_shim_reexports_same_object(old: str, new: str, name: str) -> None:
    """Symbol imported from old vs new path is the same underlying object."""
    new_mod = importlib.import_module(new)
    old_mod = importlib.import_module(old)
    assert getattr(old_mod, name) is getattr(new_mod, name)


@pytest.mark.parametrize(
    "old",
    [
        "rpx_benchmark.adapters.depth_hf",
        "rpx_benchmark.adapters.depth_unidepth",
        "rpx_benchmark.adapters.depth_metric3d",
        "rpx_benchmark.adapters.seg_hf",
    ],
)
def test_shim_emits_deprecation_warning(old: str) -> None:
    """Re-importing the old path surfaces the DeprecationWarning."""
    sys.modules.pop(old, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(old)
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, f"no DeprecationWarning when importing {old}"
    assert any("reference" in str(w.message) for w in deprecation)


def test_registry_resolves_reference_paths() -> None:
    """`models.registry.get_factory` resolves factories under reference.models."""
    from rpx_benchmark.models.registry import MODEL_REGISTRY, get_factory

    # Pick a factory whose module does not require heavy deps just to import.
    assert "depth_anything_v2_metric_indoor_small" in MODEL_REGISTRY
    factory = get_factory("depth_anything_v2_metric_indoor_small")
    assert callable(factory)
    assert factory.__module__ == "rpx_benchmark.reference.models.depth_anything_v2"

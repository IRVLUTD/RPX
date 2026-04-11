"""Tests for the model factory registry."""

from __future__ import annotations

import pytest

from rpx_benchmark.models.registry import (
    DEFERRED_MODELS,
    MODEL_REGISTRY,
    available_models,
    get_factory,
    resolve,
)

_EXPECTED_RUNNABLE = {
    "depth_anything_v2_metric_indoor_small",
    "depth_anything_v2_metric_indoor_base",
    "depth_anything_v2_metric_indoor_large",
    "depth_pro",
    "zoedepth_nyu",
    "unidepth_v2_vitb",
    "unidepth_v2_vitl",
    "metric3d_v2_vit_small",
    "metric3d_v2_vit_large",
    "metric3d_v2_vit_giant2",
}


def test_runnable_slate_matches_expected():
    runnable = set(available_models(include_deferred=False))
    assert _EXPECTED_RUNNABLE.issubset(runnable)
    assert runnable.isdisjoint(DEFERRED_MODELS)


def test_deferred_stubs_raise_on_resolve():
    for name in DEFERRED_MODELS:
        with pytest.raises(NotImplementedError, match=name):
            resolve(name, device="cpu")


def test_unknown_model_name_raises_config_error():
    from rpx_benchmark.exceptions import ConfigError
    with pytest.raises(ConfigError, match="Unknown model"):
        get_factory("no_such_model_anywhere")


def test_every_registered_entry_has_an_importable_module():
    """Lazy-import every entry in the registry to catch typos in module suffixes.

    We call ``get_factory`` (which imports the module and grabs the
    attribute) but we do NOT call the returned factory: that would try
    to load torch and real checkpoints.
    """
    for name in MODEL_REGISTRY:
        factory = get_factory(name)
        assert callable(factory), f"{name} factory is not callable"

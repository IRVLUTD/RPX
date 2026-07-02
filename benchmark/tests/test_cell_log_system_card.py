"""Tests for SystemCard threading through cells.parquet.

When the runner detects the host hardware via :class:`SystemCard.auto_detect`
and passes it to :func:`cells_from_per_sample`, the resulting cells must
carry ``gpu_name``, ``gpu_memory_gb``, ``precision``, and ``batch_size``
so cross-host aggregation can disambiguate them downstream. Without
this, latency numbers from different GPUs would be silently averaged
together.

Also covers the public-API contract for standalone use of the
cell-log module — third-party users who want to feed their own
per-sample metrics through ``cells_from_per_sample`` and write
parquet shouldn't need to install the whole runner stack to do it.
"""

from __future__ import annotations

import pytest

from rpx_benchmark import (
    FIXED_COLUMNS,
    SystemCard,
    cells_from_per_sample,
)
from rpx_benchmark.exceptions import ConfigError

# --------------------------------------------------------------------------- #
# SystemCard threading
# --------------------------------------------------------------------------- #


def _per_sample_rows():
    """Two scene/phase pairs with a couple of metric rows each."""
    return [
        {"scene": "scene_a", "phase": "clutter",      "absrel": 0.10, "latency_ms": 50.0},
        {"scene": "scene_a", "phase": "clutter",      "absrel": 0.12, "latency_ms": 52.0},
        {"scene": "scene_a", "phase": "interaction",  "absrel": 0.20, "latency_ms": 51.0},
        {"scene": "scene_a", "phase": "interaction",  "absrel": 0.22, "latency_ms": 53.0},
    ]


def test_system_card_dict_threaded_to_cells():
    """Passing a dict-shaped system_card stamps the four fields onto every cell."""
    cells = cells_from_per_sample(
        _per_sample_rows(),
        model_name="my-model",
        task="monocular_depth",
        system_card={
            "gpu_name": "NVIDIA A100",
            "gpu_memory_gb": 80.0,
            "precision": "fp16",
            "batch_size": 8,
        },
    )
    assert len(cells) == 2
    for cell in cells:
        assert cell["gpu_name"] == "NVIDIA A100"
        assert cell["gpu_memory_gb"] == 80.0
        assert cell["precision"] == "fp16"
        assert cell["batch_size"] == 8


def test_system_card_dataclass_accepted_directly():
    """Public API ergonomics: accept the SystemCard dataclass without
    requiring the caller to .to_dict() it. Third-party pip users
    typically have the dataclass in hand from SystemCard.auto_detect().
    """
    card = SystemCard(
        gpu_name="RTX 4090",
        gpu_memory_gb=24.0,
        precision="bf16",
        batch_size=4,
    )
    cells = cells_from_per_sample(
        _per_sample_rows(),
        model_name="my-model",
        task="monocular_depth",
        system_card=card,
    )
    for cell in cells:
        assert cell["gpu_name"] == "RTX 4090"
        assert cell["gpu_memory_gb"] == 24.0
        assert cell["precision"] == "bf16"
        assert cell["batch_size"] == 4


def test_no_system_card_leaves_fields_none_but_present():
    """The columns are always in the schema (so downstream readers
    don't error on missing columns), but their values are None when
    the caller didn't supply a card. This matches the parquet contract.
    """
    cells = cells_from_per_sample(
        _per_sample_rows(),
        model_name="my-model",
        task="monocular_depth",
    )
    for cell in cells:
        assert "gpu_name" in cell and cell["gpu_name"] is None
        assert "gpu_memory_gb" in cell and cell["gpu_memory_gb"] is None
        assert "precision" in cell and cell["precision"] is None
        assert "batch_size" in cell and cell["batch_size"] is None


def test_invalid_system_card_type_raises_clear_error():
    """Passing something that's neither a SystemCard nor a dict should
    surface a clear ConfigError naming what was passed, not crash deep
    in the bucketing loop with a misleading attribute error.
    """
    with pytest.raises(ConfigError, match="SystemCard, a dict, or None"):
        cells_from_per_sample(
            _per_sample_rows(),
            model_name="my-model",
            task="monocular_depth",
            system_card="A100 80GB",  # string is wrong
        )


# --------------------------------------------------------------------------- #
# Schema invariants
# --------------------------------------------------------------------------- #


def test_fixed_columns_includes_hardware_fields():
    """The schema contract every downstream reader (Φ / J aggregation,
    paper-table fill, dashboards) depends on must include the four
    hardware fields. Asserted here so any future column-shuffle PR
    that drops them lights up the test suite immediately.
    """
    for required in ("gpu_name", "gpu_memory_gb", "precision", "batch_size"):
        assert required in FIXED_COLUMNS, (
            f"FIXED_COLUMNS missing required hardware field {required!r}. "
            "Cross-host cell aggregation needs all four."
        )


# --------------------------------------------------------------------------- #
# Public-API contract — third-party standalone use
# --------------------------------------------------------------------------- #


def test_cell_log_api_is_public():
    """Verify the cell-log entry points are exported from the
    top-level package so a third-party pip user can do
    ``from rpx_benchmark import cells_from_per_sample, write_cells``
    without reaching into private submodules.
    """
    import rpx_benchmark as rpx

    for name in (
        "FIXED_COLUMNS",
        "cells_from_per_sample",
        "cell_from_metrics",
        "metric_col",
        "metric_keys",
        "read_cells",
        "write_cells",
        "SystemCard",
    ):
        assert hasattr(rpx, name), (
            f"rpx_benchmark.{name} not exported — third-party users "
            f"can't reach it via the public API."
        )

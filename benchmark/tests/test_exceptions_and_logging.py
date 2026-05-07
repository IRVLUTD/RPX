"""Tests for the exception hierarchy and logging utilities."""

from __future__ import annotations

import logging

import pytest

import rpx_benchmark as rpx
from rpx_benchmark.exceptions import (
    AdapterError,
    ConfigError,
    DatasetError,
    DownloadError,
    ManifestError,
    MetricError,
    ModelError,
    RPXError,
)
from rpx_benchmark.logging_utils import configure_logging, get_logger


def test_exception_hierarchy():
    assert issubclass(ConfigError, RPXError)
    assert issubclass(DatasetError, RPXError)
    assert issubclass(ManifestError, DatasetError)
    assert issubclass(DownloadError, DatasetError)
    assert issubclass(ModelError, RPXError)
    assert issubclass(AdapterError, ModelError)
    assert issubclass(MetricError, RPXError)


def test_error_carries_hint_and_details():
    e = RPXError("a thing went wrong", hint="try X", details={"field": "foo"})
    assert "a thing went wrong" in str(e)
    assert "try X" in str(e)
    assert e.details == {"field": "foo"}


def test_all_exceptions_reexported_from_package():
    assert rpx.RPXError is RPXError
    assert rpx.ConfigError is ConfigError
    assert rpx.ManifestError is ManifestError
    assert rpx.DownloadError is DownloadError
    assert rpx.ModelError is ModelError


def test_monocular_depth_config_raises_config_error_on_no_model():
    with pytest.raises(ConfigError, match="model is required"):
        rpx.MonocularDepthRunConfig(split="hard")


def test_monocular_depth_config_rejects_bad_split():
    import numpy as np

    bm = rpx.make_numpy_depth_model(lambda rgb: np.zeros(rgb.shape[:2], dtype=np.float32))
    with pytest.raises(ConfigError, match="Unknown split"):
        rpx.MonocularDepthRunConfig(model=bm, split="super-hard")


def test_monocular_depth_config_rejects_zero_batch_size():
    import numpy as np

    bm = rpx.make_numpy_depth_model(lambda rgb: np.zeros(rgb.shape[:2], dtype=np.float32))
    with pytest.raises(ConfigError, match="batch_size"):
        rpx.MonocularDepthRunConfig(model=bm, split="hard", batch_size=0)


def test_get_logger_nests_under_rpx_benchmark_root():
    log = get_logger("rpx_benchmark.hub")
    assert log.name == "rpx_benchmark.hub"

    log2 = get_logger("foo")
    assert log2.name == "rpx_benchmark.foo"


def test_configure_logging_is_idempotent():
    root1 = configure_logging("INFO")
    handler_count_1 = len(root1.handlers)
    root2 = configure_logging("INFO")
    assert len(root2.handlers) == handler_count_1
    root3 = configure_logging("DEBUG", force=True)
    assert len(root3.handlers) == 1


def test_configure_logging_respects_env_var(monkeypatch):
    monkeypatch.setenv("RPX_LOG_LEVEL", "DEBUG")
    root = configure_logging("INFO", force=True)
    assert root.level == logging.DEBUG


# --------------------------------------------------------------------------- #
# Invariant: library code never raises bare stdlib exceptions.
# --------------------------------------------------------------------------- #


def test_no_bare_stdlib_raises_in_library():
    """Every ``raise`` in rpx_benchmark/ must use an RPXError subclass.

    Lock-in test so a future refactor doesn't accidentally re-introduce
    bare ValueError / RuntimeError / KeyError raises. Scans the source
    tree for the pattern; excludes docstrings/comments by virtue of
    the leading-whitespace match.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "rpx_benchmark"
    assert root.is_dir(), f"rpx_benchmark package not found at {root}"

    bad_pattern = re.compile(
        r"^\s*raise\s+(ValueError|RuntimeError|TypeError|KeyError|"
        r"AssertionError|Exception)\b"
    )
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if bad_pattern.match(line):
                offenders.append(f"{py.relative_to(root.parent)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found bare stdlib raises in library code. Use RPXError "
        "subclasses with a `hint`:\n  " + "\n  ".join(offenders)
    )

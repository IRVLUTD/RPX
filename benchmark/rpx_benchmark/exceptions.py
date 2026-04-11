"""Exception hierarchy for the RPX benchmark toolkit.

Every error raised by ``rpx_benchmark`` public APIs is a subclass of
:class:`RPXError`, so callers can write::

    import rpx_benchmark as rpx

    try:
        result, report, _ = rpx.run_monocular_depth(cfg)
    except rpx.exceptions.ConfigError as e:
        # bad config: show the user what to fix
        print(f"[config] {e}")
    except rpx.exceptions.DownloadError:
        print("offline — re-run with --cache-dir pointing at a local mirror")
    except rpx.RPXError as e:
        # any other benchmark-specific failure
        print(f"[rpx] {e}")

The hierarchy is deliberately narrow: library code raises the most
specific subclass that applies, and each subclass carries a
human-readable message explaining *what* went wrong and *what the user
can do* to fix it.

Hierarchy
---------

.. code-block:: text

    RPXError
    ├── ConfigError          invalid MonocularDepthRunConfig / CLI flag
    ├── DatasetError
    │   ├── ManifestError    malformed or missing manifest file
    │   └── DownloadError    HuggingFace or network failure
    ├── ModelError
    │   └── AdapterError     input/output adapter or factory error
    └── MetricError          evaluator or metric calculation error
"""

from __future__ import annotations

from typing import Any, Optional


class RPXError(Exception):
    """Base exception for every error raised by the RPX benchmark toolkit.

    All library code raises a subclass of this so user code can write a
    single ``except RPXError`` to catch all benchmark failures without
    accidentally swallowing unrelated exceptions.

    Parameters
    ----------
    message : str
        Human-readable description of what failed. Should include a
        hint about what the user can do next.
    hint : str, optional
        Additional remediation advice rendered after the main message.
    details : dict, optional
        Structured context (e.g. offending field, expected value) that
        higher-level code may inspect.

    Examples
    --------
    >>> from rpx_benchmark.exceptions import RPXError
    >>> raise RPXError("something went wrong", hint="check the cache dir")
    ... # doctest: +SKIP
    """

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.hint = hint
        self.details = dict(details) if details else {}
        full = message
        if hint:
            full = f"{message}\n  hint: {hint}"
        super().__init__(full)


# --------------------------------------------------------------------------- #
# Config-level errors
# --------------------------------------------------------------------------- #

class ConfigError(RPXError):
    """Raised when a user-supplied config is invalid.

    Examples
    --------
    * ``MonocularDepthRunConfig`` built with both ``model`` and
      ``hf_checkpoint`` set.
    * CLI given ``--device cuda`` on a CPU-only host with
      ``--strict-device`` enabled.
    * Unknown difficulty split.
    """


# --------------------------------------------------------------------------- #
# Dataset-level errors
# --------------------------------------------------------------------------- #

class DatasetError(RPXError):
    """Base class for dataset load / manifest / download failures."""


class ManifestError(DatasetError):
    """Manifest JSON is missing, malformed, or references missing files.

    Raised by :class:`rpx_benchmark.loader.RPXDataset` when the loader
    cannot resolve a sample from the manifest it was handed.

    Examples
    --------
    * Manifest missing the ``task`` field.
    * Sample lists ``rgb`` that does not exist on disk.
    * Task value is not in :class:`rpx_benchmark.api.TaskType`.
    """


class DownloadError(DatasetError):
    """HuggingFace download or cache lookup failed.

    Raised by :mod:`rpx_benchmark.hub` when ``snapshot_download`` or
    ``hf_hub_download`` fails (network issue, bad repo id, missing
    revision, permission denied).
    """


# --------------------------------------------------------------------------- #
# Model-level errors
# --------------------------------------------------------------------------- #

class ModelError(RPXError):
    """Raised by model factories or the runner when a model misbehaves.

    Examples
    --------
    * Model returned a different number of predictions than samples.
    * Model's ``task`` attribute does not match the dataset task.
    * Prediction dataclass has a wrong shape.
    """


class AdapterError(ModelError):
    """Input or output adapter produced an invalid payload.

    Examples
    --------
    * ``InputAdapter.prepare`` raised during preprocessing.
    * ``OutputAdapter.finalize`` returned a non-``DepthPrediction`` for
      the monocular depth task.
    * HF processor's ``post_process_depth_estimation`` signature does
      not accept the kwargs the adapter wants to pass.
    """


# --------------------------------------------------------------------------- #
# Metric-level errors
# --------------------------------------------------------------------------- #

class MetricError(RPXError):
    """Raised when a metric calculator cannot compute a score.

    Examples
    --------
    * Prediction dataclass is the wrong type for the task.
    * Ground-truth shape does not match prediction shape.
    * Unknown metric name requested from a registry.
    """


__all__ = [
    "RPXError",
    "ConfigError",
    "DatasetError",
    "ManifestError",
    "DownloadError",
    "ModelError",
    "AdapterError",
    "MetricError",
]

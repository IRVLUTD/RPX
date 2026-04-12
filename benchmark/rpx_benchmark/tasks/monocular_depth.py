"""Monocular absolute depth benchmark pipeline.

Reference implementation for a task module. Every other task follows
exactly the same structure: subclass :class:`TaskRunConfig`, define a
model resolver, wire a :class:`TaskSpec`. The heavy lifting lives in
:mod:`rpx_benchmark.tasks._pipeline` so each task file stays small.

Three usage paths
-----------------

**1. Zero-code — any HuggingFace depth checkpoint via the CLI**::

    rpx bench monocular_depth \\
        --hf-checkpoint depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \\
        --split hard

**2. One-line Python — a plain numpy callable**::

    import numpy as np
    import rpx_benchmark as rpx

    def my_depth(rgb: np.ndarray) -> np.ndarray:
        return ...  # H x W float32, metres

    bm = rpx.make_numpy_depth_model(my_depth, name="my_model")
    cfg = rpx.MonocularDepthRunConfig(model=bm, split="hard", device="cpu")
    result, report, paths = rpx.run_monocular_depth(cfg)

**3. Custom adapter stack** — provide your own
:class:`~rpx_benchmark.adapters.BenchmarkableModel` and hand it to
the config via ``model=``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..adapters import BenchmarkableModel
from ..api import Difficulty, TaskType
from ..logging_utils import get_logger
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

log = get_logger(__name__)

#: Primary metric that drives ESD-weighted phase scoring.
#: AbsRel is lower-is-better, so ``delta_int < 0`` means the model
#: *degrades* on the interaction phase — the key RPX failure mode.
PRIMARY_METRIC = "absrel"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class MonocularDepthRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_monocular_depth`.

    Inherits every standard field from :class:`TaskRunConfig`. Adds
    no task-specific fields — monocular depth is the "base case" a
    new user encounters.

    Examples
    --------
    >>> from rpx_benchmark import MonocularDepthRunConfig
    >>> cfg = MonocularDepthRunConfig(
    ...     hf_checkpoint="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    ...     split="hard",
    ...     device="cpu",
    ... )
    """

    def __post_init__(self) -> None:
        self._validate_common()


# --------------------------------------------------------------------------- #
# Model resolver
# --------------------------------------------------------------------------- #

def _resolve_model(cfg: MonocularDepthRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        log.info("resolving model %r from registry", cfg.model_name)
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    from ..reference.adapters.depth_hf import make_hf_depth_model
    log.info("resolving HF checkpoint %r", cfg.hf_checkpoint)
    return make_hf_depth_model(
        cfg.hf_checkpoint, device=cfg.device, **cfg.model_kwargs,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_monocular_depth(cfg: MonocularDepthRunConfig) -> PipelineResult:
    """Run the monocular absolute depth benchmark end-to-end.

    Parameters
    ----------
    cfg : MonocularDepthRunConfig

    Returns
    -------
    (BenchmarkResult, DeploymentReadinessReport, dict)
        ``dict`` has ``"json"`` + ``"markdown"`` keys pointing at
        the written report files.

    Raises
    ------
    ConfigError, DownloadError, ManifestError, ModelError, MetricError
        Propagated from the respective subsystem with actionable hints.

    Examples
    --------
    >>> import rpx_benchmark as rpx
    >>> import numpy as np
    >>> def my_depth(rgb): return np.ones(rgb.shape[:2], dtype=np.float32) * 2.0
    >>> bm = rpx.make_numpy_depth_model(my_depth, name="unit")
    >>> cfg = rpx.MonocularDepthRunConfig(model=bm, split="hard", device="cpu")
    >>> result, report, paths = rpx.run_monocular_depth(cfg)  # doctest: +SKIP
    """
    return run_pipeline(
        task=TaskType.MONOCULAR_DEPTH,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=True,
        compute_sgc=False,
    )


# --------------------------------------------------------------------------- #
# CLI glue
# --------------------------------------------------------------------------- #

def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    from ..models.registry import available_models

    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        choices=available_models(),
        help="Registered adapter name (run `rpx models` to list).",
    )
    model_group.add_argument(
        "--hf-checkpoint",
        help="HuggingFace checkpoint id, e.g. 'apple/DepthPro-hf'.",
    )
    p.add_argument(
        "--split", required=True,
        choices=[d.value for d in Difficulty],
        help="ESD difficulty split.",
    )
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> MonocularDepthRunConfig:
    return MonocularDepthRunConfig(
        model_name=args.model,
        hf_checkpoint=args.hf_checkpoint,
        split=args.split,
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        revision=args.revision,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


# --------------------------------------------------------------------------- #
# Self-registration
# --------------------------------------------------------------------------- #

def _temporal_stability_hook(predictions, samples, camera_poses):
    """Depth-specific Temporal Stability hook registered on the TaskSpec.

    Extracted so the benchmark runner can call it uniformly across
    tasks without branching on task identity.
    """
    from ..deployment import compute_temporal_stability_depth
    depths = [p.depth_map for p in predictions]
    return compute_temporal_stability_depth(depths, camera_poses)


TASK_SPEC = TaskSpec(
    task=TaskType.MONOCULAR_DEPTH,
    display_name="Monocular Absolute Depth",
    description=(
        "Predict metric depth in metres from a single RGB frame; "
        "evaluated at the subset of pixels with valid D435 ground truth."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth"],
    higher_is_better=False,
    build_config=_build_config,
    run=run_monocular_depth,
    add_cli_arguments=_add_cli_arguments,
    temporal_stability_fn=_temporal_stability_hook,
)

register_task(TASK_SPEC)

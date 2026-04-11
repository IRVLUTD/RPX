"""Monocular absolute depth benchmark pipeline.

This module is the reference implementation for a task in the RPX
benchmark toolkit. Every other task (segmentation, tracking, visual
grounding, ...) can be written by cloning this file, swapping in the
correct ground-truth + prediction types, adjusting the primary metric,
and registering via :func:`rpx_benchmark.tasks.registry.register_task`.

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
:class:`~rpx_benchmark.adapters.BenchmarkableModel` and hand it to the
config via ``model=``.

The pipeline itself is five steps:

1. Download the (task, split) slice from HuggingFace via
   :func:`rpx_benchmark.hub.download_split` (RGB + depth only).
2. Build an :class:`~rpx_benchmark.loader.RPXDataset` over the
   resolved manifest.
3. Resolve the :class:`BenchmarkableModel`: ``cfg.model`` takes
   precedence, then ``cfg.model_name`` (registry lookup), then
   ``cfg.hf_checkpoint`` (``make_hf_depth_model`` fast path).
4. Run :class:`BenchmarkRunner` with deployment-readiness enabled so
   the report has ESD-weighted phase scores, STR, TS, FLOPs, latency.
5. Write JSON + markdown reports under
   ``./rpx_results/<model>/<split>/`` (override with
   ``cfg.output_dir``).

Errors
------

Every failure path raises a subclass of
:class:`rpx_benchmark.exceptions.RPXError`:

- :class:`~rpx_benchmark.exceptions.ConfigError` for invalid configs.
- :class:`~rpx_benchmark.exceptions.DownloadError` for HF failures.
- :class:`~rpx_benchmark.exceptions.ModelError` /
  :class:`~rpx_benchmark.exceptions.AdapterError` for model issues.
- :class:`~rpx_benchmark.exceptions.MetricError` for evaluation issues.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..adapters import BenchmarkableModel
from ..api import Difficulty, TaskType
from ..deployment import DeploymentReadinessReport
from ..exceptions import ConfigError
from ..hub import download_split
from ..loader import RPXDataset
from ..logging_utils import get_logger
from ..metrics.registry import BenchmarkResult, MetricSuite
from ..profiler import EfficiencyMetadata, count_parameters
from ..reports import format_markdown_summary, write_json
from ..runner import BenchmarkRunner, ProgressCallback
from .registry import TaskSpec, register_task

log = get_logger(__name__)

#: Primary metric used to drive ESD-weighted phase scoring.
#: Lower is better, so ``delta_int`` negative = model degrades on
#: interaction phase.
PRIMARY_METRIC = "absrel"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class MonocularDepthRunConfig:
    """Runtime configuration for :func:`run_monocular_depth`.

    Exactly one of ``model``, ``model_name``, ``hf_checkpoint`` must be
    provided. Everything else has sensible defaults.

    Parameters
    ----------
    model : BenchmarkableModel, optional
        A fully-constructed :class:`BenchmarkableModel` instance.
        Takes precedence over the other model selectors.
    model_name : str, optional
        Name registered in :mod:`rpx_benchmark.models.registry`
        (e.g. ``"depth_pro"``).
    hf_checkpoint : str, optional
        HuggingFace Hub id of any checkpoint loadable via
        ``transformers.AutoModelForDepthEstimation``
        (e.g. ``"apple/DepthPro-hf"``).
    split : Difficulty or str
        ESD difficulty split to evaluate on. Default
        :attr:`Difficulty.HARD`.
    repo_id : str, optional
        HuggingFace dataset repo id hosting RPX. Defaults to
        :data:`rpx_benchmark.hub.DEFAULT_REPO_ID`.
    cache_dir : str, optional
        Local HF cache directory. Defaults to ``~/.cache/huggingface``.
    revision : str, optional
        HF revision/tag/branch to pin.
    batch_size : int
        Dataset batch size. Default 1.
    device : str
        PyTorch device string. Auto-falls-back to ``"cpu"`` if
        ``"cuda"`` is requested but not available.
    output_dir : str, optional
        Where to write ``result.json`` + ``summary.md``. Defaults to
        ``./rpx_results/<model>/<split>/``.
    model_kwargs : dict
        Extra kwargs forwarded to the model factory when resolving
        from ``model_name`` or ``hf_checkpoint``.
    progress : ProgressCallback, optional
        Per-sample progress callback the runner will invoke.

    Raises
    ------
    ConfigError
        If fewer or more than one of the model selectors are set, or
        if ``split`` is not a valid difficulty.

    Examples
    --------
    >>> from rpx_benchmark import MonocularDepthRunConfig
    >>> cfg = MonocularDepthRunConfig(
    ...     hf_checkpoint="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    ...     split="hard",
    ...     device="cpu",
    ... )
    """

    # Model specification — exactly one of these must be provided.
    model: Optional[BenchmarkableModel] = None
    model_name: Optional[str] = None
    hf_checkpoint: Optional[str] = None

    # Dataset selection.
    split: Difficulty | str = Difficulty.HARD
    repo_id: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    batch_size: int = 1

    # Runtime.
    device: str = "cuda"
    output_dir: Optional[str] = None

    # Extras passed to the model factory (when resolving from name).
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Optional progress hook used by the runner (CLI passes one in).
    progress: Optional[ProgressCallback] = None

    def __post_init__(self) -> None:
        given = [x for x in (self.model, self.model_name, self.hf_checkpoint) if x]
        if len(given) != 1:
            raise ConfigError(
                "MonocularDepthRunConfig requires exactly one of "
                "`model`, `model_name`, or `hf_checkpoint`.",
                hint=(
                    "Pick a registered model ('rpx models' to list) or "
                    "pass a HuggingFace checkpoint id like "
                    "'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf'."
                ),
                details={"given_fields": [
                    k for k, v in (
                        ("model", self.model),
                        ("model_name", self.model_name),
                        ("hf_checkpoint", self.hf_checkpoint),
                    ) if v
                ]},
            )

        # Normalise split into a Difficulty enum up front so downstream
        # comparisons are stable.
        if isinstance(self.split, str):
            try:
                self.split = Difficulty(self.split)
            except ValueError as e:
                raise ConfigError(
                    f"Unknown difficulty split {self.split!r}.",
                    hint=f"Use one of: {[d.value for d in Difficulty]}",
                ) from e

        if self.batch_size < 1:
            raise ConfigError(
                f"batch_size must be >= 1, got {self.batch_size}",
            )

        # Device fallback is applied in run_monocular_depth so the
        # config object stays hashable/immutable until run time.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _resolve_device(requested: str) -> str:
    """Return ``requested`` unless CUDA was asked for but is unavailable.

    Falls back to CPU with an ``INFO`` log line. Torch is an optional
    dependency; if it is not importable, the original string is
    returned unchanged.
    """
    if requested != "cuda":
        return requested
    try:
        import torch
    except ImportError:
        return requested
    if torch.cuda.is_available():
        return requested
    log.warning(
        "CUDA requested but torch.cuda.is_available() is False; "
        "falling back to device='cpu'."
    )
    return "cpu"


def _resolve_model(cfg: MonocularDepthRunConfig) -> BenchmarkableModel:
    """Turn the user's selector choice into a concrete ``BenchmarkableModel``."""
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        log.info("resolving model %r from registry", cfg.model_name)
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    from ..adapters.depth_hf import make_hf_depth_model
    log.info("resolving HF checkpoint %r", cfg.hf_checkpoint)
    return make_hf_depth_model(
        cfg.hf_checkpoint, device=cfg.device, **cfg.model_kwargs
    )


def _display_name(cfg: MonocularDepthRunConfig, model: BenchmarkableModel) -> str:
    if cfg.model_name:
        return cfg.model_name
    if cfg.hf_checkpoint:
        return cfg.hf_checkpoint
    return getattr(model, "name", "model")


def _count_params_only(model: BenchmarkableModel) -> EfficiencyMetadata:
    """Static parameter count; FLOPs + latency are measured by the runner.

    We deliberately do *not* run a dummy forward pass here. HF depth
    models expect patch-aligned preprocessed inputs so a raw
    ``(1, 3, 480, 640)`` tensor would fail shape checks and throw.
    The runner wraps the first real inference in ``FlopCounterMode``
    so the count reflects the adapter's actual preprocessing pipeline.
    """
    raw = getattr(model, "model", None)
    if raw is None or not hasattr(raw, "parameters"):
        return EfficiencyMetadata(
            model_type="local",
            notes="non-torch callable; only latency is available",
        )
    return EfficiencyMetadata(
        params_m=count_parameters(raw),
        model_type="local",
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_monocular_depth(
    cfg: MonocularDepthRunConfig,
) -> Tuple[BenchmarkResult, DeploymentReadinessReport, Dict[str, Path]]:
    """Run the monocular absolute depth benchmark end-to-end.

    Parameters
    ----------
    cfg : MonocularDepthRunConfig
        See :class:`MonocularDepthRunConfig` for field docs.

    Returns
    -------
    (BenchmarkResult, DeploymentReadinessReport, dict)
        - ``BenchmarkResult``: per-sample + aggregated metrics.
        - ``DeploymentReadinessReport``: WPS, STR, TS, FLOPs, latency.
        - ``dict[str, Path]``: ``{"json": ..., "markdown": ...}``
          pointing at the written report files.

    Raises
    ------
    ConfigError
        For invalid configs (caught at :meth:`__post_init__`).
    DownloadError
        For HF download failures.
    ModelError
        For model-side failures (factory errors, shape mismatches).
    MetricError
        For evaluation-side failures.

    Examples
    --------
    >>> import rpx_benchmark as rpx
    >>> import numpy as np
    >>> def my_depth(rgb): return np.ones(rgb.shape[:2], dtype=np.float32) * 2.0
    >>> bm = rpx.make_numpy_depth_model(my_depth, name="unit")
    >>> cfg = rpx.MonocularDepthRunConfig(model=bm, split="hard", device="cpu")
    >>> result, report, paths = rpx.run_monocular_depth(cfg)  # doctest: +SKIP
    >>> print(result.aggregated["absrel"])                    # doctest: +SKIP
    """
    task = TaskType.MONOCULAR_DEPTH

    # 0. Apply CUDA fallback once, at run-time, so failing hosts do not
    # download weights only to crash at .to('cuda').
    cfg.device = _resolve_device(cfg.device)

    log.info("task=%s  split=%s  device=%s",
             task.value,
             cfg.split.value if isinstance(cfg.split, Difficulty) else cfg.split,
             cfg.device)

    # 1. Download the split slice (RGB + depth only).
    manifest_path = download_split(
        task=task,
        split=cfg.split,
        repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
        cache_dir=cfg.cache_dir,
        revision=cfg.revision,
    )
    dataset = RPXDataset.from_manifest(manifest_path, batch_size=cfg.batch_size)
    log.info("loaded %d samples from %s", len(dataset), manifest_path)

    # 2. Resolve the model (name / checkpoint / pre-built object).
    model = _resolve_model(cfg)
    display_name = _display_name(cfg, model)

    # 3. Static param count only; flops + latency come from the runner's
    # first-batch FlopCounterMode wrap and per-batch timing.
    model.setup()
    efficiency = _count_params_only(model)
    if efficiency.params_m is not None:
        log.info("model %s: %.2f M params", display_name, efficiency.params_m)

    # 4. Run with deployment-readiness. The runner owns FLOPs counting
    # and per-sample latency (median over everything except the warmup
    # batch).
    runner = BenchmarkRunner(
        model=model,
        dataset=dataset,
        metric_suite=MetricSuite.for_task(task),
        call_setup=False,
    )
    result, dr_report = runner.run_with_deployment_readiness(
        primary_metric=PRIMARY_METRIC,
        model_name=display_name,
        efficiency=efficiency,
        compute_ts=True,
        compute_sgc_flag=False,
        progress=cfg.progress,
    )

    # 5. Write reports.
    split_name = cfg.split.value if isinstance(cfg.split, Difficulty) else str(cfg.split)
    safe_name = display_name.replace("/", "__")
    out_dir = Path(cfg.output_dir or f"./rpx_results/{safe_name}/{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"

    write_json(
        json_path,
        task=task.value,
        model_name=display_name,
        split=split_name,
        repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
        result=result,
        dr_report=dr_report,
    )
    md_path.write_text(
        format_markdown_summary(
            task=task.value,
            model_name=display_name,
            split=split_name,
            repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
            result=result,
            dr_report=dr_report,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s and %s", json_path, md_path)

    return result, dr_report, {"json": json_path, "markdown": md_path}


# --------------------------------------------------------------------------- #
# CLI glue
# --------------------------------------------------------------------------- #

def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    """Populate an argparse subparser for ``rpx bench monocular_depth``."""
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
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID,
                   help="HuggingFace dataset repo id.")
    p.add_argument("--cache-dir", default=None,
                   help="HF cache directory (defaults to ~/.cache/huggingface).")
    p.add_argument("--revision", default=None,
                   help="HF revision/tag/branch.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Dataset batch size (default 1).")
    p.add_argument("--device", default="cuda",
                   help="Torch device. Auto-falls-back to CPU if unavailable.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write result.json + summary.md.")


def _build_config(args: argparse.Namespace) -> MonocularDepthRunConfig:
    """Turn an argparse Namespace into a :class:`MonocularDepthRunConfig`."""
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
)

register_task(TASK_SPEC)

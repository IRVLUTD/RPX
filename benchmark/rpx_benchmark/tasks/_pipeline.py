"""Shared plumbing for RPX task-level pipelines.

Every task runner under :mod:`rpx_benchmark.tasks` executes the same
five-step flow — download the split, load the dataset, resolve the
model, run the benchmark runner, write JSON + markdown reports.
That flow lives here so each task module only has to describe what
is *task-specific* (config subclass, model resolver, primary metric,
CLI argument parser) and delegate the boilerplate.

Design
------

- :class:`TaskRunConfig` owns every field that every task's run
  function takes. Subclass it with ``@dataclass`` when a task wants
  extra fields.
- :func:`validate_model_selector` enforces the three-way "exactly
  one of ``model`` / ``model_name`` / ``hf_checkpoint``" contract
  shared by every task.
- :func:`normalise_split` turns a string like ``"hard"`` into
  :class:`Difficulty.HARD` at config construction time.
- :func:`resolve_device` applies the CUDA-available fallback before
  any weights are downloaded.
- :func:`run_pipeline` is the entry point every task runner calls.
  It takes the task enum, the config, a primary metric key, and a
  task-specific callable that turns the config into a
  :class:`BenchmarkableModel`.

Refactor history: the original ``run_monocular_depth`` and
``run_segmentation`` each duplicated ~250 lines of plumbing. Adding
a new task used to mean cloning + editing a big file; now it is
roughly 80 lines of subclass + resolver + registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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

log = get_logger(__name__)

#: Convenience type alias for the tuple every task runner returns.
PipelineResult = Tuple[BenchmarkResult, DeploymentReadinessReport, Dict[str, Path]]


# --------------------------------------------------------------------------- #
# Shared configuration dataclass
# --------------------------------------------------------------------------- #

@dataclass
class TaskRunConfig:
    """Base configuration shared by every task runner.

    Task-specific config classes should inherit from this dataclass
    and add their own fields. The common fields below are consumed
    by :func:`run_pipeline` and cover the download, load, model
    resolution, and reporting steps.

    Parameters
    ----------
    model : BenchmarkableModel, optional
        A fully-constructed model object. Takes precedence over the
        other two selectors when provided.
    model_name : str, optional
        Name of a registered model factory (run ``rpx models`` to
        list the current slate).
    hf_checkpoint : str, optional
        HuggingFace Hub checkpoint id. Only meaningful when the task
        runner ships an HF fast path (currently
        :func:`rpx_benchmark.adapters.depth_hf.make_hf_depth_model`
        for depth and
        :func:`rpx_benchmark.adapters.seg_hf.make_hf_instance_seg_model`
        for segmentation).
    split : Difficulty or str
        ESD difficulty split to evaluate on. String values are
        normalised to :class:`Difficulty` at config construction
        time.
    repo_id : str, optional
        HuggingFace dataset repo hosting RPX. Defaults to the value
        of :data:`rpx_benchmark.hub.DEFAULT_REPO_ID` when not
        provided.
    cache_dir : str, optional
        HuggingFace cache directory override.
    revision : str, optional
        HF revision (branch / tag) to pin.
    batch_size : int
        Dataset batch size. Default 1.
    device : str
        Torch device string. Auto-falls back to ``"cpu"`` at run
        time when ``"cuda"`` is requested but unavailable.
    output_dir : str, optional
        Directory to write ``result.json`` and ``summary.md`` into.
        Defaults to ``./rpx_results/<model>/<split>/``.
    model_kwargs : dict
        Extra kwargs forwarded to the model factory when resolving
        from ``model_name`` or ``hf_checkpoint``.
    progress : ProgressCallback, optional
        Per-sample progress callback the runner will invoke.

    Raises
    ------
    ConfigError
        Subclasses must call :meth:`_validate_common` from their
        ``__post_init__`` to enforce the selector / split /
        batch-size invariants.
    """

    # Model selectors — exactly one must be provided at run time.
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

    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[ProgressCallback] = None

    def _validate_common(self) -> None:
        """Check the three shared invariants: selector, split, batch size.

        Subclasses override ``__post_init__`` to call this plus any
        extra validation they need.
        """
        validate_model_selector(self)
        self.split = normalise_split(self.split)
        if self.batch_size < 1:
            raise ConfigError(
                f"batch_size must be >= 1, got {self.batch_size}",
            )


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #

def validate_model_selector(cfg: TaskRunConfig) -> None:
    """Raise :class:`ConfigError` unless exactly one selector is set.

    Called from every task config's ``__post_init__``. The three
    selectors (``model``, ``model_name``, ``hf_checkpoint``) are
    mutually exclusive — supplying zero or two+ is a user error.
    """
    given = [x for x in (cfg.model, cfg.model_name, cfg.hf_checkpoint) if x]
    if len(given) != 1:
        raise ConfigError(
            f"{type(cfg).__name__} requires exactly one of "
            "`model`, `model_name`, or `hf_checkpoint`.",
            hint=(
                "Pick a registered model (`rpx models` lists the slate), "
                "a HuggingFace checkpoint id, or a fully-constructed "
                "BenchmarkableModel instance."
            ),
            details={"given_fields": [
                k for k, v in (
                    ("model", cfg.model),
                    ("model_name", cfg.model_name),
                    ("hf_checkpoint", cfg.hf_checkpoint),
                ) if v
            ]},
        )


def normalise_split(split: Difficulty | str) -> Difficulty:
    """Turn a string split into a :class:`Difficulty` enum.

    Raises :class:`ConfigError` with a usable hint when the string
    does not match one of the three ESD difficulty levels.
    """
    if isinstance(split, Difficulty):
        return split
    try:
        return Difficulty(split)
    except ValueError as e:
        raise ConfigError(
            f"Unknown difficulty split {split!r}.",
            hint=f"Use one of: {[d.value for d in Difficulty]}",
        ) from e


def resolve_device(requested: str) -> str:
    """Return ``requested`` or fall back to ``'cpu'`` when CUDA is unavailable.

    Emits a ``WARNING`` log line on fallback so users see why their
    ``--device cuda`` request quietly landed on CPU.
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
        "falling back to device='cpu'.",
    )
    return "cpu"


# --------------------------------------------------------------------------- #
# Profiling helper
# --------------------------------------------------------------------------- #

def _count_params_only(model: BenchmarkableModel) -> EfficiencyMetadata:
    """Static parameter count; FLOPs + latency come from the runner.

    We deliberately do not run a dummy forward pass here because HF
    and native models each expect different preprocessed inputs.
    The runner wraps the first real inference in
    :class:`torch.utils.flop_counter.FlopCounterMode` so the FLOP
    count reflects the adapter's actual preprocessing pipeline.
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


def display_name(
    cfg: TaskRunConfig,
    model: BenchmarkableModel,
) -> str:
    """Pick the best human name for a model across the three selectors."""
    if cfg.model_name:
        return cfg.model_name
    if cfg.hf_checkpoint:
        return cfg.hf_checkpoint
    return getattr(model, "name", "model")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(
    *,
    task: TaskType,
    primary_metric: str,
    cfg: TaskRunConfig,
    model_resolver: Callable[[TaskRunConfig], BenchmarkableModel],
    compute_ts: bool = True,
    compute_sgc: bool = False,
) -> PipelineResult:
    """Execute a task end-to-end.

    Parameters
    ----------
    task : TaskType
        The task enum. Used for the HuggingFace download scope and
        for metric suite construction.
    primary_metric : str
        Metric key that drives the ESD-weighted phase score. For
        lower-is-better metrics (absrel, rmse, rotation_error_deg,
        ...) the report's ``delta_int`` / ``delta_rec`` read as
        "positive = model got worse" and vice versa for
        higher-is-better metrics.
    cfg : TaskRunConfig
        A task-specific subclass of :class:`TaskRunConfig`. Its
        ``__post_init__`` must have already called
        :meth:`TaskRunConfig._validate_common`.
    model_resolver : callable
        Turns the config into a :class:`BenchmarkableModel`. Each
        task's runner supplies its own resolver so it can ship
        task-specific HF fast paths.
    compute_ts : bool
        Whether to compute Temporal Stability. Set False for tasks
        where TS is meaningless (single-sample predictions across
        unrelated pairs, e.g. pose / keypoint matching).
    compute_sgc : bool
        Whether to compute Stack-Level Geometric Coherence. Only
        meaningful when running segmentation + depth jointly.

    Returns
    -------
    (BenchmarkResult, DeploymentReadinessReport, dict)
        Same shape for every task. The dict contains ``"json"`` and
        ``"markdown"`` keys pointing at the written reports.

    Raises
    ------
    ConfigError, DownloadError, ManifestError, ModelError, MetricError
        Propagated from the respective subsystem with a hint.
    """
    cfg.device = resolve_device(cfg.device)

    split_name = cfg.split.value if isinstance(cfg.split, Difficulty) else str(cfg.split)
    log.info(
        "pipeline start: task=%s split=%s device=%s",
        task.value, split_name, cfg.device,
    )

    # 1. Download the split's task-scoped modalities.
    manifest_path = download_split(
        task=task,
        split=cfg.split,
        repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
        cache_dir=cfg.cache_dir,
        revision=cfg.revision,
    )
    dataset = RPXDataset.from_manifest(manifest_path, batch_size=cfg.batch_size)
    log.info("loaded %d samples from %s", len(dataset), manifest_path)

    # 2. Resolve the model — task-specific, handles all three selectors.
    model = model_resolver(cfg)
    name = display_name(cfg, model)

    # 3. Setup (outside the runner so we can profile before the run)
    # and static parameter count.
    model.setup()
    efficiency = _count_params_only(model)
    if efficiency.params_m is not None:
        log.info("model %s: %.2f M params", name, efficiency.params_m)

    # 4. Run the benchmark.
    runner = BenchmarkRunner(
        model=model,
        dataset=dataset,
        metric_suite=MetricSuite.for_task(task),
        call_setup=False,
    )
    result, dr_report = runner.run_with_deployment_readiness(
        primary_metric=primary_metric,
        model_name=name,
        efficiency=efficiency,
        compute_ts=compute_ts,
        compute_sgc_flag=compute_sgc,
        progress=cfg.progress,
    )

    # 5. Write reports.
    safe_name = name.replace("/", "__")
    out_dir = Path(cfg.output_dir or f"./rpx_results/{safe_name}/{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"

    write_json(
        json_path,
        task=task.value,
        model_name=name,
        split=split_name,
        repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
        result=result,
        dr_report=dr_report,
    )
    md_path.write_text(
        format_markdown_summary(
            task=task.value,
            model_name=name,
            split=split_name,
            repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
            result=result,
            dr_report=dr_report,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s and %s", json_path, md_path)

    return result, dr_report, {"json": json_path, "markdown": md_path}


__all__ = [
    "TaskRunConfig",
    "PipelineResult",
    "validate_model_selector",
    "normalise_split",
    "resolve_device",
    "display_name",
    "run_pipeline",
]

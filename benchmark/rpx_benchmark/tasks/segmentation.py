"""Object segmentation benchmark pipeline.

Structured identically to :mod:`rpx_benchmark.tasks.monocular_depth` —
this is the proof that adding a new task is a mechanical file copy
plus a handful of task-specific knobs:

- ``task``                  → :attr:`TaskType.OBJECT_SEGMENTATION`
- ``primary_metric``        → ``"miou"``
- ``higher_is_better``      → ``True`` (unlike depth)
- ``required_modalities``   → ``rgb`` + ``mask``
- model resolution          → new registry fast path (Mask2Former etc.)
- CLI args                  → same three-way model selector, just a
                              different default choice list

Usage
-----

::

    rpx bench object_segmentation \\
        --hf-checkpoint facebook/mask2former-swin-tiny-coco-instance \\
        --split hard
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
#: mIoU is higher-is-better.
PRIMARY_METRIC = "miou"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SegmentationRunConfig:
    """Runtime configuration for :func:`run_segmentation`.

    Exactly one of ``model``, ``model_name``, ``hf_checkpoint`` must
    be provided. See :class:`MonocularDepthRunConfig` for the mirror-
    image field-by-field documentation — every knob behaves the same
    way here.

    Raises
    ------
    ConfigError
        Multiple or zero selectors set, unknown difficulty split, or
        ``batch_size < 1``.
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

    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[ProgressCallback] = None

    def __post_init__(self) -> None:
        given = [x for x in (self.model, self.model_name, self.hf_checkpoint) if x]
        if len(given) != 1:
            raise ConfigError(
                "SegmentationRunConfig requires exactly one of "
                "`model`, `model_name`, or `hf_checkpoint`.",
                hint=(
                    "Pick a registered model or pass a HuggingFace "
                    "checkpoint id such as "
                    "'facebook/mask2former-swin-tiny-coco-instance'."
                ),
                details={"given_fields": [
                    k for k, v in (
                        ("model", self.model),
                        ("model_name", self.model_name),
                        ("hf_checkpoint", self.hf_checkpoint),
                    ) if v
                ]},
            )

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


# --------------------------------------------------------------------------- #
# Helpers (mirror monocular_depth.py; worth keeping duplicated so each task
# file is a self-contained story for new contributors)
# --------------------------------------------------------------------------- #

def _resolve_device(requested: str) -> str:
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


def _resolve_model(cfg: SegmentationRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        log.info("resolving segmentation model %r from registry", cfg.model_name)
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    from ..adapters.seg_hf import make_hf_instance_seg_model
    log.info("resolving HF segmentation checkpoint %r", cfg.hf_checkpoint)
    return make_hf_instance_seg_model(
        cfg.hf_checkpoint, device=cfg.device, **cfg.model_kwargs,
    )


def _display_name(cfg: SegmentationRunConfig, model: BenchmarkableModel) -> str:
    if cfg.model_name:
        return cfg.model_name
    if cfg.hf_checkpoint:
        return cfg.hf_checkpoint
    return getattr(model, "name", "model")


def _count_params_only(model: BenchmarkableModel) -> EfficiencyMetadata:
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

def run_segmentation(
    cfg: SegmentationRunConfig,
) -> Tuple[BenchmarkResult, DeploymentReadinessReport, Dict[str, Path]]:
    """Run the object-segmentation benchmark end-to-end.

    Parameters
    ----------
    cfg : SegmentationRunConfig

    Returns
    -------
    (BenchmarkResult, DeploymentReadinessReport, dict)
        Same shape as :func:`run_monocular_depth`. The
        :attr:`DeploymentReadinessReport.weighted_phase_score` is
        driven by mIoU (higher is better), so ``delta_int > 0`` means
        the model *improves* on the interaction phase and
        ``delta_int < 0`` is the "model degrades under interaction"
        failure mode.

    Raises
    ------
    ConfigError, DownloadError, ModelError, MetricError
        Propagated from the respective subsystem.
    """
    task = TaskType.OBJECT_SEGMENTATION
    cfg.device = _resolve_device(cfg.device)

    log.info("task=%s split=%s device=%s",
             task.value,
             cfg.split.value if isinstance(cfg.split, Difficulty) else cfg.split,
             cfg.device)

    manifest_path = download_split(
        task=task,
        split=cfg.split,
        repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
        cache_dir=cfg.cache_dir,
        revision=cfg.revision,
    )
    dataset = RPXDataset.from_manifest(manifest_path, batch_size=cfg.batch_size)
    log.info("loaded %d samples from %s", len(dataset), manifest_path)

    model = _resolve_model(cfg)
    display_name = _display_name(cfg, model)

    model.setup()
    efficiency = _count_params_only(model)
    if efficiency.params_m is not None:
        log.info("model %s: %.2f M params", display_name, efficiency.params_m)

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
    from .. import hub
    from ..models.registry import available_models

    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        # Filter the slate to segmentation-capable models only (none
        # registered yet; left permissive so the CLI accepts the
        # future entries).
        choices=[m for m in available_models()
                 if m.startswith(("mask2former", "sam2", "oneformer"))] or None,
        help="Registered segmentation adapter name.",
    )
    model_group.add_argument(
        "--hf-checkpoint",
        help="HuggingFace checkpoint id, e.g. "
             "'facebook/mask2former-swin-tiny-coco-instance'.",
    )
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> SegmentationRunConfig:
    return SegmentationRunConfig(
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


TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_SEGMENTATION,
    display_name="Object Segmentation",
    description=(
        "Predict a (H, W) int instance mask from a single RGB frame; "
        "evaluated with per-class mean-IoU against the D435/SAM2 "
        "ground-truth instance labels."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "mask"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_segmentation,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

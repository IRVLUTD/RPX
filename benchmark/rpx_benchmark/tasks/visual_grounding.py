"""Visual grounding benchmark pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..adapters import BenchmarkableModel
from ..api import Difficulty, TaskType
from ..exceptions import ConfigError
from ..logging_utils import get_logger
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

log = get_logger(__name__)

#: ``grounding_acc`` is higher-is-better.
PRIMARY_METRIC = "grounding_acc"


@dataclass
class VisualGroundingRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_visual_grounding`.

    Grounding models take ``(rgb, text)`` and return the referred
    bounding box. The referring expression is plucked from
    ``sample.ground_truth.text`` by the adapter so the model never
    sees the GT boxes.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: VisualGroundingRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "Visual grounding has no shipped HF fast path yet.",
        hint=(
            "Wrap your model via rpx.make_numpy_grounding_model(fn) and "
            "pass it as `model=` on the config. `fn(rgb, text)` should "
            "return a dict with keys 'boxes' and 'scores'."
        ),
    )


def run_visual_grounding(cfg: VisualGroundingRunConfig) -> PipelineResult:
    """Run the visual-grounding benchmark end-to-end."""
    return run_pipeline(
        task=TaskType.VISUAL_GROUNDING,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    p.add_argument("--model",
                   help="Registered grounding adapter name (none shipped yet).")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> VisualGroundingRunConfig:
    return VisualGroundingRunConfig(
        model_name=args.model,
        split=args.split,
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        revision=args.revision,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.VISUAL_GROUNDING,
    display_name="Visual Grounding",
    description=(
        "Given an RGB frame and a referring expression, predict the "
        "bounding box(es) of the referred object. Evaluated with top-1 "
        "IoU accuracy."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "spatial_qa"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_visual_grounding,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

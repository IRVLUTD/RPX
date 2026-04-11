"""Keypoint matching benchmark pipeline."""

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

#: ``keypoint_acc`` is higher-is-better.
PRIMARY_METRIC = "keypoint_acc"


@dataclass
class KeypointMatchingRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_keypoint_matching`.

    Matching models receive two RGB frames (``rgb_a`` + ``rgb_b``)
    and return corresponding points in each image's pixel grid.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: KeypointMatchingRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "Keypoint matching has no shipped HF fast path yet.",
        hint=(
            "Wrap your model via rpx.make_numpy_keypoint_model(fn) and "
            "pass it as `model=` on the config. `fn(rgb_a, rgb_b)` "
            "should return (points0, points1[, scores])."
        ),
    )


def run_keypoint_matching(cfg: KeypointMatchingRunConfig) -> PipelineResult:
    """Run the keypoint-matching benchmark end-to-end."""
    return run_pipeline(
        task=TaskType.KEYPOINT_MATCHING,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    p.add_argument("--model",
                   help="Registered keypoint matcher name (none shipped yet).")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> KeypointMatchingRunConfig:
    return KeypointMatchingRunConfig(
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
    task=TaskType.KEYPOINT_MATCHING,
    display_name="Keypoint Matching",
    description=(
        "Given a pair of RGB frames, predict corresponding points in "
        "each. Evaluated with percentage of matches within a pixel "
        "threshold on reprojection (default 3 px) against ground-truth "
        "correspondences derived from depth + T265 pose."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "keypoints"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_keypoint_matching,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

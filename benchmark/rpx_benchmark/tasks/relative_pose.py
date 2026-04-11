"""Relative camera pose benchmark pipeline."""

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

#: Rotation error in degrees is lower-is-better.
PRIMARY_METRIC = "rotation_error_deg"


@dataclass
class RelativePoseRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_relative_pose`.

    Pose models receive two RGB frames (``rgb_a`` and ``rgb_b``
    plucked from ``sample.metadata`` by the adapter) and return the
    predicted rotation + translation from frame A to frame B.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: RelativePoseRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "Relative-pose task has no shipped HF fast path yet.",
        hint=(
            "Wrap your model via rpx.make_numpy_pose_model(fn) and pass "
            "it as `model=` on the config. `fn(rgb_a, rgb_b)` should "
            "return a dict with keys 'rotation' (3x3 or quaternion) and "
            "'translation' (3-vector in metres)."
        ),
    )


def run_relative_pose(cfg: RelativePoseRunConfig) -> PipelineResult:
    """Run the relative-camera-pose benchmark end-to-end."""
    return run_pipeline(
        task=TaskType.RELATIVE_CAMERA_POSE,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    p.add_argument("--model",
                   help="Registered relative-pose adapter name (none shipped yet).")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> RelativePoseRunConfig:
    return RelativePoseRunConfig(
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
    task=TaskType.RELATIVE_CAMERA_POSE,
    display_name="Relative Camera Pose",
    description=(
        "Given a pair of RGB frames with a meaningful baseline, predict "
        "the 6-DoF relative camera pose (rotation + translation) from A "
        "to B. Evaluated with geodesic rotation error (degrees) and L2 "
        "translation error (metres)."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "pose"],
    higher_is_better=False,
    build_config=_build_config,
    run=run_relative_pose,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

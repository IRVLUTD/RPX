"""Object detection + open-vocabulary detection pipeline.

Handles both :attr:`TaskType.OBJECT_DETECTION` and
:attr:`TaskType.OPEN_VOCAB_DETECTION` — they share the same
Prediction / GroundTruth contract, only the ground-truth vocabulary
differs.
"""

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

#: F1 is higher-is-better.
PRIMARY_METRIC = "f1"


@dataclass
class ObjectDetectionRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_object_detection`.

    See :class:`TaskRunConfig` for the shared field reference.
    Detection currently has no registered model factories, so the
    only supported selector is ``model=`` (a pre-built
    :class:`BenchmarkableModel`, typically from
    :func:`rpx_benchmark.make_numpy_detection_model`).
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: ObjectDetectionRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "Object detection has no shipped HF fast path yet.",
        hint=(
            "Wrap your model via rpx.make_numpy_detection_model(fn) and "
            "pass it as `model=` on the config, or register a custom "
            "factory with rpx_benchmark.models.registry.register()."
        ),
    )


def run_object_detection(cfg: ObjectDetectionRunConfig) -> PipelineResult:
    """Run the object-detection benchmark end-to-end.

    The returned ``BenchmarkResult.aggregated`` has ``precision``,
    ``recall``, and ``f1`` keys produced by
    :class:`rpx_benchmark.metrics.detection.DetectionMetrics`. The
    deployment report uses ``f1`` as the primary metric and treats
    higher as better.
    """
    return run_pipeline(
        task=TaskType.OBJECT_DETECTION,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def run_open_vocab_detection(cfg: ObjectDetectionRunConfig) -> PipelineResult:
    """Run the open-vocabulary detection benchmark end-to-end.

    Uses the same metric suite as :func:`run_object_detection` but
    evaluates on :attr:`TaskType.OPEN_VOCAB_DETECTION` manifests.
    """
    return run_pipeline(
        task=TaskType.OPEN_VOCAB_DETECTION,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub

    p.add_argument(
        "--model",
        help="Registered detection adapter name (none shipped yet).",
    )
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> ObjectDetectionRunConfig:
    return ObjectDetectionRunConfig(
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
    task=TaskType.OBJECT_DETECTION,
    display_name="Object Detection",
    description=(
        "Predict category-conditioned bounding boxes from a single RGB "
        "frame; evaluated with greedy IoU matching + F1 at 0.5 IoU."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "boxes"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_object_detection,
    add_cli_arguments=_add_cli_arguments,
)

OPEN_VOCAB_TASK_SPEC = TaskSpec(
    task=TaskType.OPEN_VOCAB_DETECTION,
    display_name="Open-Vocabulary Detection",
    description=(
        "Detect objects by free-text category from a single RGB frame; "
        "same metric suite as closed-vocab detection."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "boxes", "questionnaires"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_open_vocab_detection,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)
register_task(OPEN_VOCAB_TASK_SPEC)

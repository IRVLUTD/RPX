"""Object tracking benchmark pipeline.

Per-frame protocol: the model receives one RGB frame at a time and
must return the *currently active* tracks with stable, persistent
``track_id`` strings. The shared
:func:`rpx_benchmark.tasks._pipeline.run_pipeline` drives iteration
and the tracking metric calculator
(:class:`rpx_benchmark.metrics.tracking.TrackletMetrics`) scores each
sample.

Scene-level HOTA / identity switches across full sequences are a
future extension and will hook in through the M3 runner-generalization
(``TaskPlugin.temporal_metrics``). For today, MOTA + IDF1 at the
tracklet granularity is what ships.
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

#: ``mota`` is higher-is-better, in ``[-1, 1]``.
PRIMARY_METRIC = "mota"


@dataclass
class ObjectTrackingRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_object_tracking`.

    Trackers receive one RGB frame per step and must return tracks
    (see :func:`rpx_benchmark.make_numpy_tracking_model`). The
    pipeline does not yet batch across scene boundaries — that lands
    with the M3 temporal-metrics generalization.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: ObjectTrackingRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "Object tracking has no shipped HF fast path yet.",
        hint=(
            "Wrap your tracker via rpx.make_numpy_tracking_model(fn) and "
            "pass it as `model=` on the config. `fn(rgb)` should return a "
            "list of tracks; see the docstring for accepted shapes."
        ),
    )


def run_object_tracking(cfg: ObjectTrackingRunConfig) -> PipelineResult:
    """Run the object-tracking benchmark end-to-end."""
    return run_pipeline(
        task=TaskType.OBJECT_TRACKING,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    p.add_argument("--model",
                   help="Registered tracker name (none shipped yet).")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> ObjectTrackingRunConfig:
    return ObjectTrackingRunConfig(
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
    task=TaskType.OBJECT_TRACKING,
    display_name="Object Tracking",
    description=(
        "Per-frame multi-object tracking. Models return the currently "
        "active tracks with persistent track IDs; scored with MOTA + "
        "IDF1 (tracklet granularity). Scene-level HOTA / identity-switch "
        "metrics land once the M3 temporal-metric generalisation is in."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "mask", "tracklets"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_object_tracking,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

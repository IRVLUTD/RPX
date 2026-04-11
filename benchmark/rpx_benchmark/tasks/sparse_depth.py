"""Sparse depth benchmark pipeline."""

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

#: ``sparse_absrel`` is lower-is-better.
PRIMARY_METRIC = "sparse_absrel"


@dataclass
class SparseDepthRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_sparse_depth`.

    Sparse-depth models receive ``(rgb, coordinates)`` where
    ``coordinates`` is the ``(N, 2)`` float array of pixel locations
    where depth is queried, and return an ``(N,)`` array of depth
    values in metres at those exact coordinates.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: SparseDepthRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "Sparse depth has no shipped HF fast path yet.",
        hint=(
            "Wrap your model via rpx.make_numpy_sparse_depth_model(fn) "
            "and pass it as `model=` on the config. `fn(rgb, coords)` "
            "should return an (N,) float array of depths in metres."
        ),
    )


def run_sparse_depth(cfg: SparseDepthRunConfig) -> PipelineResult:
    """Run the sparse-depth benchmark end-to-end."""
    return run_pipeline(
        task=TaskType.SPARSE_DEPTH,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    p.add_argument("--model",
                   help="Registered sparse-depth adapter name (none shipped yet).")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> SparseDepthRunConfig:
    return SparseDepthRunConfig(
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
    task=TaskType.SPARSE_DEPTH,
    display_name="Sparse Depth",
    description=(
        "Given RGB + a sparse set of pixel locations, predict metric "
        "depth at those exact locations. Evaluated with sparse AbsRel "
        "and RMSE against precomputed D435 samples."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth", "sparse_depth"],
    higher_is_better=False,
    build_config=_build_config,
    run=run_sparse_depth,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

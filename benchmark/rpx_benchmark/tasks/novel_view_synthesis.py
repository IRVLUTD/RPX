"""Novel view synthesis benchmark pipeline."""

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

#: PSNR is higher-is-better.
PRIMARY_METRIC = "psnr"


@dataclass
class NovelViewSynthesisRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_novel_view_synthesis`.

    NVS models receive ``(rgb_source, target_pose)`` where the
    target pose is a 4×4 SE(3) camera-to-world matrix plucked from
    ``sample.ground_truth.camera_pose`` by the adapter. They return
    a synthesised RGB image at the target viewpoint.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: NovelViewSynthesisRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    raise ConfigError(
        "NVS has no shipped HF fast path yet.",
        hint=(
            "Wrap your model via rpx.make_numpy_nvs_model(fn) and pass "
            "it as `model=` on the config. `fn(rgb, target_pose)` "
            "should return the synthesised RGB image."
        ),
    )


def run_novel_view_synthesis(cfg: NovelViewSynthesisRunConfig) -> PipelineResult:
    """Run the novel-view-synthesis benchmark end-to-end."""
    return run_pipeline(
        task=TaskType.NOVEL_VIEW_SYNTHESIS,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=False,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    p.add_argument("--model",
                   help="Registered NVS adapter name (none shipped yet).")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> NovelViewSynthesisRunConfig:
    return NovelViewSynthesisRunConfig(
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
    task=TaskType.NOVEL_VIEW_SYNTHESIS,
    display_name="Novel View Synthesis",
    description=(
        "Given a source RGB frame and a target camera pose, "
        "synthesise the RGB image at the target viewpoint. Evaluated "
        "with PSNR + simplified SSIM against held-out frames."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "depth", "pose"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_novel_view_synthesis,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)

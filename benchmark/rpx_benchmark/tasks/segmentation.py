"""Object segmentation benchmark pipeline.

Same template as :mod:`rpx_benchmark.tasks.monocular_depth` — swap
the task-specific knobs (``TaskType``, primary metric,
``higher_is_better``, model resolver), inherit everything else from
:func:`rpx_benchmark.tasks._pipeline.run_pipeline`.

Usage
-----

::

    rpx bench object_segmentation \\
        --hf-checkpoint facebook/mask2former-swin-tiny-coco-instance \\
        --split hard
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..adapters import BenchmarkableModel
from ..api import Difficulty, TaskType
from ..logging_utils import get_logger
from ._pipeline import PipelineResult, TaskRunConfig, run_pipeline
from .registry import TaskSpec, register_task

log = get_logger(__name__)

#: mIoU is higher-is-better, so ``delta_int > 0`` means the model
#: *improves* on the interaction phase (rare) and ``delta_int < 0``
#: is the "model degrades under interaction" failure mode.
PRIMARY_METRIC = "miou"


@dataclass
class SegmentationRunConfig(TaskRunConfig):
    """Runtime configuration for :func:`run_segmentation`.

    See :class:`TaskRunConfig` for the full field reference.

    Raises
    ------
    ConfigError
        If zero or more than one model selector is set, or if the
        split is not a valid ESD difficulty, or ``batch_size < 1``.
    """

    def __post_init__(self) -> None:
        self._validate_common()


def _resolve_model(cfg: SegmentationRunConfig) -> BenchmarkableModel:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_name is not None:
        from ..models.registry import resolve
        log.info("resolving segmentation model %r from registry", cfg.model_name)
        return resolve(cfg.model_name, device=cfg.device, **cfg.model_kwargs)
    from ..reference.adapters.seg_hf import make_hf_instance_seg_model
    log.info("resolving HF segmentation checkpoint %r", cfg.hf_checkpoint)
    return make_hf_instance_seg_model(
        cfg.hf_checkpoint, device=cfg.device, **cfg.model_kwargs,
    )


def run_segmentation(cfg: SegmentationRunConfig) -> PipelineResult:
    """Run the object-segmentation benchmark end-to-end.

    Returns the same ``(BenchmarkResult, DeploymentReadinessReport,
    {json, markdown})`` tuple shape as :func:`run_monocular_depth`.
    ``primary_metric="miou"`` and ``higher_is_better=True`` so the
    deployment-readiness report interprets deltas accordingly.
    """
    return run_pipeline(
        task=TaskType.OBJECT_SEGMENTATION,
        primary_metric=PRIMARY_METRIC,
        cfg=cfg,
        model_resolver=_resolve_model,
        compute_ts=True,
        compute_sgc=False,
    )


def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    from .. import hub
    from ..models.registry import available_models

    seg_models = [m for m in available_models()
                  if m.startswith(("mask2former", "sam2", "oneformer"))]

    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        choices=seg_models or None,
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


def _temporal_stability_hook(predictions, samples, camera_poses):
    """Segmentation Temporal Stability hook.

    Uses predicted instance masks; compatible with any segmentation
    model whose ``finalize`` step returns a single-channel int mask.
    """
    from ..deployment import compute_temporal_stability_seg
    masks = [p.mask for p in predictions]
    return compute_temporal_stability_seg(masks, camera_poses)


def _geometric_coherence_hook(predictions, samples):
    """Stack Geometric Coherence hook for segmentation runs.

    SGC requires paired mask + depth. We pull depth from
    ``sample.metadata['depth_map']`` (the loader stashes it there when
    the manifest's segmentation entry also references depth). Returns
    ``None`` when fewer than one sample carries depth — the runner
    interprets that as "not applicable".
    """
    import numpy as np

    from ..deployment import compute_sgc

    depth_maps = [
        s.metadata.get("depth_map") if s.metadata else None
        for s in samples
    ]
    valid = [
        (p.mask, d) for p, d in zip(predictions, depth_maps) if d is not None
    ]
    if not valid:
        return None
    masks_sgc = [v[0] for v in valid]
    depths_sgc = [np.asarray(v[1], dtype=np.float32) for v in valid]
    return compute_sgc(masks_sgc, depths_sgc)


TASK_SPEC = TaskSpec(
    task=TaskType.OBJECT_SEGMENTATION,
    display_name="Object Segmentation",
    description=(
        "Predict a (H, W) int instance mask from a single RGB frame; "
        "evaluated with per-class mIoU against the D435/SAM2 "
        "ground-truth instance labels."
    ),
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "mask"],
    higher_is_better=True,
    build_config=_build_config,
    run=run_segmentation,
    add_cli_arguments=_add_cli_arguments,
    temporal_stability_fn=_temporal_stability_hook,
    geometric_coherence_fn=_geometric_coherence_hook,
)

register_task(TASK_SPEC)

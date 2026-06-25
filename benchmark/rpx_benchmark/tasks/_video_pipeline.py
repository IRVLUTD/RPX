"""Per-clip pipeline for video tasks (Video Depth today, future per-clip tasks
later).

The sibling of :mod:`rpx_benchmark.tasks._pipeline`, which handles
per-frame tasks. The two pipelines are deliberately split rather than
overloaded onto a single ``run_pipeline``: the iteration unit is
different (frame vs. clip), the dataset class is different
(:class:`~rpx_benchmark.loader.RPXDataset` vs.
:class:`~rpx_benchmark.video_loader.VideoDepthDataset`), and per-clip tasks
have an explicit alignment hook between ``model.predict`` and the
metric suite that per-frame tasks don't need. Mixing both into one
function obscured both code paths during the Video Depth prototype; the
split is the result of that prototype's lesson.

The contract for a video model is exactly what
:class:`~rpx_benchmark.api.BenchmarkModel` describes: ``setup()``
loads weights, ``predict(batch[VideoSample]) -> list[VideoDepthPrediction]``
runs one inference per clip. The model declares
``depth_output_kind = "metric"`` (default) or ``"relative"`` on the
class; the pipeline applies per-clip ``(s, t)`` alignment to the
prediction *before* metrics are computed when the model is relative.
This matches the community convention for evaluating affine-invariant
video-depth models (DepthCrafter, RollingDepth, MoGe-2 video runs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..api import TaskType, VideoDepthPrediction, VideoSample
from ..cell_log import cells_from_per_sample, write_cells
from ..exceptions import ConfigError
from ..hub import DEFAULT_REPO_ID, download_split
from ..logging_utils import get_logger
from ..metrics.depth_alignment import align_pred_to_gt_pooled
from ..metrics.registry import BenchmarkResult, MetricSuite
from ..reports import format_markdown_summary, write_json
from ..video_loader import VideoDepthDataset
from ._pipeline import PipelineResult, TaskRunConfig, resolve_device

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class VideoTaskRunConfig(TaskRunConfig):
    """Adds Video Depth-specific knobs to :class:`TaskRunConfig`.

    ``frame_budget`` and ``sampling`` are the temporal-resolution
    ablation knobs from the paper (\\S5.2). Defaults mean "use every
    frame on disk", which is the headline run.
    """

    frame_budget: Optional[int] = None
    sampling: str = "all"


# --------------------------------------------------------------------------- #
# Alignment helper
# --------------------------------------------------------------------------- #


def _apply_clip_alignment(
    pred: VideoDepthPrediction,
    sample: VideoSample,
) -> VideoDepthPrediction:
    """Solve one ``(s, t)`` over the whole clip and apply it.

    Relative-depth video models produce ``pred.depth_map_seq`` up to an
    affine transform per clip. Per the community convention
    (DepthCrafter, RollingDepth, Video DA), we fit a single ``(s, t)``
    over every valid pixel in the clip and apply it uniformly across
    all ``T`` frames. Per-frame fitting would let the model cheat
    temporal-inconsistency punishments; per-scene fitting (pooling all
    three phases) would conflate phase-stability with calibration drift.
    """
    gt = sample.ground_truth
    aligned_seq = align_pred_to_gt_pooled(
        pred_seq=pred.depth_map_seq.astype(np.float32),
        gt_seq=gt.depth_map_seq.astype(np.float32),
        mode="ls_affine",
        valid_seq=gt.valid_mask_seq.astype(bool),
    )
    return VideoDepthPrediction(depth_map_seq=aligned_seq)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_video_pipeline(
    *,
    task: TaskType,
    primary_metric: str,
    cfg: VideoTaskRunConfig,
) -> PipelineResult:
    """End-to-end per-clip pipeline: download → load → predict (+ align)
    → metric suite → cell-log row → JSON / markdown summary.

    The only Video Depth-specific knobs are ``frame_budget`` and ``sampling``;
    everything else lines up with the per-frame
    :func:`~rpx_benchmark.tasks._pipeline.run_pipeline` so the team
    doesn't have to learn two contracts.
    """
    cfg.device = resolve_device(cfg.device)
    split_name = cfg.split.value if hasattr(cfg.split, "value") else str(cfg.split)
    repo_id = cfg.repo_id or DEFAULT_REPO_ID

    log.info(
        "video pipeline: task=%s split=%s device=%s frame_budget=%s sampling=%s",
        task.value,
        split_name,
        cfg.device,
        cfg.frame_budget,
        cfg.sampling,
    )

    if cfg.manifest_path:
        manifest_path = Path(cfg.manifest_path)
        if not manifest_path.exists():
            raise ConfigError(
                f"manifest_path does not exist: {manifest_path}",
                hint="Generate per-task manifests for a local v2-webp "
                "tree with `python -m rpx_benchmark.dataset_hub.cli "
                "manifest ...` and pass the resulting "
                "<staging>/manifests/video_depth/<split>.json path.",
            )
        log.info("using local manifest: %s", manifest_path)
    else:
        manifest_path = download_split(
            task=task,
            split=cfg.split,
            repo_id=repo_id,
            cache_dir=cfg.cache_dir,
            revision=cfg.revision,
        )
    dataset = VideoDepthDataset.from_manifest(
        manifest_path,
        batch_size=cfg.batch_size,
        frame_budget=cfg.frame_budget,
        sampling=cfg.sampling,
    )
    log.info("loaded %d clips from %s", len(dataset), manifest_path)

    model = cfg.model
    name = getattr(model, "name", "model")
    is_relative = getattr(model, "depth_output_kind", "metric") == "relative"

    model.setup()
    if is_relative:
        log.info("model %s is relative-depth → per-clip (s,t) alignment will be applied", name)

    metric_suite = MetricSuite.for_task(task)

    per_sample: List[dict] = []
    n_clips = len(dataset)
    for clip_idx, batch in enumerate(dataset, start=1):
        if not batch:
            continue
        t0 = time.perf_counter()
        predictions = model.predict(batch)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(batch), 1)

        if len(predictions) != len(batch):
            raise ConfigError(
                f"model returned {len(predictions)} predictions for a batch of "
                f"{len(batch)} clips; the contract is one prediction per clip",
                hint=(
                    "Check the model's predict(): a video-depth model must "
                    "return one VideoDepthPrediction per input VideoSample."
                ),
            )

        for sample, pred in zip(batch, predictions, strict=False):
            if is_relative:
                pred = _apply_clip_alignment(pred, sample)
            metrics: Dict[str, float] = dict(
                metric_suite.evaluate(pred, sample.ground_truth)
            )
            meta = sample.metadata or {}
            metrics.update(
                id=sample.id,
                scene=meta.get("scene_id"),
                phase=sample.phase,
                difficulty=sample.difficulty,
                frame_budget=int(meta.get("frame_budget", 0)),
                latency_ms=latency_ms,
            )
            per_sample.append(metrics)

        if clip_idx % 5 == 0 or clip_idx == n_clips:
            log.info("video pipeline: %d/%d clips processed", clip_idx, n_clips)

    result = metric_suite.build_result(per_sample)

    safe_name = name.replace("/", "__")
    out_dir = Path(cfg.output_dir or f"./rpx_results/{safe_name}/{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    cells_path = out_dir / "cells.parquet"

    # Auto-detected metric keys exclude obvious metadata (id/scene/phase/
    # difficulty/latency_ms via the cell_log defaults) but ``frame_budget``
    # looks numeric, so list metrics explicitly to keep it out.
    auto_metric_keys = sorted(
        k for k, v in (per_sample[0].items() if per_sample else [])
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        and k not in {"frame_budget", "latency_ms"}
    )
    cells = cells_from_per_sample(
        per_sample,
        model_name=name,
        task=task.value,
        metric_keys=auto_metric_keys,
    )
    write_cells(cells, cells_path)
    log.info("wrote %d cell rows to %s", len(cells), cells_path)

    write_json(
        json_path,
        task=task.value,
        model_name=name,
        split=split_name,
        repo_id=repo_id,
        result=result,
        dr_report=None,
    )
    md_path.write_text(
        format_markdown_summary(
            task=task.value,
            model_name=name,
            split=split_name,
            repo_id=repo_id,
            result=result,
            dr_report=None,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s and %s", json_path, md_path)

    paths: Dict[str, Path] = {
        "json": json_path,
        "markdown": md_path,
        "cells": cells_path,
        "out_dir": out_dir,
    }

    if cfg.upload_to_box:
        from ..box_upload import DEFAULT_BOX_FOLDER_ID, upload_run_dir

        try:
            summary = upload_run_dir(
                out_dir,
                task=task.value,
                model_name=name,
                split=split_name,
                root_folder_id=cfg.box_folder_id or DEFAULT_BOX_FOLDER_ID,
            )
            log.info(
                "box: %d uploaded, %d skipped → box:%s",
                summary["uploaded"],
                summary["skipped"],
                summary["remote_path"],
            )
            paths["box_remote"] = summary["remote_path"]
        except Exception as e:  # noqa: BLE001
            log.error(
                "Box upload failed (%s: %s) — local artefacts at %s are intact",
                type(e).__name__,
                e,
                out_dir,
            )

    return result, None, paths  # type: ignore[return-value]


__all__ = ["VideoTaskRunConfig", "run_video_pipeline"]

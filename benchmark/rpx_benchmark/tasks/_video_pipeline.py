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

import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..api import TaskType, VideoDepthPrediction, VideoSample
from ..cell_log import cells_from_per_sample, write_cells
from ..exceptions import AdapterError, ConfigError
from ..hub import DEFAULT_REPO_ID, download_split
from ..logging_utils import get_logger
from ..metrics.depth_alignment import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    align_pred_to_gt_pooled,
)
from ..metrics.registry import MetricSuite
from ..reports import format_markdown_summary, write_json
from ..runner import _make_cuda_sync
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
    save_predictions: bool = False
    resume_predictions: bool = False
    compute_fscore: bool = False


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
    fit_mask = (
        gt.valid_mask_seq.astype(bool)
        & np.isfinite(pred.depth_map_seq)
        & np.isfinite(gt.depth_map_seq)
        & (gt.depth_map_seq > DEPTH_MIN_M)
        & (gt.depth_map_seq < DEPTH_MAX_M)
    )
    aligned_seq = align_pred_to_gt_pooled(
        pred_seq=pred.depth_map_seq.astype(np.float32),
        gt_seq=gt.depth_map_seq.astype(np.float32),
        mode="ls_affine",
        valid_seq=fit_mask,
    )
    return VideoDepthPrediction(depth_map_seq=aligned_seq)


def _prediction_path(predictions_dir: Path, sample: VideoSample) -> Path:
    meta = sample.metadata or {}
    scene = meta.get("scene_id") or str(sample.id)
    phase = meta.get("phase_idx")
    if phase is None:
        phase = sample.phase.value if sample.phase is not None else "unknown"
    return predictions_dir / str(scene) / str(phase) / "depth.npz"


def _paper_valid_mask(sample: VideoSample) -> np.ndarray:
    gt = np.asarray(sample.ground_truth.depth_map_seq)
    return (
        np.asarray(sample.ground_truth.valid_mask_seq, dtype=bool)
        & np.isfinite(gt)
        & (gt > DEPTH_MIN_M)
        & (gt < DEPTH_MAX_M)
    )


def _validate_prediction(
    depth: np.ndarray,
    sample: VideoSample,
    *,
    raise_on_error: bool,
) -> np.ndarray | None:
    expected_shape = np.asarray(sample.ground_truth.depth_map_seq).shape
    error: str | None = None
    if depth.dtype != np.dtype(np.float32):
        error = f"dtype {depth.dtype} is not float32"
    elif depth.ndim != 3 or depth.shape != expected_shape:
        error = f"shape {depth.shape} does not match {expected_shape}"
    elif not np.isfinite(depth).all():
        error = "prediction contains non-finite values"
    else:
        valid = _paper_valid_mask(sample)
        if not valid.any():
            error = "clip has no GT pixels in 0.3 < depth < 5.0 m"
        elif float(np.ptp(depth[valid])) <= 1e-6:
            error = "prediction is degenerate (constant on paper-valid pixels)"
    if error is None:
        return depth
    if raise_on_error:
        raise AdapterError(
            f"Refusing invalid Video Depth prediction for {sample.id!r}: {error}",
            hint=(
                "Expected one finite, non-degenerate float32 depth sequence "
                "matching GT. Evaluation applies the documented [0.3, 5.0] "
                "prediction clip on the strict paper-valid GT mask."
            ),
        )
    return None


def _load_cached_prediction(
    path: Path,
    sample: VideoSample,
) -> tuple[VideoDepthPrediction, float] | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"depth", "frame_indices", "latency_ms"}:
                return None
            depth = np.asarray(payload["depth"])
            frame_indices = np.asarray(payload["frame_indices"])
            latency_raw = np.asarray(payload["latency_ms"])
        expected_frames = np.asarray(sample.ground_truth.frame_indices, dtype=np.int32)
        if frame_indices.dtype != np.dtype(np.int32):
            return None
        if not np.array_equal(frame_indices, expected_frames):
            return None
        if latency_raw.size != 1:
            return None
        latency_ms = float(latency_raw.reshape(-1)[0])
        if not np.isfinite(latency_ms) or latency_ms < 0:
            return None
        depth = _validate_prediction(depth, sample, raise_on_error=False)
        if depth is None:
            return None
        return VideoDepthPrediction(depth_map_seq=depth), latency_ms
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _atomic_save_prediction(
    path: Path,
    prediction: VideoDepthPrediction,
    sample: VideoSample,
    latency_ms: float,
) -> None:
    depth = np.asarray(prediction.depth_map_seq, dtype=np.float32)
    _validate_prediction(depth, sample, raise_on_error=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".npz.part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(
                handle,
                depth=depth,
                frame_indices=np.asarray(
                    sample.ground_truth.frame_indices,
                    dtype=np.int32,
                ),
                latency_ms=np.asarray(latency_ms, dtype=np.float64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json.part",
            dir=path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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
    cfg.device = resolve_device(cfg.device, require_cuda=cfg.require_cuda)
    if cfg.resume_predictions and not cfg.save_predictions:
        raise ConfigError(
            "resume_predictions requires save_predictions",
            hint="Enable both flags so missing or invalid cache entries can be replaced.",
        )
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
            max_samples=cfg.max_samples,
        )
    dataset = VideoDepthDataset.from_manifest(
        manifest_path,
        batch_size=cfg.batch_size,
        frame_budget=cfg.frame_budget,
        sampling=cfg.sampling,
        max_samples=cfg.max_samples,
        compute_fscore=cfg.compute_fscore,
    )
    log.info("loaded %d clips from %s", len(dataset), manifest_path)

    model = cfg.model
    name = getattr(model, "name", "model")
    is_relative = getattr(model, "depth_output_kind", "metric") == "relative"

    if is_relative:
        log.info("model %s is relative-depth → per-clip (s,t) alignment will be applied", name)

    safe_name = name.replace("/", "__")
    out_dir = Path(cfg.output_dir or f"./rpx_results/{safe_name}/{split_name}")
    predictions_dir = out_dir / "predictions"
    prediction_stats = {
        "cache_hits": 0,
        "inferred_new": 0,
        "invalid_recomputed": 0,
    }

    # Auto-detect SystemCard (GPU / precision / batch_size) so every
    # cell row carries the host hardware context. Required for
    # downstream cross-host aggregation per
    # benchmark/docs/adapter_status.md publication gate.
    from ..profiler import SystemCard

    system_card = SystemCard.auto_detect(
        precision=getattr(model, "native_precision", "fp32"),
        batch_size=cfg.batch_size,
    )

    metric_suite = MetricSuite.for_task(task)

    per_sample: List[dict] = []
    n_clips = len(dataset)
    cuda_sync = _make_cuda_sync()
    model_is_setup = False
    for clip_idx, batch in enumerate(dataset, start=1):
        if not batch:
            continue
        predictions: List[VideoDepthPrediction | None] = [None] * len(batch)
        latencies_ms: List[float | None] = [None] * len(batch)
        missing_indices: List[int] = []
        invalid_existing: set[int] = set()
        for index, sample in enumerate(batch):
            path = _prediction_path(predictions_dir, sample)
            if cfg.resume_predictions and path.exists():
                cached = _load_cached_prediction(path, sample)
                if cached is not None:
                    predictions[index], latencies_ms[index] = cached
                    prediction_stats["cache_hits"] += 1
                    continue
                invalid_existing.add(index)
            missing_indices.append(index)

        if missing_indices:
            if not model_is_setup:
                model.setup()
                model_is_setup = True
            missing_batch = [batch[index] for index in missing_indices]
            cuda_sync()
            t0 = time.perf_counter()
            inferred = model.predict(missing_batch)
            cuda_sync()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if len(inferred) != len(missing_batch):
                raise ConfigError(
                    f"model returned {len(inferred)} predictions for "
                    f"{len(missing_batch)} clips; expected one per clip",
                )
            per_clip_latency_ms = elapsed_ms / len(missing_batch)
            for index, pred in zip(missing_indices, inferred, strict=True):
                sample = batch[index]
                if is_relative:
                    pred = _apply_clip_alignment(pred, sample)
                canonical = np.asarray(pred.depth_map_seq, dtype=np.float32)
                _validate_prediction(canonical, sample, raise_on_error=True)
                pred = VideoDepthPrediction(depth_map_seq=canonical)
                predictions[index] = pred
                latencies_ms[index] = per_clip_latency_ms
                if index in invalid_existing:
                    prediction_stats["invalid_recomputed"] += 1
                else:
                    prediction_stats["inferred_new"] += 1
                if cfg.save_predictions:
                    _atomic_save_prediction(
                        _prediction_path(predictions_dir, sample),
                        pred,
                        sample,
                        per_clip_latency_ms,
                    )

        for sample, pred, latency_ms in zip(
            batch,
            predictions,
            latencies_ms,
            strict=True,
        ):
            if pred is None or latency_ms is None:  # pragma: no cover - invariant
                raise ConfigError("internal error: unresolved video prediction")
            evaluation_pred = VideoDepthPrediction(
                depth_map_seq=np.clip(
                    pred.depth_map_seq,
                    DEPTH_MIN_M,
                    DEPTH_MAX_M,
                ).astype(np.float32, copy=False),
            )
            metrics: Dict[str, float] = dict(
                metric_suite.evaluate(evaluation_pred, sample.ground_truth)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    cells_path = out_dir / "cells.parquet"
    metadata_path = out_dir / "run_metadata.json"

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
        system_card=system_card,
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
    _atomic_write_json(
        metadata_path,
        {
            "task": task.value,
            "model_name": name,
            "split": split_name,
            "save_predictions": cfg.save_predictions,
            "resume_predictions": cfg.resume_predictions,
            "compute_fscore": cfg.compute_fscore,
            "prediction_evaluation_policy": {
                "operation": "clip",
                "minimum_m": DEPTH_MIN_M,
                "maximum_m": DEPTH_MAX_M,
                "raw_cache_preserved": True,
            },
            "prediction_stats": prediction_stats,
        },
    )

    paths: Dict[str, Path] = {
        "json": json_path,
        "markdown": md_path,
        "cells": cells_path,
        "out_dir": out_dir,
        "run_metadata": metadata_path,
    }
    if cfg.save_predictions:
        paths["predictions_dir"] = predictions_dir
        paths["prediction_stats"] = prediction_stats

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

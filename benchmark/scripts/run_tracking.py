#!/usr/bin/env python3
"""Run the paper-valid RPX D3 tracking benchmark on real scene-phase clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

DEFAULT_DATASET_REPO = "anonymous/RPX"
PINNED_DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
PINNED_EGO_DATASET_REVISION = "f082723002bad5800dd85e583115b4ea05734d31"
MOS_EXPECTED_SPLITS = {
    "easy": (24750, 99),
    "medium": (24750, 99),
    "hard": (25500, 102),
}
EGO_EXPECTED_SPLITS = {
    "easy": (7552, 33),
    "medium": (7691, 33),
    "hard": (7878, 34),
}
EXPECTED_SPLITS = MOS_EXPECTED_SPLITS
TRACKING_DATASETS = {
    "mos": {
        "revision": PINNED_DATASET_REVISION,
        "manifest_name": "object_tracking",
        "expected_splits": MOS_EXPECTED_SPLITS,
        "fixed_clip_frames": 250,
    },
    "ego": {
        "revision": PINNED_EGO_DATASET_REVISION,
        "manifest_name": "ego_object_tracking",
        "expected_splits": EGO_EXPECTED_SPLITS,
        "fixed_clip_frames": None,
    },
}
EXPECTED_SHAPE = (480, 640)
EGO_EXPECTED_SHAPE = (1080, 1920)
EXPECTED_SHAPES = {"mos": EXPECTED_SHAPE, "ego": EGO_EXPECTED_SHAPE}

# Prefer the benchmark source tree containing this script over an older
# rpx_benchmark wheel installed in a cumulative model environment. This is
# required when the evaluator is bind-mounted into an immutable model image.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))


def _rpx_git_sha() -> str:
    """Return the adapter identity embedded by the image build."""

    return os.environ.get("RPX_GIT_SHA", "unknown")


def _resume_compatible_git_shas() -> set[str]:
    """Return explicitly approved prior revisions for prediction-only reuse.

    This is intentionally opt-in. It is used for evaluator-only recovery fixes
    where already committed model masks are unchanged and must not be
    recomputed. The accepted revisions are emitted in the run metadata.
    """

    return {
        revision.strip()
        for revision in os.environ.get("RPX_RESUME_COMPATIBLE_GIT_SHAS", "").split(",")
        if revision.strip()
    }


sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracking_models import TRACKER_CLASSES  # noqa: E402
from tracking_text_runtime import (  # noqa: E402
    TEXT_VOCAB_REVISION,
    TEXT_VOCAB_SHA256,
    TrackingTextVocabulary,
)

from rpx_benchmark import TaskType, download_split  # noqa: E402
from rpx_benchmark.exceptions import ConfigError, DatasetError  # noqa: E402
from rpx_benchmark.metrics.tracking_paper import (  # noqa: E402
    PAPER_TRACKING_METRICS,
    paper_tracking_metrics,
)


@dataclass(frozen=True)
class Clip:
    scene: str
    phase: int
    split: str
    root: Path
    samples: tuple[dict[str, Any], ...]

    @property
    def key(self) -> str:
        return f"{self.scene}__{self.phase}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RPX D3: run each tracker's declared mask, box, detector, or text "
            "initialization protocol and evaluate with official TrackEval."
        )
    )
    parser.add_argument("--model", choices=sorted(TRACKER_CLASSES), default="sam2")
    parser.add_argument("--split", choices=sorted(MOS_EXPECTED_SPLITS), required=True)
    parser.add_argument(
        "--dataset-protocol",
        choices=sorted(TRACKING_DATASETS),
        default="mos",
        help="Use the canonical multi-object scenes (mos) or ego videos (ego).",
    )
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument(
        "--revision",
        help="Dataset revision; defaults to the immutable revision pinned for the protocol.",
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument(
        "--text-vocab",
        help=(
            "Pinned scene_condition_vocab.parquet. Required by text-prompted "
            "trackers and rejected if its SHA-256 differs from text-initialization v1."
        ),
    )
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--resume-predictions", action="store_true")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dataset-workers", type=int, default=8)
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, mask=mask.astype(np.int32, copy=False))
    temporary.replace(path)


def _load_mask_file(
    path: Path,
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise DatasetError(f"Tracking mask {path} must be 2-D, got {mask.shape}.")
    mask = mask.astype(np.int32, copy=False)
    if mask.shape != expected_shape:
        raise DatasetError(
            f"Tracking mask {path} has shape {mask.shape}; expected {expected_shape}."
        )
    if np.any(mask < 0):
        raise DatasetError(f"Tracking mask {path} contains negative IDs.")
    return mask


def _load_prediction(
    path: Path,
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["mask"]:
                return None
            mask = archive["mask"]
    except (OSError, ValueError, KeyError):
        return None
    if (
        mask.shape != expected_shape
        or not np.issubdtype(mask.dtype, np.integer)
        or np.any(mask < 0)
    ):
        return None
    return mask.astype(np.int32, copy=False)


def _prediction_path(output_dir: Path, clip: Clip, sample: dict[str, Any]) -> Path:
    frame = Path(str(sample["rgb"])).stem
    return output_dir / "predictions" / clip.scene / str(clip.phase) / f"{frame}.npz"


def _resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_clips(manifest_path: Path, split: str) -> list[Clip]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != TaskType.OBJECT_TRACKING.value:
        raise DatasetError(f"Expected object_tracking manifest, got {manifest.get('task')!r}.")
    root = Path(manifest["root"])
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in manifest.get("samples") or []:
        groups[(str(sample["scene_id"]), int(sample["phase"]))].append(sample)

    clips: list[Clip] = []
    for (scene, phase), samples in sorted(groups.items()):
        samples.sort(key=lambda item: int(item["frame_idx"]))
        frame_indices = [int(sample["frame_idx"]) for sample in samples]
        expected = list(range(len(samples)))
        if frame_indices != expected:
            raise DatasetError(f"{scene}/phase {phase} frame indices are not contiguous from zero.")
        for sample in samples:
            for field in ("rgb", "mask"):
                path = _resolve(str(sample[field]), root)
                if not path.is_file():
                    raise DatasetError(f"Manifest path is missing: {path}")
        clips.append(
            Clip(
                scene=scene,
                phase=phase,
                split=split,
                root=root,
                samples=tuple(samples),
            )
        )
    return clips


def _validate_split(clips: list[Clip], split: str, dataset_protocol: str = "mos") -> None:
    spec = TRACKING_DATASETS[dataset_protocol]
    expected_frames, expected_clips = spec["expected_splits"][split]
    actual_frames = sum(len(clip.samples) for clip in clips)
    if (actual_frames, len(clips)) != (expected_frames, expected_clips):
        raise DatasetError(
            f"{split} has {actual_frames} frames/{len(clips)} clips; "
            f"expected {expected_frames}/{expected_clips}."
        )
    fixed_clip_frames = spec["fixed_clip_frames"]
    if fixed_clip_frames is not None and any(
        len(clip.samples) != fixed_clip_frames for clip in clips
    ):
        raise DatasetError(
            f"{split} contains a clip that is not exactly {fixed_clip_frames} frames."
        )


def _stage_video(clip: Clip, samples: tuple[dict[str, Any], ...], scratch: Path) -> Path:
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for index, sample in enumerate(samples):
        source = _resolve(str(sample["rgb"]), clip.root).resolve()
        # SAM 2 enumerates *.jpg files, while RPX RGB is losslessly represented
        # by the already-decoded release image. PIL determines format from file
        # contents, so a symlink avoids a 75,000-frame JPEG re-encode.
        (scratch / f"{index:05d}.jpg").symlink_to(source)
    return scratch


def _clip_predictions(
    clip: Clip,
    samples: tuple[dict[str, Any], ...],
    output_dir: Path,
    model_name: str,
    rpx_git_sha: str,
    dataset_protocol: str = "mos",
) -> list[np.ndarray] | None:
    marker_path = output_dir / "predictions" / clip.scene / str(clip.phase) / "_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    evaluator_git_sha = os.environ.get("RPX_EVALUATOR_GIT_SHA", rpx_git_sha)
    compatible_revisions = _resume_compatible_git_shas()
    accepted_adapter_revisions = {rpx_git_sha, *compatible_revisions}
    accepted_evaluator_revisions = {evaluator_git_sha, *compatible_revisions}
    if (
        marker.get("model") != model_name
        or marker.get("frames") != len(samples)
        or marker.get("rpx_git_sha") not in accepted_adapter_revisions
        or marker.get("evaluator_git_sha", marker.get("rpx_git_sha"))
        not in accepted_evaluator_revisions
        or marker.get("dataset_protocol", "mos") != dataset_protocol
        or (
            model_name in {"grounded-sam2", "sam3.1"}
            and (
                marker.get("prompt_type") != TRACKER_CLASSES[model_name].prompt_type
                or marker.get("tracking_mode")
                != getattr(TRACKER_CLASSES[model_name], "tracking_mode", None)
            )
        )
    ):
        return None
    predictions: list[np.ndarray] = []
    expected_shape = EXPECTED_SHAPES[dataset_protocol]
    for sample in samples:
        prediction = _load_prediction(_prediction_path(output_dir, clip, sample), expected_shape)
        if prediction is None:
            return None
        predictions.append(prediction)
    return predictions


def _write_complete_marker(
    clip: Clip,
    sample_count: int,
    output_dir: Path,
    model_name: str,
    tracker_class: type,
    rpx_git_sha: str,
    model_outputs: str | None = None,
    dataset_protocol: str = "mos",
) -> None:
    marker = output_dir / "predictions" / clip.scene / str(clip.phase) / "_complete.json"
    _atomic_json(
        marker,
        {
            "model": model_name,
            "model_id": tracker_class.model_id,
            "model_revision": tracker_class.model_revision,
            "rpx_git_sha": rpx_git_sha,
            "evaluator_git_sha": os.environ.get("RPX_EVALUATOR_GIT_SHA", rpx_git_sha),
            "dataset_protocol": dataset_protocol,
            "prompt_type": tracker_class.prompt_type,
            "tracking_mode": getattr(tracker_class, "tracking_mode", None),
            "frames": sample_count,
            "model_outputs": model_outputs,
        },
    )


def _write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ConfigError(
            "Writing RPX tracking tables requires pandas and pyarrow.",
            hint="Use the tracking Docker image.",
        ) from exc
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "cells.csv", index=False)
    frame.to_parquet(output_dir / "cells.parquet", index=False)


def _previous_cells(output_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load hardware measurements from an earlier resumable invocation."""

    path = output_dir / "cells.parquet"
    if not path.is_file():
        return {}
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except (ImportError, OSError, ValueError):
        return {}
    if not {"scene", "phase"}.issubset(frame.columns):
        return {}
    return {(str(row["scene"]), int(row["phase"])): row for row in frame.to_dict(orient="records")}


def _finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _load_open_vocabulary_metadata(output_dir: Path, clip: Clip) -> dict[str, Any]:
    path = output_dir / "open_vocabulary_predictions" / f"{clip.key}.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _track_to_source_identity(metadata: dict[str, Any]) -> dict[int, int]:
    """Return predicted track ID -> RPX ground-truth mask identity."""

    mapping: dict[int, int] = {}
    for detection in metadata.get("detections") or []:
        track_id = detection.get("predicted_track_id")
        source_id = detection.get("source_mask_index")
        if track_id is not None and source_id is not None:
            mapping[int(track_id)] = int(source_id)
    for prompt in metadata.get("prompts") or []:
        if not isinstance(prompt, dict):
            continue
        source_id = prompt.get("source_mask_index")
        if source_id is None:
            continue
        for track_id in prompt.get("predicted_track_ids") or []:
            mapping[int(track_id)] = int(source_id)
    return mapping


def _semantic_identity_metrics(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    prediction_metadata: dict[str, Any],
) -> dict[str, float | int]:
    """Score the geometry only against the identity named by each prompt.

    Standard TrackEval is intentionally class agnostic and may associate a
    spatially overlapping prediction with any GT identity.  This companion
    metric does no reassignment: a track initialized by prompt/mask identity N
    receives credit only on GT pixels whose released mask identity is also N.
    """

    track_to_source = _track_to_source_identity(prediction_metadata)
    source_to_tracks: dict[int, list[int]] = defaultdict(list)
    for track_id, source_id in track_to_source.items():
        source_to_tracks[source_id].append(track_id)

    gt_object_frames = 0
    localized_object_frames = 0
    iou_sum = 0.0
    hits_at_05 = 0
    for prediction, ground_truth in zip(pred_masks, gt_masks, strict=True):
        for source_id in np.unique(ground_truth):
            source_id = int(source_id)
            if source_id <= 0:
                continue
            gt_object_frames += 1
            tracks = source_to_tracks.get(source_id, [])
            predicted = (
                np.isin(prediction, tracks) if tracks else np.zeros_like(prediction, dtype=bool)
            )
            if np.any(predicted):
                localized_object_frames += 1
            expected = ground_truth == source_id
            intersection = int(np.count_nonzero(predicted & expected))
            union = int(np.count_nonzero(predicted | expected))
            iou = intersection / union if union else 0.0
            iou_sum += iou
            hits_at_05 += int(iou >= 0.5)

    denominator = max(gt_object_frames, 1)
    return {
        "prompt_count": len(prediction_metadata.get("prompts") or []),
        "predicted_track_count": len(track_to_source),
        "initialized_identity_count": len(source_to_tracks),
        "gt_object_frames": gt_object_frames,
        "localized_object_frames": localized_object_frames,
        "identity_mask_iou_sum": iou_sum,
        "identity_mask_iou_mean": iou_sum / denominator,
        "identity_mask_hits_at_0_5": hits_at_05,
        "identity_mask_accuracy_at_0_5": hits_at_05 / denominator,
    }


def main() -> None:
    args = _parse_args()
    dataset_spec = TRACKING_DATASETS[args.dataset_protocol]
    expected_revision = str(dataset_spec["revision"])
    if args.revision is None:
        args.revision = expected_revision
    if args.revision != expected_revision:
        raise ConfigError(
            f"RPX {args.dataset_protocol} tracking is pinned to dataset revision "
            f"{expected_revision}; "
            f"got {args.revision}."
        )
    if args.max_clips is not None and args.max_clips < 1:
        raise ConfigError("--max-clips must be at least 1.")
    if args.max_frames is not None and args.max_frames < 2:
        raise ConfigError("--max-frames must be at least 2.")
    if args.resume_predictions and not args.save_predictions:
        raise ConfigError("--resume-predictions requires --save-predictions.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rpx_git_sha = _rpx_git_sha()
    previous_metadata: dict[str, Any] = {}
    previous_metadata_path = output_dir / "run_metadata.json"
    if previous_metadata_path.is_file():
        try:
            previous_metadata = json.loads(previous_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_metadata = {}
    previous_cells = _previous_cells(output_dir)
    tracker_class = TRACKER_CLASSES[args.model]
    expected_shape = EXPECTED_SHAPES[args.dataset_protocol]
    text_initialized = tracker_class.prompt_type == "text"
    detector_initialized = tracker_class.prompt_type == "detector"
    score_start = 0 if (detector_initialized or text_initialized) else 1
    if text_initialized and not args.text_vocab:
        raise ConfigError(f"{args.model} requires --text-vocab.")
    vocabulary = TrackingTextVocabulary(args.text_vocab) if text_initialized else None
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        # The canonical smoke gate is one partial clip. Restricting the
        # manifest before snapshot selection avoids downloading all 99 Easy
        # cells merely to validate eight frames. Production leaves this unset.
        download_max_samples = (
            args.max_frames if args.max_clips == 1 and args.max_frames is not None else None
        )
        manifest_path = download_split(
            task=TaskType.OBJECT_TRACKING,
            split=args.split,
            repo_id=args.repo,
            cache_dir=args.cache_dir,
            revision=args.revision,
            max_workers=args.dataset_workers,
            max_samples=download_max_samples,
            manifest_name=str(dataset_spec["manifest_name"]),
        )
    clips = _load_clips(manifest_path, args.split)
    if args.max_clips is None and args.max_frames is None:
        _validate_split(clips, args.split, args.dataset_protocol)
    if args.max_clips is not None:
        clips = clips[: args.max_clips]

    rows: list[dict[str, Any]] = []
    tracker: Any | None = None
    cache_hits = 0
    inferred_clips = 0
    forwards = 0
    run_started = time.time()
    scratch_root = output_dir / ".scratch"

    for clip_index, clip in enumerate(clips, start=1):
        samples = clip.samples[: args.max_frames] if args.max_frames else clip.samples
        previous_row = previous_cells.get((clip.scene, clip.phase), {})
        clip_prediction_metadata: dict[str, Any] = {}
        predictions = (
            _clip_predictions(
                clip,
                samples,
                output_dir,
                args.model,
                rpx_git_sha,
                args.dataset_protocol,
            )
            if args.resume_predictions
            else None
        )
        if predictions is not None:
            clip_prediction_metadata = _load_open_vocabulary_metadata(output_dir, clip)
            cache_hits += 1
            latencies = [0.0] * len(samples)
            peak_allocated_mb = _finite_or_nan(previous_row.get("peak_gpu_memory_allocated_mb"))
            peak_reserved_mb = _finite_or_nan(previous_row.get("peak_gpu_memory_reserved_mb"))
            clip_wall_time_s = _finite_or_nan(previous_row.get("clip_wall_time_s"))
            print(f"[{clip_index}/{len(clips)}] resume {clip.key}: {len(samples)} frames")
        else:
            if tracker is None:
                tracker = tracker_class(device=args.device)
            video_dir = _stage_video(clip, samples, scratch_root / clip.key)
            try:
                torch.cuda.reset_peak_memory_stats()
                clip_started = time.perf_counter()
                if text_initialized:
                    assert vocabulary is not None
                    text_prompts = vocabulary.prompts_for(
                        clip.scene, args.dataset_protocol, clip.phase
                    )
                    predictions, latencies = tracker.track(
                        video_dir=video_dir,
                        frame_shape=expected_shape,
                        frame_count=len(samples),
                        text_prompts=text_prompts,
                    )
                else:
                    first_mask = _load_mask_file(
                        _resolve(str(samples[0]["mask"]), clip.root), expected_shape
                    )
                    predictions, latencies = tracker.track(
                        video_dir=video_dir,
                        first_frame_mask=first_mask,
                        frame_count=len(samples),
                    )
                metadata_exporter = getattr(tracker, "prediction_metadata", None)
                if callable(metadata_exporter):
                    exported = metadata_exporter()
                    if isinstance(exported, dict):
                        clip_prediction_metadata = exported
                torch.cuda.synchronize()
                clip_wall_time_s = time.perf_counter() - clip_started
                peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
                peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            finally:
                shutil.rmtree(video_dir, ignore_errors=True)
            inferred_clips += 1
            forwards += max(0, len(samples) - score_start)
            if args.save_predictions:
                for sample, prediction in zip(samples, predictions, strict=True):
                    _atomic_mask(_prediction_path(output_dir, clip, sample), prediction)
                metadata_name = None
                if clip_prediction_metadata:
                    metadata_name = f"{clip.key}.json"
                    _atomic_json(
                        output_dir / "open_vocabulary_predictions" / metadata_name,
                        clip_prediction_metadata,
                    )
                _write_complete_marker(
                    clip,
                    len(samples),
                    output_dir,
                    args.model,
                    tracker_class,
                    rpx_git_sha,
                    metadata_name,
                    args.dataset_protocol,
                )
            print(f"[{clip_index}/{len(clips)}] inferred {clip.key}: {len(samples)} frames")

        gt_masks = [
            _load_mask_file(_resolve(str(sample["mask"]), clip.root), expected_shape)
            for sample in samples
        ]
        # Mask/box-prompted trackers receive GT spatial information on frame 0,
        # so it is excluded. Detector- and text-driven trackers receive no GT
        # spatial prompt and are evaluated on every frame, including frame 0.
        metrics = paper_tracking_metrics(
            pred_masks=predictions[score_start:],
            gt_masks=gt_masks[score_start:],
        )
        semantic_metrics = (
            _semantic_identity_metrics(
                predictions[score_start:],
                gt_masks[score_start:],
                clip_prediction_metadata,
            )
            if text_initialized
            else None
        )
        measured_latencies = [value for value in latencies[score_start:] if value > 0]
        latency_ms = float(np.median(measured_latencies)) if measured_latencies else np.nan
        if not np.isfinite(latency_ms):
            latency_ms = _finite_or_nan(previous_row.get("latency_ms"))
        parameter_count = (
            tracker.parameter_count
            if tracker is not None
            else previous_row.get("parameter_count", previous_metadata.get("parameter_count"))
        )
        row: dict[str, Any] = {
            "model": args.model,
            "task": "object_tracking",
            "dataset_protocol": args.dataset_protocol,
            "split": args.split,
            "scene": clip.scene,
            "phase": clip.phase,
            "n_frames": len(samples),
            "n_scored_frames": len(samples) - score_start,
            "latency_ms": latency_ms,
            "throughput_fps": 1000.0 / latency_ms if latency_ms > 0 else np.nan,
            "peak_gpu_memory_allocated_mb": peak_allocated_mb,
            "peak_gpu_memory_reserved_mb": peak_reserved_mb,
            "clip_wall_time_s": clip_wall_time_s,
            "parameter_count": parameter_count,
        }
        row.update({f"metric:{name}": metrics[name] for name in PAPER_TRACKING_METRICS})
        if semantic_metrics is not None:
            row.update({f"semantic:{name}": value for name, value in semantic_metrics.items()})
        rows.append(row)
        _write_tables(rows, output_dir)

    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    phase_summary: dict[str, dict[str, float]] = {}
    for phase in sorted({int(row["phase"]) for row in rows}):
        phase_rows = [row for row in rows if int(row["phase"]) == phase]
        phase_summary[str(phase)] = {
            metric: float(np.mean([row[f"metric:{metric}"] for row in phase_rows]))
            for metric in PAPER_TRACKING_METRICS
        }
    semantic_summary: dict[str, Any] | None = None
    if text_initialized:
        gt_object_frames = int(sum(int(row.get("semantic:gt_object_frames", 0)) for row in rows))
        localized_object_frames = int(
            sum(int(row.get("semantic:localized_object_frames", 0)) for row in rows)
        )
        identity_iou_sum = float(
            sum(float(row.get("semantic:identity_mask_iou_sum", 0.0)) for row in rows)
        )
        identity_hits = int(
            sum(int(row.get("semantic:identity_mask_hits_at_0_5", 0)) for row in rows)
        )
        denominator = max(gt_object_frames, 1)
        semantic_summary = {
            "definition": (
                "No cross-identity reassignment: each predicted track is scored "
                "only against the released RPX mask identity named by its prompt."
            ),
            "prompt_count": int(sum(int(row.get("semantic:prompt_count", 0)) for row in rows)),
            "predicted_track_count": int(
                sum(int(row.get("semantic:predicted_track_count", 0)) for row in rows)
            ),
            "initialized_identity_count": int(
                sum(int(row.get("semantic:initialized_identity_count", 0)) for row in rows)
            ),
            "gt_object_frames": gt_object_frames,
            "localized_object_frames": localized_object_frames,
            "identity_mask_iou_mean": identity_iou_sum / denominator,
            "identity_mask_hits_at_0_5": identity_hits,
            "identity_mask_accuracy_at_0_5": identity_hits / denominator,
        }
    result = {
        "model": args.model,
        "rpx_git_sha": rpx_git_sha,
        "evaluator_git_sha": os.environ.get("RPX_EVALUATOR_GIT_SHA", rpx_git_sha),
        "resume_compatible_git_shas": sorted(_resume_compatible_git_shas()),
        "task": "object_tracking",
        "dataset_protocol": args.dataset_protocol,
        "split": args.split,
        "protocol": {
            "initialization": (
                "fixed_scene_text_vocabulary_primary_color_plus_canonical_name"
                if text_initialized
                else "detector_every_frame_no_rpx_prompt"
                if detector_initialized
                else f"ground_truth_first_frame_{tracker_class.prompt_type}"
            ),
            "scored_frames": (
                "0_to_end (all frames)"
                if (detector_initialized or text_initialized)
                else "1_to_end (initialization frame excluded)"
            ),
            "association_representation": "tight_boxes_derived_from_instance_masks",
            "association_iou_threshold": 0.5,
            "association_semantics": "class_agnostic_geometry",
            "metric_implementation": "TrackEval",
            "hota": "mean_over_0.05_to_0.95",
            "open_vocabulary": bool(getattr(tracker_class, "vocabulary_name", None)),
            "vocabulary": getattr(tracker_class, "vocabulary_name", None),
            "vocabulary_size": getattr(tracker_class, "vocabulary_size", None),
            "text_vocabulary_revision": (TEXT_VOCAB_REVISION if text_initialized else None),
            "text_vocabulary_sha256": (TEXT_VOCAB_SHA256 if text_initialized else None),
        },
        "dataset": {
            "repo": args.repo,
            "revision": args.revision,
            "protocol": args.dataset_protocol,
            "manifest_name": dataset_spec["manifest_name"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "model_checkpoint": {
            "repo": tracker_class.model_id,
            "revision": tracker_class.model_revision,
            "filename": tracker_class.checkpoint_filename,
            "sha256": (
                getattr(tracker, "checkpoint_sha256", None)
                if tracker is not None
                else (previous_metadata.get("model_checkpoint") or {}).get("sha256")
            ),
        },
        "clips": len(rows),
        "frames": int(sum(row["n_frames"] for row in rows)),
        "phase_metrics": phase_summary,
        "semantic_identity_metrics": semantic_summary,
        "prediction_stats": {
            "complete_clip_cache_hits": cache_hits,
            "inferred_clips": inferred_clips,
            "model_propagation_frames": forwards,
        },
        "initial_inference_stats": (
            previous_metadata.get("initial_inference_stats")
            or previous_metadata.get("prediction_stats")
            if args.resume_predictions
            else {
                "complete_clip_cache_hits": cache_hits,
                "inferred_clips": inferred_clips,
                "model_propagation_frames": forwards,
            }
        ),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_compute_capability": list(torch.cuda.get_device_capability()),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
    }
    _atomic_json(output_dir / "result.json", result)
    _atomic_json(
        output_dir / "run_metadata.json",
        {
            **result,
            "started_unix": run_started,
            "finished_unix": time.time(),
            "parameter_count": (
                tracker.parameter_count
                if tracker is not None
                else previous_metadata.get("parameter_count")
            ),
            "pid": os.getpid(),
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

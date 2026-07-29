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
from PIL import Image

DEFAULT_DATASET_REPO = "IRVLUTD/RPX"
PINNED_DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
EXPECTED_SPLITS = {
    "easy": (24750, 99),
    "medium": (24750, 99),
    "hard": (25500, 102),
}
EXPECTED_SHAPE = (480, 640)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracking_models import SAM2Tracker  # noqa: E402

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
            "RPX D3: initialize a tracker with the GT first-frame mask and "
            "evaluate complete scene-phase clips with official TrackEval."
        )
    )
    parser.add_argument("--model", choices=["sam2"], default="sam2")
    parser.add_argument("--split", choices=sorted(EXPECTED_SPLITS), required=True)
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path")
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


def _load_mask_file(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise DatasetError(f"Tracking mask {path} must be 2-D, got {mask.shape}.")
    mask = mask.astype(np.int32, copy=False)
    if mask.shape != EXPECTED_SHAPE:
        raise DatasetError(
            f"Tracking mask {path} has shape {mask.shape}; expected {EXPECTED_SHAPE}."
        )
    if np.any(mask < 0):
        raise DatasetError(f"Tracking mask {path} contains negative IDs.")
    return mask


def _load_prediction(path: Path) -> np.ndarray | None:
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
        mask.shape != EXPECTED_SHAPE
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
        raise DatasetError(
            f"Expected object_tracking manifest, got {manifest.get('task')!r}."
        )
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
            raise DatasetError(
                f"{scene}/phase {phase} frame indices are not contiguous from zero."
            )
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


def _validate_split(clips: list[Clip], split: str) -> None:
    expected_frames, expected_clips = EXPECTED_SPLITS[split]
    actual_frames = sum(len(clip.samples) for clip in clips)
    if (actual_frames, len(clips)) != (expected_frames, expected_clips):
        raise DatasetError(
            f"{split} has {actual_frames} frames/{len(clips)} clips; "
            f"expected {expected_frames}/{expected_clips}."
        )
    if any(len(clip.samples) != 250 for clip in clips):
        raise DatasetError(f"{split} contains a clip that is not exactly 250 frames.")


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
) -> list[np.ndarray] | None:
    marker_path = output_dir / "predictions" / clip.scene / str(clip.phase) / "_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("model") != "sam2" or marker.get("frames") != len(samples):
        return None
    predictions: list[np.ndarray] = []
    for sample in samples:
        prediction = _load_prediction(_prediction_path(output_dir, clip, sample))
        if prediction is None:
            return None
        predictions.append(prediction)
    return predictions


def _write_complete_marker(
    clip: Clip,
    sample_count: int,
    output_dir: Path,
) -> None:
    marker = output_dir / "predictions" / clip.scene / str(clip.phase) / "_complete.json"
    _atomic_json(
        marker,
        {
            "model": "sam2",
            "model_id": SAM2Tracker.model_id,
            "model_revision": SAM2Tracker.model_revision,
            "frames": sample_count,
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


def main() -> None:
    args = _parse_args()
    if args.revision != PINNED_DATASET_REVISION:
        raise ConfigError(
            f"RPX D3 is pinned to dataset revision {PINNED_DATASET_REVISION}; "
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
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        # The canonical smoke gate is one partial clip. Restricting the
        # manifest before snapshot selection avoids downloading all 99 Easy
        # cells merely to validate eight frames. Production leaves this unset.
        download_max_samples = (
            args.max_frames
            if args.max_clips == 1 and args.max_frames is not None
            else None
        )
        manifest_path = download_split(
            task=TaskType.OBJECT_TRACKING,
            split=args.split,
            repo_id=args.repo,
            cache_dir=args.cache_dir,
            revision=args.revision,
            max_workers=args.dataset_workers,
            max_samples=download_max_samples,
        )
    clips = _load_clips(manifest_path, args.split)
    if args.max_clips is None and args.max_frames is None:
        _validate_split(clips, args.split)
    if args.max_clips is not None:
        clips = clips[: args.max_clips]

    rows: list[dict[str, Any]] = []
    tracker: SAM2Tracker | None = None
    cache_hits = 0
    inferred_clips = 0
    forwards = 0
    run_started = time.time()
    scratch_root = output_dir / ".scratch"

    for clip_index, clip in enumerate(clips, start=1):
        samples = clip.samples[: args.max_frames] if args.max_frames else clip.samples
        predictions = (
            _clip_predictions(clip, samples, output_dir) if args.resume_predictions else None
        )
        if predictions is not None:
            cache_hits += 1
            latencies = [0.0] * len(samples)
            print(f"[{clip_index}/{len(clips)}] resume {clip.key}: {len(samples)} frames")
        else:
            if tracker is None:
                tracker = SAM2Tracker(device=args.device)
            first_mask = _load_mask_file(_resolve(str(samples[0]["mask"]), clip.root))
            video_dir = _stage_video(clip, samples, scratch_root / clip.key)
            try:
                predictions, latencies = tracker.track(
                    video_dir=video_dir,
                    first_frame_mask=first_mask,
                    frame_count=len(samples),
                )
            finally:
                shutil.rmtree(video_dir, ignore_errors=True)
            inferred_clips += 1
            forwards += max(0, len(samples) - 1)
            if args.save_predictions:
                for sample, prediction in zip(samples, predictions, strict=True):
                    _atomic_mask(_prediction_path(output_dir, clip, sample), prediction)
                _write_complete_marker(clip, len(samples), output_dir)
            print(f"[{clip_index}/{len(clips)}] inferred {clip.key}: {len(samples)} frames")

        gt_masks = [
            _load_mask_file(_resolve(str(sample["mask"]), clip.root)) for sample in samples
        ]
        metrics = paper_tracking_metrics(pred_masks=predictions, gt_masks=gt_masks)
        measured_latencies = [value for value in latencies[1:] if value > 0]
        row: dict[str, Any] = {
            "model": args.model,
            "task": "object_tracking",
            "split": args.split,
            "scene": clip.scene,
            "phase": clip.phase,
            "n_frames": len(samples),
            "latency_ms": (
                float(np.median(measured_latencies)) if measured_latencies else np.nan
            ),
        }
        row.update({f"metric:{name}": metrics[name] for name in PAPER_TRACKING_METRICS})
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
    result = {
        "model": args.model,
        "task": "object_tracking",
        "split": args.split,
        "protocol": {
            "initialization": "ground_truth_first_frame_instance_masks",
            "association_representation": "tight_boxes_derived_from_instance_masks",
            "association_iou_threshold": 0.5,
            "metric_implementation": "TrackEval",
            "hota": "mean_over_0.05_to_0.95",
        },
        "dataset": {
            "repo": args.repo,
            "revision": args.revision,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "model_checkpoint": {
            "repo": SAM2Tracker.model_id,
            "revision": SAM2Tracker.model_revision,
        },
        "clips": len(rows),
        "frames": int(sum(row["n_frames"] for row in rows)),
        "phase_metrics": phase_summary,
        "prediction_stats": {
            "complete_clip_cache_hits": cache_hits,
            "inferred_clips": inferred_clips,
            "model_propagation_frames": forwards,
        },
    }
    _atomic_json(output_dir / "result.json", result)
    _atomic_json(
        output_dir / "run_metadata.json",
        {
            **result,
            "started_unix": run_started,
            "finished_unix": time.time(),
            "parameter_count": tracker.parameter_count if tracker is not None else None,
            "pid": os.getpid(),
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

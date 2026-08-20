#!/usr/bin/env python3
"""Run one registered NVS model on RPX single-object (SOS) archives."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _members(archive: tarfile.TarFile, suffix: str) -> dict[int, tarfile.TarInfo]:
    result = {}
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith(suffix):
            result[int(Path(member.name).stem)] = member
    return result


def _read(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = archive.extractfile(member)
    if stream is None:
        raise OSError(f"could not read {member.name}")
    return stream.read()


def _rgb(payload: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)


def _depth(payload: bytes) -> np.ndarray:
    raw = np.asarray(Image.open(io.BytesIO(payload)))
    return raw.astype(np.float32) / 1000.0


def _pose(payload: bytes) -> np.ndarray:
    from rpx_benchmark.ar_pose_audit import pose_from_npz_bytes

    return pose_from_npz_bytes(payload)


def _ar_comparison(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Compare board visibility and pose in a generated and real target image."""
    from rpx_benchmark.ar_pose_audit import (
        DEFAULT_CAMERA_MATRIX,
        estimate_board_pose,
        make_detector,
        rotation_error_deg,
    )

    detector = make_detector()
    pred_obs = estimate_board_pose(pred, detector, DEFAULT_CAMERA_MATRIX)
    gt_obs = estimate_board_pose(gt, detector, DEFAULT_CAMERA_MATRIX)
    row: dict[str, Any] = {
        "gt_board_detected": gt_obs is not None,
        "prediction_board_detected": pred_obs is not None,
        "gt_marker_ids": [] if gt_obs is None else list(gt_obs.marker_ids),
        "prediction_marker_ids": [] if pred_obs is None else list(pred_obs.marker_ids),
    }
    if pred_obs is not None and gt_obs is not None:
        gt_ids = set(gt_obs.marker_ids)
        pred_ids = set(pred_obs.marker_ids)
        row.update(
            {
                "marker_id_recall": len(gt_ids & pred_ids) / max(1, len(gt_ids)),
                "board_pose_translation_error_m": float(
                    np.linalg.norm(
                        pred_obs.camera_from_board[:3, 3] - gt_obs.camera_from_board[:3, 3]
                    )
                ),
                "board_pose_rotation_error_deg": rotation_error_deg(
                    gt_obs.camera_from_board, pred_obs.camera_from_board
                ),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="depthsplat")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ar-check", action="store_true")
    args = parser.parse_args()

    from nvs_models import MODEL_DISPLAY_NAMES
    from run_nvs import _build_model

    from rpx_benchmark.nvs_eval import evaluate_single_sample

    objects_root = args.dataset_root / "objects"
    sequences = (
        [objects_root / item / "0" for item in args.objects]
        if args.objects
        else sorted(objects_root.glob("*/0"))
    )
    sequences = [path for path in sequences if path.is_dir()]
    if args.max_objects is not None:
        sequences = sequences[: args.max_objects]
    if not sequences:
        raise SystemExit(f"No SOS object sequences under {objects_root}")

    adapter = _build_model(args.model, args.device)
    display = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    output = args.output_root / display / "sos"
    rows = []

    for sequence in sequences:
        rgb_path = sequence / "rgb.tar"
        depth_path = sequence / "depth.tar"
        pose_path = sequence / "labels" / "cam_pose" / "v1.tar"
        with (
            tarfile.open(rgb_path) as rgb_tar,
            tarfile.open(depth_path) as depth_tar,
            tarfile.open(pose_path) as pose_tar,
        ):
            rgb_members = _members(rgb_tar, ".png")
            depth_members = _members(depth_tar, ".png")
            pose_members = _members(pose_tar, ".npz")
            frames = sorted(set(rgb_members) & set(depth_members) & set(pose_members))
            if len(frames) < 3:
                continue
            # Explicit forward extrapolation: K=2 contexts come from the
            # first 40% of the trial and targets from the final 40%, leaving
            # a 20% temporal guard band.
            context_end = max(1, int(np.floor(0.40 * len(frames))) - 1)
            context_ids = [frames[0], frames[context_end]]
            target_start = min(len(frames) - 1, int(np.ceil(0.60 * len(frames))))
            available = frames[target_start:]
            target_positions = np.linspace(
                0, len(available) - 1, min(args.max_targets, len(available)), dtype=int
            )
            target_ids = [available[int(i)] for i in np.unique(target_positions)]
            context_rgbs = [_rgb(_read(rgb_tar, rgb_members[i])) for i in context_ids]
            context_depths = [_depth(_read(depth_tar, depth_members[i])) for i in context_ids]
            context_poses = [_pose(_read(pose_tar, pose_members[i])) for i in context_ids]

            for target_id in target_ids:
                gt_rgb = _rgb(_read(rgb_tar, rgb_members[target_id]))
                gt_depth = _depth(_read(depth_tar, depth_members[target_id]))
                target_pose = _pose(_read(pose_tar, pose_members[target_id]))
                rendered = adapter(context_rgbs, context_depths, context_poses, target_pose)
                metrics = evaluate_single_sample(
                    pred_rgb=rendered["rgb"],
                    gt_rgb=gt_rgb,
                    pred_depth=rendered.get("depth"),
                    gt_depth=gt_depth,
                    compute_lpips=False,
                )
                row: dict[str, Any] = {
                    "object_id": sequence.parent.name,
                    "target_frame": target_id,
                    "context_frames": context_ids,
                    **metrics,
                }
                if not args.skip_ar_check:
                    try:
                        row["ar_tag_check"] = _ar_comparison(rendered["rgb"], gt_rgb)
                    except (ImportError, RuntimeError) as error:
                        row["ar_tag_check"] = {"available": False, "reason": str(error)}
                rows.append(row)

                stem = f"{target_id:05d}"
                frame_dir = output / "prediction_frames" / sequence.parent.name
                gt_dir = output / "target_frames" / sequence.parent.name
                frame_dir.mkdir(parents=True, exist_ok=True)
                gt_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rendered["rgb"]).save(frame_dir / f"{stem}.png")
                Image.fromarray(gt_rgb).save(gt_dir / f"{stem}.png")

    if not rows:
        raise SystemExit("No SOS samples were evaluated")
    metric_names = ("psnr", "ssim", "depth_absrel", "depth_rmse", "depth_delta1")
    aggregate = {
        name: float(np.mean([row[name] for row in rows if row.get(name) is not None]))
        for name in metric_names
        if any(row.get(name) is not None for row in rows)
    }
    payload = {
        "model": args.model,
        "display_name": display,
        "dataset": "RPX single-object sequences",
        "num_samples": len(rows),
        "aggregate": aggregate,
        "samples": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"num_samples": len(rows), **aggregate}, indent=2))
    print(f"SOS results: {output}")


if __name__ == "__main__":
    main()

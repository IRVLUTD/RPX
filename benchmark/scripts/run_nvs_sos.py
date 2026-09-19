#!/usr/bin/env python3
"""Run one registered NVS model on RPX single-object (SOS) archives."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


PROTOCOLS = ("identity", "interpolation", "sliding-near", "far", "all")
DEFAULT_WINDOW_STARTS = (0, 100, 200, 300)
DEFAULT_CONTEXT_OFFSETS = (0, 49)
DEFAULT_TARGET_OFFSETS = (55, 60, 70)
METRIC_NAMES = ("psnr", "ssim", "depth_absrel", "depth_rmse", "depth_delta1")


def _members(archive: tarfile.TarFile, suffix: str | tuple[str, ...]) -> dict[int, tarfile.TarInfo]:
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
    """Decode either legacy NPZ or release-format ``(7,)`` NPY poses."""
    from scipy.spatial.transform import Rotation

    loaded = np.load(io.BytesIO(payload))
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            position = np.asarray(loaded["position"], dtype=np.float64).reshape(3)
            quaternion = np.asarray(loaded["orientation"], dtype=np.float64).reshape(4)
        finally:
            loaded.close()
    else:
        pose7 = np.asarray(loaded, dtype=np.float64).reshape(7)
        position = pose7[:3]
        quaternion = pose7[3:]

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


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


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _local_samples(
    frames: Sequence[int],
    protocol: str,
    window_starts: Sequence[int],
    context_offsets: Sequence[int],
    target_offsets: Sequence[int],
    max_targets: int,
) -> list[dict[str, Any]]:
    if len(context_offsets) != 2:
        raise ValueError(f"K=2 requires two context offsets, got {tuple(context_offsets)}")
    frame_set = set(frames)
    samples: list[dict[str, Any]] = []
    for start in window_starts:
        contexts = tuple(start + offset for offset in context_offsets)
        if any(frame not in frame_set for frame in contexts):
            continue
        if protocol == "identity":
            targets = (contexts[-1],)
        elif protocol == "interpolation":
            midpoint = int(round((contexts[0] + contexts[1]) / 2.0))
            targets = (midpoint,)
        elif protocol == "sliding-near":
            targets = tuple(start + offset for offset in target_offsets[:max_targets])
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unsupported local protocol: {protocol}")
        for target in targets:
            if target not in frame_set:
                continue
            samples.append(
                {
                    "protocol": protocol,
                    "window_start": int(start),
                    "context_frames": list(contexts),
                    "target_frame": int(target),
                }
            )
    return samples


def _far_samples(frames: Sequence[int], max_targets: int) -> list[dict[str, Any]]:
    context_end = max(1, int(np.floor(0.40 * len(frames))) - 1)
    contexts = [int(frames[0]), int(frames[context_end])]
    target_start = min(len(frames) - 1, int(np.ceil(0.60 * len(frames))))
    available = frames[target_start:]
    positions = np.linspace(0, len(available) - 1, min(max_targets, len(available)), dtype=int)
    return [
        {
            "protocol": "far",
            "window_start": None,
            "context_frames": contexts,
            "target_frame": int(available[int(position)]),
        }
        for position in np.unique(positions)
    ]


def _protocol_samples(
    frames: Sequence[int],
    protocol: str,
    *,
    window_starts: Sequence[int] = DEFAULT_WINDOW_STARTS,
    context_offsets: Sequence[int] = DEFAULT_CONTEXT_OFFSETS,
    target_offsets: Sequence[int] = DEFAULT_TARGET_OFFSETS,
    max_targets: int = 5,
) -> list[dict[str, Any]]:
    requested = (
        ("identity", "interpolation", "sliding-near", "far") if protocol == "all" else (protocol,)
    )
    samples: list[dict[str, Any]] = []
    for item in requested:
        if item == "far":
            samples.extend(_far_samples(frames, max_targets))
        else:
            samples.extend(
                _local_samples(
                    frames,
                    item,
                    window_starts,
                    context_offsets,
                    target_offsets,
                    max_targets,
                )
            )
    return samples


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _target_baseline(
    context_ids: Sequence[int],
    context_poses: Sequence[np.ndarray],
    target_id: int,
    target_pose: np.ndarray,
) -> dict[str, Any]:
    translations = [
        float(np.linalg.norm(np.asarray(pose)[:3, 3] - np.asarray(target_pose)[:3, 3]))
        for pose in context_poses
    ]
    nearest_index = int(np.argmin(translations))
    return {
        "nearest_context_frame": int(context_ids[nearest_index]),
        "frame_gap": abs(int(target_id) - int(context_ids[nearest_index])),
        "translation_m": translations[nearest_index],
        "rotation_deg": _rotation_error_deg(context_poses[nearest_index], target_pose),
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in rows if row.get(name) is not None]))
        for name in METRIC_NAMES
        if any(row.get(name) is not None for row in rows)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="depthsplat")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="sliding-near")
    parser.add_argument(
        "--window-starts",
        type=_csv_ints,
        default=DEFAULT_WINDOW_STARTS,
        help="comma-separated local-window starts (default: 0,100,200,300)",
    )
    parser.add_argument(
        "--context-offsets",
        type=_csv_ints,
        default=DEFAULT_CONTEXT_OFFSETS,
        help="two K=2 offsets within each local window (default: 0,49)",
    )
    parser.add_argument(
        "--target-offsets",
        type=_csv_ints,
        default=DEFAULT_TARGET_OFFSETS,
        help="near-extrapolation offsets within each local window (default: 55,60,70)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ar-check", action="store_true")
    args = parser.parse_args()
    if args.max_targets < 1:
        parser.error("--max-targets must be at least 1")
    if len(args.context_offsets) != 2:
        parser.error("--context-offsets must contain exactly two values for K=2")
    if args.context_offsets[0] >= args.context_offsets[1]:
        parser.error("--context-offsets must be strictly increasing")
    if args.protocol in ("sliding-near", "all") and any(
        offset <= args.context_offsets[-1] for offset in args.target_offsets
    ):
        parser.error("near target offsets must be after the second context offset")

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
    output = args.output_root / display / "sos" / args.protocol
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
            rgb_members = _members(rgb_tar, (".png", ".webp", ".jpg", ".jpeg"))
            depth_members = _members(depth_tar, ".png")
            pose_members = _members(pose_tar, (".npz", ".npy"))
            frames = sorted(set(rgb_members) & set(depth_members) & set(pose_members))
            if len(frames) < 3:
                continue
            samples = _protocol_samples(
                frames,
                args.protocol,
                window_starts=args.window_starts,
                context_offsets=args.context_offsets,
                target_offsets=args.target_offsets,
                max_targets=args.max_targets,
            )
            for sample in samples:
                context_ids = sample["context_frames"]
                target_id = sample["target_frame"]
                print(
                    f"[sos] object={sequence.parent.name} protocol={sample['protocol']} "
                    f"context={context_ids} target={target_id}"
                )
                context_rgbs = [_rgb(_read(rgb_tar, rgb_members[i])) for i in context_ids]
                context_depths = [_depth(_read(depth_tar, depth_members[i])) for i in context_ids]
                context_poses = [_pose(_read(pose_tar, pose_members[i])) for i in context_ids]
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
                    "protocol": sample["protocol"],
                    "window_start": sample["window_start"],
                    "target_frame": target_id,
                    "context_frames": context_ids,
                    "target_baseline": _target_baseline(
                        context_ids, context_poses, target_id, target_pose
                    ),
                    **metrics,
                }
                if not args.skip_ar_check:
                    try:
                        row["ar_tag_check"] = _ar_comparison(rendered["rgb"], gt_rgb)
                    except (ImportError, RuntimeError) as error:
                        row["ar_tag_check"] = {"available": False, "reason": str(error)}
                rows.append(row)

                stem = f"ctx{context_ids[0]:05d}_{context_ids[1]:05d}__tgt{target_id:05d}"
                frame_dir = output / "prediction_frames" / sequence.parent.name
                gt_dir = output / "target_frames" / sequence.parent.name
                frame_dir.mkdir(parents=True, exist_ok=True)
                gt_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rendered["rgb"]).save(frame_dir / f"{stem}.png")
                Image.fromarray(gt_rgb).save(gt_dir / f"{stem}.png")

    if not rows:
        raise SystemExit("No SOS samples were evaluated")
    aggregate = _aggregate(rows)
    by_protocol = {
        protocol: _aggregate([row for row in rows if row["protocol"] == protocol])
        for protocol in sorted({row["protocol"] for row in rows})
    }
    payload = {
        "model": args.model,
        "display_name": display,
        "dataset": "RPX single-object sequences",
        "num_samples": len(rows),
        "sampling_protocol": {
            "requested": args.protocol,
            "context_count": 2,
            "window_starts": list(args.window_starts),
            "context_offsets": list(args.context_offsets),
            "target_offsets": list(args.target_offsets),
            "definitions": {
                "identity": "target equals the second context; adapter sanity diagnostic",
                "interpolation": "target is the temporal midpoint between two contexts",
                "sliding-near": "target is shortly after the second context in four local windows",
                "far": "contexts from the first 40% and targets from the final 40%",
            },
        },
        "aggregate": aggregate,
        "by_protocol": by_protocol,
        "samples": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"num_samples": len(rows), **aggregate}, indent=2))
    print(f"SOS results: {output}")


if __name__ == "__main__":
    main()

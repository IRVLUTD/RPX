#!/usr/bin/env python3
"""Audit RPX SOS AR-board poses against synchronized saved camera poses."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np

from rpx_benchmark.ar_pose_audit import (
    DEFAULT_CAMERA_MATRIX,
    audit_sequence,
    write_results,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="RPX dataset checkout/snapshot root"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo", default="anonymous/RPX")
    parser.add_argument(
        "--dataset-revision",
        default="2e2a387f7f93e98c177b2e039c141eacda94e5fc",
        help="Pinned Hugging Face dataset revision represented by --dataset-root",
    )
    parser.add_argument(
        "--objects", nargs="*", help="Optional object IDs; default is every objects/*/0 sequence"
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel object sequences")
    parser.add_argument("--max-frames", type=int, help="Debug limit per sequence")
    parser.add_argument("--fx", type=float, default=float(DEFAULT_CAMERA_MATRIX[0, 0]))
    parser.add_argument("--fy", type=float, default=float(DEFAULT_CAMERA_MATRIX[1, 1]))
    parser.add_argument("--cx", type=float, default=float(DEFAULT_CAMERA_MATRIX[0, 2]))
    parser.add_argument("--cy", type=float, default=float(DEFAULT_CAMERA_MATRIX[1, 2]))
    parser.add_argument(
        "--distortion",
        type=float,
        nargs="*",
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help="OpenCV distortion coefficients; default is zero",
    )
    parser.add_argument("--reprojection-threshold-px", type=float, default=4.0)
    parser.add_argument(
        "--min-markers",
        type=int,
        default=2,
        help="Minimum RANSAC-consistent tags for primary metrics",
    )
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=2.0)
    parser.add_argument(
        "--pose-axis-convention",
        choices=("t265", "opencv"),
        default="t265",
        help="Use t265 for raw save_device_data.py poses; opencv only for pre-converted poses",
    )
    return parser.parse_args()


def _run_one(arguments: tuple[str, np.ndarray, np.ndarray, int | None, float, str, int, float]):
    sequence, K, distortion, max_frames, threshold, axis_convention, min_markers, max_rmse = (
        arguments
    )
    return audit_sequence(
        sequence,
        K,
        distortion,
        max_frames=max_frames,
        reprojection_threshold_px=threshold,
        pose_axis_convention=axis_convention,
        min_markers=min_markers,
        max_reprojection_rmse_px=max_rmse,
    )


def main() -> None:
    args = _arguments()
    object_root = args.dataset_root / "objects"
    if args.objects:
        sequences = [object_root / object_id / "0" for object_id in args.objects]
    else:
        sequences = sorted(path for path in object_root.glob("*/0") if path.is_dir())
    if not sequences:
        raise SystemExit(f"No SOS sequences found under {object_root}")

    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]])
    distortion = np.asarray(args.distortion, dtype=np.float64)
    jobs = [
        (
            str(sequence),
            K,
            distortion,
            args.max_frames,
            args.reprojection_threshold_px,
            args.pose_axis_convention,
            args.min_markers,
            args.max_reprojection_rmse_px,
        )
        for sequence in sequences
    ]
    sequence_metrics = []
    frame_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_one, job): Path(job[0]).parent.name for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            object_id = futures[future]
            metrics, rows = future.result()
            sequence_metrics.append(metrics)
            frame_rows.extend(rows)
            print(
                f"{object_id}: detections={metrics['matched_frames']}/"
                f"{metrics['total_paired_frames']} "
                f"coverage={metrics['detection_coverage']:.1%}",
                flush=True,
            )

    sequence_metrics.sort(key=lambda item: str(item["object_id"]))
    frame_rows.sort(key=lambda item: (str(item["object_id"]), int(item["frame_idx"])))
    metadata = {
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "dataset_root": str(args.dataset_root.resolve()),
        "sensor_frame_assumption": "D435 RGB and saved T265 pose are already aligned",
        "sensor_transform_fitted": False,
        "pose_axis_convention": args.pose_axis_convention,
        "world_origin_comparison": "anchor-relative per sequence",
        "board_geometry": "FewSOL 18-marker board, 70.4 x 50.4 cm, 5.9 cm markers",
        "marker_family": "ALVAR MarkerData IDs 0-17",
        "camera_matrix": K.tolist(),
        "distortion": distortion.tolist(),
        "reprojection_threshold_px": args.reprojection_threshold_px,
        "min_markers": args.min_markers,
        "max_reprojection_rmse_px": args.max_reprojection_rmse_px,
    }
    payload = write_results(args.output_dir, sequence_metrics, frame_rows, metadata)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

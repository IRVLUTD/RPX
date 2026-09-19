#!/usr/bin/env python3
"""Engineering smoke for YOLOE detection plus ByteTrack association.

This is not the RPX D3 paper protocol. YOLOE belongs to the paper's D2 model
roster; D3 trackers are initialized with GT first-frame instance masks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLOE
from ultralytics.utils import ASSETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "bus"],
    )
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)

    return parser.parse_args()


def create_test_video(
    output_path: Path,
    frame_count: int,
    fps: float,
) -> tuple[int, int]:
    source_path = Path(ASSETS) / "bus.jpg"
    image = cv2.imread(str(source_path))

    if image is None:
        raise RuntimeError(
            f"Could not read Ultralytics test image: {source_path}"
        )

    height, width = image.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create test video: {output_path}"
        )

    # Deterministic, slight frame-to-frame motion.
    for frame_index in range(frame_count):
        phase = (
            2.0
            * math.pi
            * frame_index
            / max(frame_count, 1)
        )

        dx = int(round(10.0 * math.sin(phase)))
        dy = int(round(4.0 * math.cos(phase)))

        transform = np.float32(
            [
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
            ]
        )

        frame = cv2.warpAffine(
            image,
            transform,
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )

        writer.write(frame)

    writer.release()

    if not output_path.is_file():
        raise RuntimeError("Test video was not created")

    return width, height


def get_class_name(
    names: Any,
    class_id: int,
) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, (list, tuple)):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    return str(class_id)


def main() -> None:
    args = parse_args()

    print(
        "NOTICE: YOLOE+ByteTrack is an engineering smoke only; "
        "do not report it as an RPX D3 benchmark result."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLOE checkpoint missing: {model_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable inside the container"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU, "
            f"found {torch.cuda.device_count()}"
        )

    input_video = output_dir / "input.mp4"
    annotated_video = output_dir / "annotated.mp4"
    detections_file = output_dir / "detections.json"
    summary_file = output_dir / "summary.json"

    width, height = create_test_video(
        output_path=input_video,
        frame_count=args.frames,
        fps=args.fps,
    )

    model = YOLOE(str(model_path))
    model.set_classes(args.classes)

    result_stream = model.track(
        source=str(input_video),
        stream=True,
        persist=True,
        tracker=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    output_writer = cv2.VideoWriter(
        str(annotated_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    if not output_writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {annotated_video}"
        )

    records: list[dict[str, Any]] = []
    track_counts: Counter[int] = Counter()

    processed_frames = 0
    total_detections = 0
    boxes_finite = True
    boxes_valid = True

    for frame_index, result in enumerate(result_stream):
        processed_frames += 1

        annotated_frame = result.plot()
        output_writer.write(annotated_frame)

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().float().numpy()
        scores = boxes.conf.detach().cpu().float().numpy()
        class_ids = boxes.cls.detach().cpu().long().numpy()

        if boxes.id is None:
            track_ids: list[int | None] = [
                None for _ in range(len(boxes))
            ]
        else:
            ids = boxes.id.detach().cpu().long().numpy()
            track_ids = [int(value) for value in ids]

        for box, score, class_id, track_id in zip(
            xyxy,
            scores,
            class_ids,
            track_ids,
            strict=True,
        ):
            x1, y1, x2, y2 = [
                float(value) for value in box
            ]

            values = np.asarray(
                [x1, y1, x2, y2, float(score)]
            )

            boxes_finite = (
                boxes_finite
                and bool(np.isfinite(values).all())
            )

            boxes_valid = boxes_valid and (
                0.0 <= x1 <= width
                and 0.0 <= x2 <= width
                and 0.0 <= y1 <= height
                and 0.0 <= y2 <= height
                and x2 >= x1
                and y2 >= y1
            )

            if track_id is not None:
                track_counts[track_id] += 1

            records.append(
                {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "class_id": int(class_id),
                    "label": get_class_name(
                        result.names,
                        int(class_id),
                    ),
                    "score": float(score),
                    "box_xyxy": [x1, y1, x2, y2],
                }
            )

            total_detections += 1

    output_writer.release()

    repeated_tracks = {
        str(track_id): count
        for track_id, count in track_counts.items()
        if count >= 2
    }

    checks = {
        "processed_all_frames": (
            processed_frames == args.frames
        ),
        "has_detections": total_detections > 0,
        "has_track_ids": len(track_counts) > 0,
        "has_repeated_tracks": len(repeated_tracks) > 0,
        "boxes_are_finite": boxes_finite,
        "boxes_are_valid": boxes_valid,
        "annotated_video_exists": (
            annotated_video.is_file()
        ),
    }

    summary = {
        "paper_valid_rpx_d3": False,
        "model": str(model_path),
        "classes": args.classes,
        "tracker": args.tracker,
        "device": args.device,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "unique_track_ids": len(track_counts),
        "repeated_tracks": repeated_tracks,
        "checks": checks,
        "passed": all(checks.values()),
    }

    detections_file.write_text(
        json.dumps(records, indent=2) + "\n"
    )

    summary_file.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))

    if not summary["passed"]:
        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "YOLOE tracking smoke failed: "
            + ", ".join(failed_checks)
        )

    print("YOLOE TRACKING SMOKE: PASSED")


if __name__ == "__main__":
    main()

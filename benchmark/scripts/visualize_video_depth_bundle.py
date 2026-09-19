#!/usr/bin/env python3
"""Visualize one exported RPX Video Depth scene-phase prediction bundle.

The bundle is expected to contain:

* ``depth.npz`` with ``depth``, ``frame_indices`` and ``latency_ms``;
* ``rgb/`` with one image per predicted frame;
* optionally ``gt_depth/`` with matching uint16 millimetre PNG files; and
* optionally ``bundle.json`` describing the selected model/scene/phase.

The script writes lossless side-by-side PNG frames and, when ``ffmpeg`` is
available, an H.264 MP4. Prediction and GT use the same fixed depth colour
scale, making visual comparison meaningful across the complete clip.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--depth-min", type=float, default=0.3)
    parser.add_argument("--depth-max", type=float, default=5.0)
    parser.add_argument("--error-max", type=float, default=1.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Render at most this many evenly spaced frames.",
    )
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="Write PNG frames only.",
    )
    args = parser.parse_args()
    if args.depth_min >= args.depth_max:
        parser.error("--depth-min must be smaller than --depth-max")
    if args.error_max <= 0:
        parser.error("--error-max must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    return args


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _turbo(values: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation of Google's Turbo colour map."""
    x = np.clip(values.astype(np.float32), 0.0, 1.0)
    coefficients = np.asarray(
        [
            [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
            [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
            [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
        ],
        dtype=np.float32,
    )
    powers = np.stack([np.ones_like(x), x, x**2, x**3, x**4, x**5], axis=-1)
    rgb = powers @ coefficients.T
    return np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _depth_colour(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = (np.clip(depth, low, high) - low) / (high - low)
    # Near depth is warm and far depth is cool.
    colour = _turbo(1.0 - normalized)
    colour[~valid] = 0
    return colour


def _error_colour(error: np.ndarray, maximum: float, valid: np.ndarray) -> np.ndarray:
    normalized = np.clip(error / maximum, 0.0, 1.0)
    # Black -> yellow -> red highlights large errors without another dependency.
    red = np.asarray(255.0 * normalized, dtype=np.uint8)
    green = np.asarray(255.0 * np.minimum(normalized * 2.0, 1.0), dtype=np.uint8)
    green = np.where(normalized > 0.5, np.asarray(255.0 * (2.0 - 2.0 * normalized), dtype=np.uint8), green)
    blue = np.zeros_like(red)
    colour = np.stack([red, green, blue], axis=-1)
    colour[~valid] = 0
    return colour


def _load_gt(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.float32) / 1000.0
    return raw.astype(np.float32)


def _label(image: Image.Image, title: str) -> Image.Image:
    band = 34
    canvas = Image.new("RGB", (image.width, image.height + band), "black")
    canvas.paste(image, (0, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, fill="white", font=ImageFont.load_default())
    return canvas


def _select_indices(total: int, maximum: int | None) -> np.ndarray:
    if maximum is None or maximum >= total:
        return np.arange(total, dtype=np.int32)
    return np.unique(np.linspace(0, total - 1, maximum).round().astype(np.int32))


def main() -> None:
    args = _parse_args()
    bundle = args.bundle.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else bundle / "visualization"
    )
    frame_output = output / "frames"
    frame_output.mkdir(parents=True, exist_ok=True)

    prediction_path = bundle / "depth.npz"
    if not prediction_path.is_file():
        raise SystemExit(f"Missing prediction container: {prediction_path}")

    with np.load(prediction_path, allow_pickle=False) as payload:
        expected = {"depth", "frame_indices", "latency_ms"}
        if set(payload.files) != expected:
            raise SystemExit(
                f"Unexpected NPZ keys: {sorted(payload.files)}; expected {sorted(expected)}"
            )
        prediction = np.asarray(payload["depth"], dtype=np.float32)
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32)
        latency_ms = float(np.asarray(payload["latency_ms"]).reshape(-1)[0])

    rgb_files = _image_files(bundle / "rgb")
    gt_files = _image_files(bundle / "gt_depth")
    if prediction.ndim != 3:
        raise SystemExit(f"Prediction must be (T,H,W), got {prediction.shape}")
    if len(rgb_files) != prediction.shape[0]:
        raise SystemExit(
            f"RGB/prediction mismatch: {len(rgb_files)} RGB files versus "
            f"{prediction.shape[0]} prediction frames"
        )
    if gt_files and len(gt_files) != prediction.shape[0]:
        raise SystemExit(
            f"GT/prediction mismatch: {len(gt_files)} GT files versus "
            f"{prediction.shape[0]} prediction frames"
        )

    metadata_path = bundle / "bundle.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    selected = _select_indices(prediction.shape[0], args.max_frames)

    print(
        f"Rendering {len(selected)}/{prediction.shape[0]} frames "
        f"for {metadata.get('model', 'video-depth model')} "
        f"{metadata.get('scene', '')}/{metadata.get('phase', '')}"
    )
    print(f"Stored clip latency: {latency_ms:.3f} ms")

    for rendered_index, sequence_index in enumerate(selected):
        rgb = Image.open(rgb_files[int(sequence_index)]).convert("RGB")
        pred = prediction[int(sequence_index)]
        if pred.shape != (rgb.height, rgb.width):
            raise SystemExit(
                f"Frame {sequence_index}: prediction shape {pred.shape} does not "
                f"match RGB {(rgb.height, rgb.width)}"
            )

        panels = [
            _label(rgb, f"RGB | original frame {int(frame_indices[sequence_index])}"),
            _label(
                Image.fromarray(_depth_colour(pred, args.depth_min, args.depth_max)),
                f"Prediction | fixed {args.depth_min:g}-{args.depth_max:g} m",
            ),
        ]

        if gt_files:
            gt = _load_gt(gt_files[int(sequence_index)])
            valid = (
                np.isfinite(gt)
                & (gt > args.depth_min)
                & (gt < args.depth_max)
            )
            error = np.abs(np.clip(pred, args.depth_min, args.depth_max) - gt)
            panels.extend(
                [
                    _label(
                        Image.fromarray(_depth_colour(gt, args.depth_min, args.depth_max)),
                        f"Ground truth | fixed {args.depth_min:g}-{args.depth_max:g} m",
                    ),
                    _label(
                        Image.fromarray(_error_colour(error, args.error_max, valid)),
                        f"Absolute error | capped at {args.error_max:g} m",
                    ),
                ]
            )

        canvas = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "black",
        )
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        canvas.save(frame_output / f"{rendered_index:06d}.png", compress_level=2)

        if (rendered_index + 1) % 10 == 0 or rendered_index + 1 == len(selected):
            print(f"Rendered {rendered_index + 1}/{len(selected)}")

    mp4_path = output / "video_depth_comparison.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not args.no_mp4 and ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_output / "%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]
        subprocess.run(command, check=True)
        print(f"MP4: {mp4_path}")
    elif not args.no_mp4:
        print("ffmpeg was not found; PNG frames were created but MP4 was skipped.")

    print(f"Frames: {frame_output}")


if __name__ == "__main__":
    main()

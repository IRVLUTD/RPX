#!/usr/bin/env python3
"""Run every executable GenCeption RPX task on local smoke-test inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from genception_runtime import load_rgb_video, runtime_from_environment
from PIL import Image


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    return None if xs.size == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _paths(video_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in video_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not paths:
        raise SystemExit(f"no RGB frames found in {video_dir}")
    return paths


def run_variant(args: argparse.Namespace, variant: str) -> dict:
    runtime = runtime_from_environment(variant)
    runtime.setup()
    paths = _paths(args.video_dir)
    video = load_rgb_video(paths[: args.max_frames])
    with Image.open(args.image) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)

    image_depth = runtime.predict_depth(image[None])
    video_depth = runtime.predict_depth(video)
    track_scores = runtime.predict_mask_scores(video, args.track_expression)
    regular_scores = runtime.predict_mask_scores(image[None], args.vqa_expression)
    result = {
        "variant": variant,
        "image_depth": {"shape": list(image_depth.shape), "finite": bool(np.isfinite(image_depth).all())},
        "video_depth": {"shape": list(video_depth.shape), "finite": bool(np.isfinite(video_depth).all())},
        "tracking": {"shape": list(track_scores.shape), "bbox_last": _bbox(track_scores[-1] >= args.threshold)},
        "vqa_regular": {"bbox": _bbox(regular_scores[-1] >= args.threshold)},
    }
    if args.reference_image:
        with Image.open(args.reference_image) as handle:
            reference = np.asarray(handle.convert("RGB").resize((image.shape[1], image.shape[0])))
        two_phase = np.concatenate(
            [np.repeat(reference[None], 40, axis=0), np.repeat(image[None], 41, axis=0)]
        )
        scores = runtime.predict_mask_scores(two_phase, args.in_context_expression)
        result["vqa_in_context"] = {"bbox": _bbox(scores[-1] >= args.threshold)}
    else:
        result["vqa_in_context"] = {"skipped": "--reference-image was not supplied"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("1.3b", "14b", "both"), default="both")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--track-expression", required=True)
    parser.add_argument("--vqa-expression", required=True)
    parser.add_argument(
        "--in-context-expression",
        default="the object in the target phase that matches the object in the reference phase",
    )
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("genception_smoke.json"))
    args = parser.parse_args()
    variants = ("1.3b", "14b") if args.variant == "both" else (args.variant,)
    report = {variant: run_variant(args, variant) for variant in variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

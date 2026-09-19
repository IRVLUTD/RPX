#!/usr/bin/env python3
"""Validate one RPX D3 tracking smoke output without loading a model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

EXPECTED_SHAPE = (480, 640)
EXPECTED_SHAPES = {"mos": EXPECTED_SHAPE, "ego": (1080, 1920)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument(
        "--expected-dataset-protocol",
        choices=("mos", "ego"),
        default="mos",
    )
    parser.add_argument("--require-resume-hit", action="store_true")
    parser.add_argument("--expected-rpx-revision")
    args = parser.parse_args()

    output = Path(args.output_dir)
    result_path = output / "result.json"
    metadata_path = output / "run_metadata.json"
    cells_path = output / "cells.parquet"
    required = (result_path, metadata_path, cells_path, output / "cells.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing smoke artefacts: {missing}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if result.get("model") != args.model:
        raise SystemExit(f"result model is {result.get('model')!r}; expected {args.model!r}")
    if result.get("clips") != args.expected_clips:
        raise SystemExit(f"result has {result.get('clips')} clips; expected {args.expected_clips}")
    if result.get("frames") != args.expected_frames:
        raise SystemExit(
            f"result has {result.get('frames')} frames; expected {args.expected_frames}"
        )
    for name, payload in (("result", result), ("metadata", metadata)):
        if payload.get("dataset_protocol", "mos") != args.expected_dataset_protocol:
            raise SystemExit(
                f"{name} dataset protocol is "
                f"{payload.get('dataset_protocol', 'mos')!r}; expected "
                f"{args.expected_dataset_protocol!r}"
            )
    if args.expected_rpx_revision:
        for name, payload in (("result", result), ("metadata", metadata)):
            if payload.get("rpx_git_sha") != args.expected_rpx_revision:
                raise SystemExit(
                    f"{name} RPX revision is {payload.get('rpx_git_sha')!r}; "
                    f"expected {args.expected_rpx_revision!r}"
                )

    if args.expected_rpx_revision:
        checkpoint_sha = (result.get("model_checkpoint") or {}).get("sha256")
        if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
            raise SystemExit(f"Invalid checkpoint SHA-256: {checkpoint_sha!r}")

    prediction_files = sorted((output / "predictions").glob("*/*/*.npz"))
    markers = sorted((output / "predictions").glob("*/*/_complete.json"))
    bad: list[str] = []
    for path in prediction_files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if archive.files != ["mask"]:
                    bad.append(f"{path}: keys={archive.files}")
                    continue
                mask = archive["mask"]
            expected_shape = EXPECTED_SHAPES[args.expected_dataset_protocol]
            if mask.shape != expected_shape:
                bad.append(f"{path}: shape={mask.shape}")
            elif not np.issubdtype(mask.dtype, np.integer):
                bad.append(f"{path}: dtype={mask.dtype}")
            elif np.any(mask < 0):
                bad.append(f"{path}: negative instance IDs")
        except Exception as exc:  # corruption must fail closed
            bad.append(f"{path}: {exc!r}")

    for path in markers:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("model") != args.model:
            bad.append(f"{path}: marker model={marker.get('model')!r}")
        if marker.get("frames") != args.expected_frames:
            bad.append(f"{path}: marker frames={marker.get('frames')!r}")
        if marker.get("dataset_protocol", "mos") != args.expected_dataset_protocol:
            bad.append(f"{path}: marker dataset protocol={marker.get('dataset_protocol', 'mos')!r}")
        if args.expected_rpx_revision and marker.get("rpx_git_sha") != args.expected_rpx_revision:
            bad.append(f"{path}: marker RPX revision={marker.get('rpx_git_sha')!r}")

    if len(prediction_files) != args.expected_frames:
        bad.append(f"prediction files={len(prediction_files)}; expected={args.expected_frames}")
    if len(markers) != args.expected_clips:
        bad.append(f"complete markers={len(markers)}; expected={args.expected_clips}")

    stats = metadata.get("prediction_stats") or {}
    if args.require_resume_hit:
        if stats.get("complete_clip_cache_hits") != args.expected_clips:
            bad.append(
                "resume cache hits="
                f"{stats.get('complete_clip_cache_hits')}; expected={args.expected_clips}"
            )
        if stats.get("model_propagation_frames") != 0:
            bad.append(f"resume model forwards={stats.get('model_propagation_frames')}; expected=0")

    print(f"Model: {args.model}")
    print(f"Prediction files: {len(prediction_files)}")
    print(f"Complete clips: {len(markers)}")
    print(f"Invalid entries: {len(bad)}")
    print(f"Prediction stats: {stats}")
    for item in bad[:20]:
        print("BAD:", item)
    if bad:
        raise SystemExit(1)
    print("RPX tracking infrastructure validation: PASS")


if __name__ == "__main__":
    main()

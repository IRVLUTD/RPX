#!/usr/bin/env python3
"""Run PaliGemma 2 on the real RPX VQA smoke fixture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from vqa_models.paligemma2 import CHECKPOINTS, PaliGemma2Runner

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_rgb
from rpx_benchmark.vqa.prompts import build_prompt
from rpx_benchmark.vqa.roster import get_model


def synchronize_cuda() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="paligemma2-3b")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/vqa_smoke/v1/manifest.jsonl")
    )
    parser.add_argument("--image-cache", type=Path, default=Path("/cache/rpx-vqa/images"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    get_model(args.model)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run the container with --gpus all")

    samples = load_manifest(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    image_paths = {sample.sample_id: fetch_rgb(sample, args.image_cache) for sample in samples}
    runner = PaliGemma2Runner(args.model)
    for _ in range(args.warmup):
        sample = samples[0]
        prompt = build_prompt(sample, args.model)
        runner.predict(
            image_paths[sample.sample_id],
            prompt.text,
            prompt.max_new_tokens,
            prompt.output_kind,
        )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            prompt = build_prompt(sample, args.model)
            synchronize_cuda()
            started = time.perf_counter()
            raw_output = runner.predict(
                image_paths[sample.sample_id],
                prompt.text,
                prompt.max_new_tokens,
                prompt.output_kind,
            )
            synchronize_cuda()
            elapsed_ms = (time.perf_counter() - started) * 1000
            row = {
                "sample_id": sample.sample_id,
                "raw_output": raw_output,
                "latency_ms": elapsed_ms,
                "model": args.model,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(samples)}] {sample.sample_id} {raw_output!r}", flush=True)


if __name__ == "__main__":
    main()

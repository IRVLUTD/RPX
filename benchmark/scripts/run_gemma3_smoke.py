#!/usr/bin/env python3
"""Run Gemma 3 bbox inference and save synchronized per-sample latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from vqa_models.gemma3 import CHECKPOINTS, Gemma3Runner

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_rgb
from rpx_benchmark.vqa.prompts import build_prompt


def synchronize_cuda() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="gemma3-12b")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=Path("/cache/rpx-vqa/images"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run the container with --gpus all")
    samples = load_manifest(args.manifest)
    if args.limit is not None:
        samples = samples[:args.limit]
    image_paths = {s.sample_id: fetch_rgb(s, args.image_cache) for s in samples}
    runner = Gemma3Runner(args.model)
    for _ in range(args.warmup):
        sample = samples[0]
        spec = build_prompt(sample, args.model)
        runner.predict(image_paths[sample.sample_id], spec.text, spec.max_new_tokens)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            spec = build_prompt(sample, args.model)
            synchronize_cuda()
            started = time.perf_counter()
            raw = runner.predict(image_paths[sample.sample_id], spec.text, spec.max_new_tokens)
            synchronize_cuda()
            latency_ms = (time.perf_counter() - started) * 1000
            row = {"sample_id": sample.sample_id, "raw_output": raw, "latency_ms": latency_ms, "model": args.model}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(samples)}] {sample.sample_id} {latency_ms:.1f} ms {raw!r}", flush=True)


if __name__ == "__main__":
    main()

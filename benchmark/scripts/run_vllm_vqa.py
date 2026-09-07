#!/usr/bin/env python3
"""Run any RPX VQA model exclusively through vLLM offline inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import vllm
from vqa_models.vllm_backend import CHECKPOINTS, VLLMVQARunner

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_rgb
from rpx_benchmark.vqa.prompts import build_prompt
from rpx_benchmark.vqa.roster import get_model


def synchronize_cuda() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="append after validated completed rows"
    )
    args = parser.parse_args()
    get_model(args.model)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; expose exactly one GPU to this container")
    if torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible GPU is required per vLLM VQA engine")

    samples = load_manifest(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    image_paths = {sample.sample_id: fetch_rgb(sample, args.image_cache) for sample in samples}
    runner = VLLMVQARunner(
        args.model,
        args.image_cache,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    for _ in range(args.warmup):
        sample = samples[0]
        spec = build_prompt(sample, args.model)
        runner.predict(
            image_paths[sample.sample_id], spec.text, spec.max_new_tokens, spec.output_kind
        )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and args.predictions.is_file():
        expected_checkpoint = CHECKPOINTS[args.model]
        with args.predictions.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                row = json.loads(line)
                if (
                    row.get("backend") != "vllm"
                    or row.get("model") != args.model
                    or row.get("checkpoint") != expected_checkpoint.repo_id
                    or row.get("revision") != expected_checkpoint.revision
                ):
                    raise SystemExit(
                        f"cannot resume incompatible prediction at line {line_number}"
                    )
                sample_id = str(row["sample_id"])
                if sample_id in completed:
                    raise SystemExit(f"duplicate resumed prediction: {sample_id}")
                completed.add(sample_id)
        unknown = completed - {sample.sample_id for sample in samples}
        if unknown:
            raise SystemExit(f"cannot resume predictions outside this manifest: {sorted(unknown)}")

    mode = "a" if completed else "w"
    checkpoint = CHECKPOINTS[args.model]
    with args.predictions.open(mode, encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            if sample.sample_id in completed:
                print(f"[{index}/{len(samples)}] {sample.sample_id} already complete", flush=True)
                continue
            spec = build_prompt(sample, args.model)
            synchronize_cuda()
            started = time.perf_counter()
            raw = runner.predict(
                image_paths[sample.sample_id],
                spec.text,
                spec.max_new_tokens,
                spec.output_kind,
            )
            synchronize_cuda()
            latency_ms = (time.perf_counter() - started) * 1000
            row = {
                "sample_id": sample.sample_id,
                "raw_output": raw,
                "latency_ms": latency_ms,
                "model": args.model,
                "backend": "vllm",
                "vllm_version": vllm.__version__,
                "checkpoint": checkpoint.repo_id,
                "revision": checkpoint.revision,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(samples)}] {sample.sample_id} {latency_ms:.1f} ms {raw!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()

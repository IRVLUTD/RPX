#!/usr/bin/env python3
"""Run an RPX VQA model through its frozen inference backend."""

from __future__ import annotations

import argparse
import importlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from vqa_models.backend_registry import CHECKPOINTS, create_runner, provenance

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images_many
from rpx_benchmark.vqa.prompts import build_prompt
from rpx_benchmark.vqa.roster import get_model

# Import torch only after the selected runtime has installed CUDA compatibility paths.
torch = importlib.import_module("torch")


def synchronize_cuda() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def validate_gpu_environment(server_url: str | None) -> None:
    """Require one GPU only when this process constructs the model engine.

    A remote client performs preprocessing, HTTP calls, and scoring only. It
    may run inside a resident tensor-parallel container that intentionally
    exposes multiple GPUs (DeepSeek-VL2 full uses two), so applying the local
    engine invariant to that client incorrectly rejects a healthy server.
    """
    if server_url:
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; expose exactly one GPU to this container")
    if torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible GPU is required per VQA engine")


class RemoteRunner:
    """Thin client for a model kept resident by serve_vllm_vqa.py."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")
        self._metadata: dict = {}

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.server_url}/health", timeout=30) as response:
            return json.load(response)

    def predict(self, image_paths, prompt: str, max_tokens: int, output_kind: str) -> str:
        payload = json.dumps(
            {
                "image_paths": [str(path) for path in image_paths],
                "prompt": prompt,
                "max_tokens": max_tokens,
                "output_kind": output_kind,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.server_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {error.code} from {self.server_url}/predict: {body}"
            ) from error
        self._metadata = result.get("adapter_metadata") or {}
        self.latency_ms = float(result["latency_ms"])
        return str(result["raw_output"])

    def prediction_metadata(self) -> dict:
        return dict(self._metadata)


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
        "--server-url",
        help="reuse a resident VQA engine instead of loading another model",
    )
    parser.add_argument(
        "--resume", action="store_true", help="append after validated completed rows"
    )
    args = parser.parse_args()
    model = get_model(args.model)
    if "bbox" not in model.capabilities:
        raise SystemExit(
            f"{model.key} does not support the scored direct one-stage bbox protocol"
        )
    validate_gpu_environment(args.server_url)

    samples = load_manifest(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    # fetch_images independently fetches+verifies every locator a sample
    # needs (target only for normal rows; reference then target, in that
    # order, for in-context rows -- reference_crop_sha256 is verified as
    # part of this call, raising DownloadError on any mismatch).
    image_paths = fetch_images_many(samples, args.image_cache)
    if args.server_url:
        runner = RemoteRunner(args.server_url)
        health = runner.health()
        checkpoint = CHECKPOINTS[args.model]
        if (
            health.get("model") != args.model
            or health.get("checkpoint") != checkpoint.repo_id
            or health.get("revision") != checkpoint.revision
        ):
            raise SystemExit(f"resident engine provenance mismatch: {health}")
    else:
        runner = create_runner(
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
        expected = {"model": args.model, **provenance(args.model)}
        with args.predictions.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                row = json.loads(line)
                if (
                    any(row.get(key) != value for key, value in expected.items())
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
    runtime = provenance(args.model)
    with args.predictions.open(mode, encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            if sample.sample_id in completed:
                print(f"[{index}/{len(samples)}] {sample.sample_id} already complete", flush=True)
                continue
            spec = build_prompt(sample, args.model)
            if not args.server_url:
                synchronize_cuda()
            started = time.perf_counter()
            raw = runner.predict(
                image_paths[sample.sample_id],
                spec.text,
                spec.max_new_tokens,
                spec.output_kind,
            )
            if not args.server_url:
                synchronize_cuda()
            latency_ms = (
                runner.latency_ms
                if args.server_url
                else (time.perf_counter() - started) * 1000
            )
            row = {
                "sample_id": sample.sample_id,
                "raw_output": raw,
                "latency_ms": latency_ms,
                "model": args.model,
                **runtime,
                "checkpoint": checkpoint.repo_id,
                "revision": checkpoint.revision,
                "in_context": sample.is_in_context,
                "num_images": len(image_paths[sample.sample_id]),
                "prompt_text": spec.text,
                "output_kind": spec.output_kind,
                "adapter_metadata": runner.prediction_metadata(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(samples)}] {sample.sample_id} {latency_ms:.1f} ms {raw!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()

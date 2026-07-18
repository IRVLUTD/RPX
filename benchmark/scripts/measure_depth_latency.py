#!/usr/bin/env python3
"""Measure DA-V2 Large latency on a fixed RPX subset without prediction writes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from run_depth import _build_model  # noqa: E402

from rpx_benchmark.loader import RPXDataset  # noqa: E402

MODEL_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1:
        parser.error("--warmup and --samples must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Latency protocol requires CUDA.")
    dataset = RPXDataset.from_manifest(
        args.manifest_path,
        batch_size=1,
        max_samples=args.warmup + args.samples,
    )
    frames = [sample for batch in dataset for sample in batch]
    if len(frames) != args.warmup + args.samples:
        raise SystemExit(
            f"Latency subset has {len(frames)} frames; expected {args.warmup + args.samples}."
        )

    _model, adapter = _build_model(
        "da-v2-large",
        device="cuda",
        batch_size=1,
        precision="fp16",
    )
    if getattr(adapter, "model_id", None) != MODEL_CHECKPOINT:
        raise SystemExit("Latency protocol loaded the wrong checkpoint.")
    if getattr(adapter, "actual_torch_dtype", None) != "torch.float16":
        raise SystemExit(
            f"Latency protocol expected torch.float16; got {adapter.actual_torch_dtype}."
        )

    for sample in frames[: args.warmup]:
        adapter([np.asarray(sample.rgb, dtype=np.uint8)])
        torch.cuda.synchronize()

    timings_ms: list[float] = []
    for sample in frames[args.warmup :]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        adapter([np.asarray(sample.rgb, dtype=np.uint8)])
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(timings_ms, dtype=np.float64)
    payload = {
        "schema_version": "rpx-d1f-latency-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model": "da-v2-large",
        "model_checkpoint": MODEL_CHECKPOINT,
        "actual_torch_dtype": "torch.float16",
        "device": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "subset": "first 103 manifest-ordered Easy frames (3 warm-up, 100 measured)",
        "warmup_forwards": args.warmup,
        "measured_forwards": args.samples,
        "prediction_writes": False,
        "cache_hit_timing": False,
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "git_sha": os.environ.get("RPX_GIT_SHA", "unknown"),
        "docker_digest": os.environ.get("RPX_DOCKER_DIGEST", "unknown"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".part")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, args.output)
    print(f"latency: {args.output} (median={payload['median_ms']:.3f} ms)")


if __name__ == "__main__":
    main()

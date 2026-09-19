#!/usr/bin/env python3
"""Validate and combine RPX depth hardware-profile JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

FRAME_MODELS = (
    "da-v2-large",
    "depth-pro",
    "unidepth-v2",
    "metric3d-v2",
    "moge-2-vit-l",
    "da3-metric-l",
    "hyden",
    "lotus-2",
    "zipdepth",
    "fe2e",
)
VIDEO_MODELS = (
    "video-da",
    "da3-video",
    "vggt-omega",
    "vigeo",
    "monst3r",
    "depth-crafter",
    "gem-depth",
    "dvd",
    "chrono-depth",
    "rolling-depth",
)
HEADLINE = (
    "params_total",
    "flops_per_sample",
    "latency_p50_ms_per_sample",
)


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument(
        "--allow-incomplete-flops",
        action="store_true",
        help="Keep rows whose FLOP counter explicitly reports unsupported/failed.",
    )
    args = parser.parse_args()

    root = Path(args.input_root)
    expected = {**{m: "image" for m in FRAME_MODELS}, **{m: "video" for m in VIDEO_MODELS}}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for model, task in expected.items():
        path = root / task / f"{model}.json"
        if not path.is_file():
            errors.append(f"{model}: missing {path}")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("model") != model or row.get("task") != task:
            errors.append(f"{model}: identity mismatch in {path}")
            continue
        for key in ("params_total", "latency_p50_ms_per_sample"):
            if not _finite_positive(row.get(key)):
                errors.append(f"{model}: invalid {key}={row.get(key)!r}")
        if not _finite_positive(row.get("flops_per_sample")):
            message = (
                f"{model}: FLOPs unavailable "
                f"(status={row.get('flop_status', 'missing')})"
            )
            if not args.allow_incomplete_flops:
                errors.append(message)
            row["flops_note"] = message
        rows.append(row)

    fields = [
        "task", "model", "input_protocol", "frames_per_sample", "precision",
        "parameter_dtypes", "params_total", "params_total_m",
        "parameter_storage_mb", "flops_per_sample", "flops_per_sample_g",
        "flops_per_frame_g", "macs_per_sample_g", "flop_status",
        "latency_p50_ms_per_sample", "latency_p95_ms_per_sample",
        "latency_p99_ms_per_sample", "latency_mean_ms_per_sample",
        "latency_p50_ms_per_frame", "throughput_frames_per_second",
        "peak_cuda_allocated_mb", "peak_cuda_reserved_mb", "peak_cpu_rss_mb",
        "gpu_name", "gpu_total_memory_mb", "torch_version", "cuda_runtime",
        "batch_size", "distinct_samples", "warmup_forwards",
        "repetitions_per_sample", "timed_forwards",
        "dataset_revision", "code_sha", "container_image",
        "concurrent_host_load", "profiled_at_utc", "flops_note",
    ]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["task"], row["model"])))

    validation = {
        "expected_models": len(expected),
        "profiles_found": len(rows),
        "headline_fields": list(HEADLINE),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    validation_path = Path(args.validation_json)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    print(f"CSV: {output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

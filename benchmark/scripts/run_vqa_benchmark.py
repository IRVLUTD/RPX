#!/usr/bin/env python3
"""Run one shard of the full RPX VQA benchmark plan.

Supports configurable batching for real throughput (default batch size 8;
the acceptance/smoke gates use run_vllm_vqa.py's batch size 1 for isolated
per-request latency instead -- this script's batched timing is always
recorded as amortized_latency_ms / batch_throughput_qps, never as the
isolated-request latency_ms field the gates use). Resumable, rejects
duplicate or missing sample IDs on resume, retains every raw output, and
never crashes the whole shard on one bad row -- a failing batch is retried
sample-by-sample and any row that still fails is recorded in failures.jsonl
instead of predictions.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
import traceback
from pathlib import Path

from run_vllm_vqa import RemoteRunner
from vqa_models.backend_registry import CHECKPOINTS, create_runner, provenance

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images_many
from rpx_benchmark.vqa.metrics import score_predictions
from rpx_benchmark.vqa.outputs import parse_output
from rpx_benchmark.vqa.prompts import build_prompt
from rpx_benchmark.vqa.roster import get_model

torch = importlib.import_module("torch")


def synchronize_cuda() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def shard_of(samples: list, shard_index: int, shard_count: int) -> list:
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    return [sample for i, sample in enumerate(ordered) if i % shard_count == shard_index]


def load_existing(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise SystemExit(f"duplicate sample_id in {path} at line {line_number}: {sample_id}")
            rows[sample_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-config-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--server-url",
        help="reuse a resident serve_vllm_vqa.py engine (safe, sequential requests)",
    )
    parser.add_argument(
        "--max-row-attempts",
        type=int,
        default=3,
        help="retry an individual inference before recording a terminal failure",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="on resume, archive and retry rows previously recorded as failed",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"shard-index {args.shard_index} out of range for shard-count {args.shard_count}")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be >= 1")
    if args.max_row_attempts < 1:
        raise SystemExit("max-row-attempts must be >= 1")

    model = get_model(args.model)
    if not args.server_url:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; expose exactly one GPU to this container")
        if torch.cuda.device_count() != 1:
            raise SystemExit("exactly one visible GPU is required per VQA engine")

    all_samples = load_manifest(args.manifest)
    shard = shard_of(all_samples, args.shard_index, args.shard_count)
    shard_ids = {sample.sample_id for sample in shard}
    if len(shard_ids) != len(shard):
        raise SystemExit("duplicate sample_id within this shard (should be impossible)")

    checkpoint = CHECKPOINTS[args.model]
    runtime = provenance(args.model)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    if args.resume:
        completed = load_existing(args.predictions)
        failed = load_existing(args.failures)
        overlap = set(completed) & set(failed)
        if overlap:
            raise SystemExit(f"sample_id present in both predictions and failures: {sorted(overlap)}")
        unknown = (set(completed) | set(failed)) - shard_ids
        if unknown:
            raise SystemExit(f"cannot resume: IDs outside this shard: {sorted(unknown)[:10]}")
        expected_provenance = runtime
        for source_rows in (completed, failed):
            for sample_id, row in source_rows.items():
                if row.get("model") != args.model or any(
                    row.get(key) != value for key, value in expected_provenance.items()
                ):
                    raise SystemExit(f"cannot resume incompatible prediction/failure: {sample_id}")
        if args.retry_failures and failed:
            history_path = args.failures.with_suffix(args.failures.suffix + ".history")
            with history_path.open("a", encoding="utf-8") as history:
                for row in failed.values():
                    history.write(
                        json.dumps(
                            {**row, "retry_queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                            sort_keys=True,
                        )
                        + "\n"
                    )
            failed = {}

    remaining = [sample for sample in shard if sample.sample_id not in completed and sample.sample_id not in failed]
    print(
        f"shard {args.shard_index}/{args.shard_count}: {len(shard)} rows total, "
        f"{len(completed)} already predicted, {len(failed)} already failed, {len(remaining)} to run",
        flush=True,
    )

    image_paths = fetch_images_many(remaining, args.image_cache)
    if args.server_url:
        runner = RemoteRunner(args.server_url)
        health = runner.health()
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
            max_num_seqs=max(1, args.batch_size),
        )

    run_config = {
        "model": args.model,
        "checkpoint": checkpoint.repo_id,
        "checkpoint_revision": checkpoint.revision,
        **runtime,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "manifest_rows": len(all_samples),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_size": len(shard),
        "batch_size": args.batch_size,
        "sampling": {
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 3 if args.model.startswith("florence2-") else 1,
        },
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "server_url": args.server_url,
        "resident_engine": bool(args.server_url),
        "max_row_attempts": args.max_row_attempts,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.run_config_out.parent.mkdir(parents=True, exist_ok=True)
    args.run_config_out.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    pred_handle = args.predictions.open("a" if completed else "w", encoding="utf-8")
    fail_handle = args.failures.open("a" if failed else "w", encoding="utf-8")

    def _predict_one(sample, spec):
        last_error: Exception | None = None
        for attempt in range(1, args.max_row_attempts + 1):
            try:
                started = time.perf_counter()
                raw = runner.predict(
                    image_paths[sample.sample_id],
                    spec.text,
                    spec.max_new_tokens,
                    spec.output_kind,
                )
                if not args.server_url:
                    synchronize_cuda()
                elapsed_ms = (
                    runner.latency_ms
                    if args.server_url
                    else (time.perf_counter() - started) * 1000
                )
                return raw, elapsed_ms
            except Exception as error:  # noqa: BLE001
                last_error = error
                print(
                    f"{sample.sample_id}: attempt {attempt}/{args.max_row_attempts} failed: {error}",
                    flush=True,
                )
                if attempt < args.max_row_attempts:
                    time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error

    def _record_success(
        sample,
        raw: str,
        amortized_ms: float,
        batch_size: int,
        throughput_qps: float,
        adapter_metadata: dict | None = None,
    ) -> None:
        spec = build_prompt(sample, args.model)
        row = {
            "sample_id": sample.sample_id,
            "raw_output": raw,
            "amortized_latency_ms": amortized_ms,
            "batch_size": batch_size,
            "batch_throughput_qps": throughput_qps,
            "model": args.model,
            **runtime,
            "checkpoint": checkpoint.repo_id,
            "revision": checkpoint.revision,
            "in_context": sample.is_in_context,
            "num_images": len(image_paths[sample.sample_id]),
            "prompt_text": spec.text,
            "output_kind": spec.output_kind,
            "adapter_metadata": adapter_metadata or {},
        }
        pred_handle.write(json.dumps(row, sort_keys=True) + "\n")
        pred_handle.flush()
        completed[sample.sample_id] = row

    def _record_failure(sample, error: Exception) -> None:
        row = {
            "sample_id": sample.sample_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "model": args.model,
            **runtime,
            "checkpoint": checkpoint.repo_id,
            "revision": checkpoint.revision,
        }
        fail_handle.write(json.dumps(row, sort_keys=True) + "\n")
        fail_handle.flush()
        failed[sample.sample_id] = row

    processed = 0
    for start in range(0, len(remaining), args.batch_size):
        chunk = remaining[start : start + args.batch_size]
        specs = {sample.sample_id: build_prompt(sample, args.model) for sample in chunk}
        requests = [
            (
                image_paths[sample.sample_id],
                specs[sample.sample_id].text,
                specs[sample.sample_id].max_new_tokens,
                specs[sample.sample_id].output_kind,
            )
            for sample in chunk
        ]
        if args.server_url:
            # A resident engine was started with max_num_seqs=1.  Preserve
            # correctness and real server-side latency by issuing one request
            # at a time; the four models still run concurrently on four GPUs.
            for sample in chunk:
                spec = specs[sample.sample_id]
                try:
                    raw, row_ms = _predict_one(sample, spec)
                    _record_success(
                        sample,
                        raw,
                        row_ms,
                        1,
                        1000 / row_ms if row_ms > 0 else float("inf"),
                        runner.prediction_metadata(),
                    )
                except Exception as row_error:  # noqa: BLE001
                    _record_failure(sample, row_error)
            processed += len(chunk)
            print(f"[{processed}/{len(remaining)}] shard progress", flush=True)
            continue

        synchronize_cuda()
        started = time.perf_counter()
        try:
            outputs = runner.predict_batch(requests)
            adapter_metadata = runner.batch_prediction_metadata()
            if len(adapter_metadata) != len(outputs):
                raise RuntimeError("adapter metadata coverage mismatch")
            synchronize_cuda()
            wall_ms = (time.perf_counter() - started) * 1000
            amortized = wall_ms / len(chunk)
            throughput = len(chunk) / (wall_ms / 1000) if wall_ms > 0 else float("inf")
            for sample, raw, metadata in zip(
                chunk, outputs, adapter_metadata, strict=True
            ):
                _record_success(
                    sample, raw, amortized, len(chunk), throughput, metadata
                )
        except Exception:  # noqa: BLE001 -- isolate the batch, then retry per-row
            for sample in chunk:
                spec = specs[sample.sample_id]
                synchronize_cuda()
                try:
                    raw, row_ms = _predict_one(sample, spec)
                    _record_success(
                        sample,
                        raw,
                        row_ms,
                        1,
                        1000 / row_ms if row_ms > 0 else float("inf"),
                        runner.prediction_metadata(),
                    )
                except Exception as row_error:  # noqa: BLE001
                    _record_failure(sample, row_error)
        processed += len(chunk)
        print(f"[{processed}/{len(remaining)}] shard progress", flush=True)

    pred_handle.close()
    fail_handle.close()

    covered = set(completed) | set(failed)
    missing = shard_ids - covered
    extra = covered - shard_ids
    if missing or extra:
        raise SystemExit(f"shard coverage mismatch after run: missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}")

    scored_samples = {sample.sample_id: sample for sample in shard if sample.sample_id in completed}
    parsed_by_id = {
        sample_id: parse_output(scored_samples[sample_id], row["raw_output"], args.model)
        for sample_id, row in completed.items()
    }
    metrics = score_predictions(
        (scored_samples[sample_id], parsed_by_id[sample_id]) for sample_id in completed
    )
    metrics_by_context = {}
    for context_name, in_context in (("regular", False), ("in_context", True)):
        subset_ids = [
            sample_id
            for sample_id, sample in scored_samples.items()
            if sample.is_in_context is in_context
        ]
        if subset_ids:
            metrics_by_context[context_name] = score_predictions(
                (scored_samples[sample_id], parsed_by_id[sample_id])
                for sample_id in subset_ids
            )
    amortized_values = [row["amortized_latency_ms"] for row in completed.values()]
    report = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_size": len(shard),
        "succeeded": len(completed),
        "failed": len(failed),
        "batch_size": args.batch_size,
        "metrics": metrics,
        "metrics_by_context": metrics_by_context,
        "amortized_latency_ms": {
            "mean": sum(amortized_values) / len(amortized_values) if amortized_values else None,
            "min": min(amortized_values) if amortized_values else None,
            "max": max(amortized_values) if amortized_values else None,
        },
        "run_config": run_config,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

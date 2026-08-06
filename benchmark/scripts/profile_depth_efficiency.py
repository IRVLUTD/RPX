#!/usr/bin/env python3
"""Profile one RPX image- or video-depth model without rerunning accuracy."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

Forward = Callable[[], Any]


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _sum_flops(node: Any) -> int:
    if isinstance(node, dict):
        return sum(_sum_flops(value) for value in node.values())
    try:
        return int(node)
    except (TypeError, ValueError):
        return 0


def _torch_profiler_flops(
    forward: Forward, torch: Any
) -> int:
    """Count supported ATen operations without retaining an autograd graph.

    ``FlopCounterMode`` is preferred because it gives the most direct
    operator-level accounting.  VGGT's full 250-frame forward, however,
    requires substantially more memory when its autograd graph is retained
    for that mode.  The PyTorch profiler can count the same supported ATen
    operations during inference mode, which keeps the measurement feasible
    on a 48 GiB GPU.  This is intentionally used only as a documented
    fallback and its method is recorded in the output provenance.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode(), torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
    ) as profiler:
        output = forward()
        _sync(torch)
        del output

    return sum(int(event.flops or 0) for event in profiler.key_averages())


def _candidate_modules(root: Any, torch: Any) -> Iterable[Any]:
    module_type = torch.nn.Module
    seen_objects: set[int] = set()
    seen_modules: set[int] = set()
    queue: list[tuple[Any, int]] = [(root, 0)]
    names = (
        "model",
        "_model",
        "torch_module",
        "_pipe",
        "pipe",
        "pipeline",
        "unet",
        "vae",
        "transformer",
        "text_encoder",
        "image_encoder",
        "depth_model",
        "core",
        "net",
    )

    while queue:
        obj, depth = queue.pop(0)
        if obj is None or id(obj) in seen_objects or depth > 4:
            continue
        seen_objects.add(id(obj))

        if isinstance(obj, module_type):
            if id(obj) not in seen_modules:
                seen_modules.add(id(obj))
                yield obj
            continue

        components = getattr(obj, "components", None)
        if isinstance(components, dict):
            queue.extend((value, depth + 1) for value in components.values())

        for name in names:
            with contextlib.suppress(Exception):
                value = getattr(obj, name)
                if value is not obj:
                    queue.append((value, depth + 1))


def _parameter_information(
    root: Any, torch: Any
) -> tuple[int | None, int | None, int | None, str | None]:
    unique: dict[int, Any] = {}
    for module in _candidate_modules(root, torch):
        for parameter in module.parameters():
            unique.setdefault(id(parameter), parameter)
    if not unique:
        return None, None, None, None

    total = sum(parameter.numel() for parameter in unique.values())
    trainable = sum(
        parameter.numel()
        for parameter in unique.values()
        if parameter.requires_grad
    )
    storage = sum(
        parameter.numel() * parameter.element_size()
        for parameter in unique.values()
    )
    dtypes = ",".join(
        sorted({str(parameter.dtype) for parameter in unique.values()})
    )
    return int(total), int(trainable), int(storage), dtypes


def _load_image_inputs(
    args: argparse.Namespace,
) -> tuple[Any, list[tuple[str, Forward]], int, str]:
    from rpx_benchmark.loader import RPXDataset
    from run_depth import _build_model

    _placeholder, adapter = _build_model(
        args.model,
        args.device,
        batch_size=1,
        acknowledge_unverified=args.acknowledge_unverified,
        precision=args.precision,
    )
    dataset = RPXDataset.from_manifest(
        args.manifest_path,
        batch_size=1,
        max_samples=args.samples,
    )
    samples = [batch[0] for batch in dataset]
    if len(samples) != args.samples:
        raise RuntimeError(
            f"Requested {args.samples} image samples, loaded {len(samples)}"
        )

    forwards: list[tuple[str, Forward]] = []
    shapes: set[tuple[int, ...]] = set()
    for index, sample in enumerate(samples):
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        shapes.add(tuple(rgb.shape))
        sample_id = str(getattr(sample, "id", f"manifest_index_{index}"))

        def forward(rgb: np.ndarray = rgb) -> Any:
            result = adapter([rgb])
            if isinstance(result, (list, tuple)):
                result = result[0]
            output = np.asarray(result)
            if output.ndim != 2 or not np.isfinite(output).all():
                raise RuntimeError(
                    f"{args.model} returned invalid depth shape={output.shape}"
                )
            return result

        forwards.append((sample_id, forward))

    if len(shapes) != 1:
        raise RuntimeError(f"Image inputs have mixed shapes: {shapes}")
    shape = next(iter(shapes))
    return adapter, forwards, 1, f"1xRGB:{shape[1]}x{shape[0]}"


def _load_video_inputs(
    args: argparse.Namespace,
) -> tuple[Any, list[tuple[str, Forward]], int, str]:
    from rpx_benchmark.video_loader import VideoDepthDataset
    from run_video_depth import _load_model

    model = _load_model(
        args.model,
        args.device,
        acknowledge_unverified=args.acknowledge_unverified,
    )
    dataset = VideoDepthDataset.from_manifest(
        args.manifest_path,
        batch_size=1,
        frame_budget=args.frame_budget,
        sampling="all" if args.frame_budget is None else "stride",
        max_samples=args.samples,
        compute_fscore=False,
    )
    samples = [batch[0] for batch in dataset]
    if len(samples) != args.samples:
        raise RuntimeError(
            f"Requested {args.samples} video samples, loaded {len(samples)}"
        )
    model.setup()

    forwards: list[tuple[str, Forward]] = []
    shapes: set[tuple[int, ...]] = set()
    for index, sample in enumerate(samples):
        rgb = np.asarray(sample.rgb_seq)
        shapes.add(tuple(rgb.shape))
        expected = tuple(rgb.shape[:3])
        sample_id = str(getattr(sample, "id", f"manifest_index_{index}"))

        def forward(sample: Any = sample, expected: tuple[int, ...] = expected) -> Any:
            result = model.predict([sample])
            if len(result) != 1:
                raise RuntimeError(
                    f"{args.model} returned {len(result)} video predictions"
                )
            depth = np.asarray(result[0].depth_map_seq)
            if depth.shape != expected:
                raise RuntimeError(
                    f"{args.model} returned {depth.shape}; expected {expected}"
                )
            if not np.isfinite(depth).all():
                raise RuntimeError(f"{args.model} returned non-finite depth")
            return result

        forwards.append((sample_id, forward))

    if len(shapes) != 1:
        raise RuntimeError(f"Video inputs have mixed shapes: {shapes}")
    shape = next(iter(shapes))
    frames = int(shape[0])
    return model, forwards, frames, f"1xclip:{frames}x{shape[2]}x{shape[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("image", "video"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frame-budget", type=int)
    parser.add_argument("--allow-noncanonical-input", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--acknowledge-unverified", action="store_true")
    args = parser.parse_args()

    if args.samples < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("--samples, --warmup and --repeats must all be >=1")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    loader = _load_image_inputs if args.task == "image" else _load_video_inputs
    print(f"[profile] loading {args.task} model {args.model}", flush=True)
    model, forwards, frames_per_sample, input_protocol = loader(args)
    canonical_forward = forwards[0][1]

    if not args.allow_noncanonical_input:
        if args.task == "image" and input_protocol != "1xRGB:640x480":
            raise RuntimeError(
                f"Paper protocol requires 640x480; got {input_protocol}"
            )
        if args.task == "video" and frames_per_sample != 250:
            raise RuntimeError(
                "Paper protocol requires a full 250-frame clip; "
                f"got {frames_per_sample}"
            )

    total_params, trainable_params, parameter_bytes, dtypes = (
        _parameter_information(model, torch)
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(
        f"[profile] warm-up={args.warmup}; distinct_samples={len(forwards)}; "
        f"repeats={args.repeats}",
        flush=True,
    )
    latency_records: list[dict[str, Any]] = []
    elapsed_ms: list[float] = []
    # RollingDepth performs test-time depth alignment with an optimizer, so
    # disabling autograd breaks its normal inference path. Other adapters use
    # inference_mode to avoid measuring avoidable autograd overhead.
    inference_context = (
        contextlib.nullcontext
        if args.model == "rolling-depth"
        else torch.inference_mode
    )

    with inference_context():
        for index in range(args.warmup):
            output = canonical_forward()
            _sync(torch)
            del output
            print(f"[profile] warm-up {index + 1}/{args.warmup}", flush=True)

        total_forwards = len(forwards) * args.repeats
        timed_index = 0
        for repetition in range(args.repeats):
            for sample_index, (sample_id, forward) in enumerate(forwards):
                _sync(torch)
                started = time.perf_counter()
                output = forward()
                _sync(torch)
                latency_ms = (time.perf_counter() - started) * 1000.0
                del output
                elapsed_ms.append(latency_ms)
                latency_records.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": sample_id,
                        "repetition": repetition,
                        "latency_ms": latency_ms,
                    }
                )
                timed_index += 1
                print(
                    f"[profile] timed {timed_index}/{total_forwards} "
                    f"sample={sample_id}: {latency_ms:.3f} ms",
                    flush=True,
                )

    peak_allocated_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else None
    )
    peak_reserved_mb = (
        float(torch.cuda.max_memory_reserved() / (1024**2))
        if torch.cuda.is_available()
        else None
    )
    peak_cpu_rss_mb = float(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )

    flops: int | None = None
    flop_status = "skipped" if args.skip_flops else "unavailable"
    flop_method: str | None = None
    # A full VGGT clip fits for normal inference, but not when
    # FlopCounterMode retains the complete autograd graph.  Do not attempt
    # that counter first: after an OOM its failed graph remains allocated
    # until this process exits, leaving no memory for a fallback.  Profile
    # VGGT directly in inference mode instead.
    if not args.skip_flops and args.model == "vggt-omega":
        print(
            "[profile] VGGT FLOPs: inference-mode torch.profiler",
            flush=True,
        )
        try:
            counted = _torch_profiler_flops(canonical_forward, torch)
            if counted > 0:
                flops = counted
                flop_status = "measured:torch_profiler"
                flop_method = "torch.profiler.key_averages(with_flops=True)"
            else:
                flop_status = "unsupported_or_zero:torch_profiler"
                flop_method = "torch.profiler.key_averages(with_flops=True)"
        except Exception as exc:  # noqa: BLE001
            flop_status = f"failed:{type(exc).__name__}:{exc}"
    elif not args.skip_flops:
        print("[profile] FLOPs: one canonical instrumented forward", flush=True)
        try:
            from torch.utils.flop_counter import FlopCounterMode

            # FLOP counting is an instrumented forward, not a latency
            # measurement.  Keep the normal inference context above intact;
            # a few official implementations need a different context here:
            #
            # * RollingDepth runs an optimisation-based alignment pass.
            # * VGGT's counter integration inspects autograd edges for some
            #   operations, so inference_mode leaves it with no graph to
            #   inspect.
            # * GemDepth performs in-place metadata updates that PyTorch
            #   forbids on inference tensors, but which are valid in no_grad.
            if args.model == "vggt-omega":
                flop_context = torch.enable_grad()
            elif args.model == "rolling-depth":
                flop_context = contextlib.nullcontext()
            elif args.model == "gem-depth":
                flop_context = torch.no_grad()
            else:
                flop_context = torch.inference_mode()

            # The VGGT adapter uses inference_mode in ordinary production
            # inference.  Tell it that this single invocation is the
            # instrumented FLOP pass so it can preserve the required graph.
            profile_flag = "RPX_FLOP_PROFILE"
            previous_profile_flag = os.environ.get(profile_flag)
            if args.model == "vggt-omega":
                os.environ[profile_flag] = "1"
            try:
                with flop_context, FlopCounterMode(display=False) as counter:
                    output = canonical_forward()
                    _sync(torch)
                    del output
            finally:
                if previous_profile_flag is None:
                    os.environ.pop(profile_flag, None)
                else:
                    os.environ[profile_flag] = previous_profile_flag
            counted = _sum_flops(counter.get_flop_counts())
            if counted > 0:
                flops = counted
                flop_status = "measured"
                flop_method = "torch.utils.flop_counter.FlopCounterMode"
            else:
                flop_status = "unsupported_or_zero"
        except Exception as exc:  # noqa: BLE001
            # A full VGGT clip fits normally on the 48 GiB evaluation GPU,
            # but retaining its autograd graph solely for FlopCounterMode
            # does not.  Fall back to the PyTorch profiler, which performs
            # the same inference-mode forward and reports its supported
            # ATen-op FLOP estimates without keeping that graph alive.
            if args.model == "vggt-omega":
                primary_error = f"{type(exc).__name__}:{exc}"
                print(
                    "[profile] VGGT FlopCounterMode failed; "
                    "retrying with inference-mode torch.profiler",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    counted = _torch_profiler_flops(canonical_forward, torch)
                    if counted > 0:
                        flops = counted
                        flop_status = "measured:torch_profiler_fallback"
                        flop_method = (
                            "torch.profiler.key_averages(with_flops=True)"
                        )
                    else:
                        flop_status = "unsupported_or_zero:torch_profiler"
                        flop_method = (
                            "torch.profiler.key_averages(with_flops=True)"
                        )
                except Exception as fallback_exc:  # noqa: BLE001
                    flop_status = (
                        f"failed:{primary_error};"
                        f"fallback:{type(fallback_exc).__name__}:{fallback_exc}"
                    )
            else:
                flop_status = f"failed:{type(exc).__name__}:{exc}"

    p50_ms = _percentile(elapsed_ms, 50)
    p95_ms = _percentile(elapsed_ms, 95)
    p99_ms = _percentile(elapsed_ms, 99)
    mean_ms = statistics.fmean(elapsed_ms)
    throughput = frames_per_sample * 1000.0 / p50_ms

    gpu_name = None
    gpu_total_mb = None
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_total_mb = (
            torch.cuda.get_device_properties(gpu_index).total_memory / (1024**2)
        )

    payload: dict[str, Any] = {
        "model": args.model,
        "task": args.task,
        "input_protocol": input_protocol,
        "frames_per_sample": frames_per_sample,
        "precision": getattr(model, "native_precision", args.precision),
        "parameter_dtypes": dtypes,
        "batch_size": 1,
        "distinct_samples": len(forwards),
        "warmup_forwards": args.warmup,
        "repetitions_per_sample": args.repeats,
        "timed_forwards": len(forwards) * args.repeats,
        "params_total": total_params,
        "params_total_m": total_params / 1e6 if total_params else None,
        "params_trainable": trainable_params,
        "parameter_storage_bytes": parameter_bytes,
        "parameter_storage_mb": (
            parameter_bytes / (1024**2) if parameter_bytes else None
        ),
        "flops_per_sample": flops,
        "flops_per_sample_g": flops / 1e9 if flops else None,
        "flops_per_frame_g": (
            flops / frames_per_sample / 1e9 if flops else None
        ),
        "macs_per_sample": flops / 2 if flops else None,
        "macs_per_sample_g": flops / 2e9 if flops else None,
        "flop_status": flop_status,
        "flop_method": flop_method,
        "latency_p50_ms_per_sample": p50_ms,
        "latency_p95_ms_per_sample": p95_ms,
        "latency_p99_ms_per_sample": p99_ms,
        "latency_mean_ms_per_sample": mean_ms,
        "latency_p50_ms_per_frame": p50_ms / frames_per_sample,
        "latency_samples_ms": elapsed_ms,
        "latency_records": latency_records,
        "throughput_frames_per_second": throughput,
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
        "peak_cpu_rss_mb": peak_cpu_rss_mb,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "adapter_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "manifest_path": str(args.manifest_path),
        "dataset_revision": "2e2a387f7f93e98c177b2e039c141eacda94e5fc",
        "code_sha": os.environ.get("RPX_GIT_SHA"),
        "container_image": os.environ.get("PROFILE_CONTAINER_IMAGE"),
        "concurrent_host_load": os.environ.get(
            "PROFILE_CONCURRENT_HOST_LOAD", "unknown"
        ),
        "profiled_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)

    gc.collect()
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[profile] saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()

"""Measure a callable on fixed NPZ inputs with warmup and CUDA synchronization.

The callable is imported as module:function and receives the NPZ arrays as
keyword arguments. It must include the preprocessing/postprocessing intended
for measurement. Inputs are loaded once, outside timing. This measures repeated
calls on fixed inputs, not accuracy, distinct dataset samples, or throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..exceptions import ConfigError
from ..profiler import LatencyProfiler, MemoryProfiler, SystemCard


def load_callable(name: str) -> Callable[..., Any]:
    """Load a model callable without imposing a framework or API client."""
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute:
        raise ConfigError("Model must use module:function notation")
    fn = getattr(importlib.import_module(module), attribute)
    if not callable(fn):
        raise ConfigError(f"{name} is not callable")
    return fn


def measure(
    fn: Callable[..., Any],
    inputs: dict[str, np.ndarray],
    *,
    device: str = "cpu",
    warmup: int = 10,
    repeats: int = 1000,
    precision: str = "unspecified",
    unit: str = "request",
) -> dict[str, Any]:
    """Return measured percentiles, raw timings, memory and a system card.

    CUDA measurements synchronize all visible devices before and after every
    call. CUDA must be available when requested. CPU RSS is a process-lifetime
    high-water mark; CUDA values are allocator peaks per visible device.
    Precision is a recorded description, never an automatic model conversion.
    """
    if device not in {"cpu", "cuda"} or warmup < 0 or repeats < 1:
        raise ConfigError("device must be cpu/cuda, warmup >= 0, repeats >= 1")
    if not inputs or not precision.strip() or not unit.strip():
        raise ConfigError("Provide inputs, precision and a measurement unit")
    torch: Any = None
    cuda_devices: list[int] = []
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        if device == "cuda":
            raise ConfigError("CUDA profiling requires PyTorch and an available CUDA GPU") from None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("CUDA requested but no CUDA GPU is available")
        cuda_devices = list(range(torch.cuda.device_count()))

    def sync() -> None:
        for index in cuda_devices:
            torch.cuda.synchronize(index)

    for _ in range(warmup):
        sync()
        fn(**inputs)
        sync()
    memory = MemoryProfiler(sample_cuda=False, sample_mps=False).reset()
    for index in cuda_devices:
        torch.cuda.reset_peak_memory_stats(index)
    latency = LatencyProfiler(warmup=0)
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        prediction = fn(**inputs)
        sync()
        latency.add_sample_seconds(time.perf_counter() - started)
        del prediction
    peaks = memory.peaks()
    cuda_peaks = {
        str(index): torch.cuda.max_memory_allocated(index) / (1024**2) for index in cuda_devices
    }
    card = SystemCard.auto_detect(precision=precision)
    # CPU runs should not present a visible but unused GPU as measured hardware.
    if device == "cpu":
        card.gpu_name, card.gpu_memory_gb, card.gpu_count = "", 0.0, 0
    card.input_resolution = "caller_defined; see inputs"
    return {
        "timing_scope": "synchronized_callable_wall_time; preloaded_inputs",
        "device": device,
        "unit": unit,
        "warmup_calls": warmup,
        "measured_calls": repeats,
        "inputs": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in inputs.items()
        },
        "latency": latency.percentiles(),
        "latency_samples_ms": list(latency.samples_ms()),
        "memory": {
            "cpu_process_lifetime_peak_mb": peaks["cpu_mb"],
            "cuda_peak_allocated_mb_by_visible_device": cuda_peaks,
        },
        "system_card": card.to_dict(),
        "notes": [
            "Warmup excluded. Fixed-input repeats are not distinct samples.",
            "Precision and callable device placement are caller-controlled.",
            "API callables include transport time and exclude remote GPU accounting.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="module:function")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", default="unspecified")
    parser.add_argument("--unit", default="request", help="frame, clip, pair or request")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as payload:
        inputs = {key: payload[key] for key in payload.files}
    result = measure(
        load_callable(args.model),
        inputs,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        precision=args.precision,
        unit=args.unit,
    )
    result["model_callable"] = args.model
    result["inputs_sha256"] = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

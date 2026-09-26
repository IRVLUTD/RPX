"""Opt-in measurement of real model adapter calls for hardware reporting.

Timing covers the synchronized adapter call, including adapter preprocessing
and postprocessing. It excludes weights and dataset loading, evaluation, and
prediction persistence. The raw call records remain available for audit.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any, Callable

import numpy as np

from .exceptions import ConfigError


class ProfileComplete(BaseException):
    """Stop a dedicated profiling process after reaching its sample budget."""


class _GlobalOnlyTracker:
    parents = {"Global"}
    is_bw = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _global_flop_counter():
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    if not hasattr(counter, "mod_tracker"):
        raise ConfigError("Unsupported PyTorch FLOP counter: missing mod_tracker")
    counter.mod_tracker = _GlobalOnlyTracker()
    return counter


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parameter_inventory(model: Any) -> dict:
    """Count unique loaded Parameters across an adapter and its components."""
    import torch

    seen: set[int] = set()
    parameters: dict[int, Any] = {}
    modules: list[str] = []
    provenance: dict[str, dict] = {}
    queue = [("adapter", model, 0)]
    ignored = (
        ModuleType,
        type,
        FunctionType,
        MethodType,
        BuiltinFunctionType,
        str,
        bytes,
        int,
        float,
        Path,
        np.ndarray,
        torch.Tensor,
    )
    while queue:
        name, obj, depth = queue.pop()
        if obj is None or id(obj) in seen or depth > 6:
            continue
        seen.add(id(obj))
        if isinstance(obj, ignored):
            continue
        declared = {}
        for key in (
            "model_id",
            "model_revision",
            "source_revision",
            "checkpoint_filename",
            "checkpoint_path",
            "checkpoint_sha256",
        ):
            value = inspect.getattr_static(obj, key, None)
            if isinstance(value, (str, Path)):
                declared[key] = str(value)
        if declared:
            provenance[name] = declared
        if isinstance(obj, torch.nn.Module):
            modules.append(name)
            parameters.update({id(parameter): parameter for parameter in obj.parameters()})
            continue
        if isinstance(obj, dict):
            queue.extend((f"{name}.{key}", value, depth + 1) for key, value in obj.items())
        elif isinstance(obj, (tuple, list)):
            queue.extend((f"{name}.{key}", value, depth + 1) for key, value in enumerate(obj))
        else:
            try:
                attributes = object.__getattribute__(obj, "__dict__")
            except (AttributeError, TypeError):
                continue
            if isinstance(attributes, dict):
                queue.extend(
                    (f"{name}.{key}", value, depth + 1)
                    for key, value in attributes.items()
                    if not isinstance(value, (str, bytes, int, float, np.ndarray, torch.Tensor))
                    and not key.startswith("__")
                )
    count = sum(parameter.numel() for parameter in parameters.values()) if modules else None
    return {
        "parameter_count": count,
        "params_m": count / 1e6 if count is not None else None,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters.values())
        if modules
        else None,
        "parameter_status": "loaded_modules" if modules else "unavailable",
        "parameter_modules": sorted(modules),
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters.values()}),
        "checkpoint_metadata": provenance,
        "parameter_scope": (
            "unique loaded parameters, including auxiliary models; "
            "not necessarily all active per call"
        ),
    }


class HardwareProfile:
    def __init__(self, model: Any, config: dict) -> None:
        import torch

        self.model = model
        self.config = config
        self.output = Path(config["output_dir"])
        self.records: list[dict] = []
        self.target = int(config["samples"])
        self.warmup = int(config.get("warmup_calls", 1))
        self.flop_calls = int(config.get("flop_calls", 1))
        self.output.mkdir(parents=True, exist_ok=True)
        cuda = torch.cuda.is_available()
        self.system = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "rpx_git_sha": os.environ.get("RPX_GIT_SHA"),
            "gpu_name": torch.cuda.get_device_name() if cuda else None,
            "gpu_capability": list(torch.cuda.get_device_capability()) if cuda else None,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory
            if cuda
            else None,
        }
        self.inventory = parameter_inventory(model)
        environment = Path(sys.prefix) / "rpx-environment.json"
        self.provenance = json.loads(environment.read_text()) if environment.is_file() else None
        self.write("running")

    def finish(self) -> None:
        self.write("insufficient_samples")
        raise ProfileComplete()

    def call(
        self, fn: Callable, *, units: int, metadata: dict, cache_hits: Callable | None = None
    ) -> Any:
        import torch

        if units < 1:
            raise ConfigError("Profiling requires at least one input unit.")
        index = len(self.records)
        instrument = index < self.flop_calls
        excluded = index < max(self.warmup, self.flop_calls)
        counter = _global_flop_counter() if instrument else None
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            if counter is not None:
                with counter:
                    result = fn()
            else:
                result = fn()
            if cuda:
                torch.cuda.synchronize()
        except Exception as exc:
            self.write("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - started
        if cache_hits is not None and any(cache_hits() or []):
            self.write("failed", error="Cached predictions encountered in fresh profile")
            raise ConfigError("Fresh hardware profiling cannot include cached predictions.")
        total_flops = counter.get_total_flops() if counter is not None else None
        operations = (
            {
                str(key): int(value)
                for key, value in counter.get_flop_counts().get("Global", {}).items()
            }
            if counter
            else {}
        )
        record = {
            "call_index": index,
            "units": units,
            "metadata": metadata,
            "excluded_from_timing": excluded,
            "instrumented": instrument,
            "adapter_wall_time_s": elapsed,
            "counted_flops": int(total_flops) if total_flops is not None else None,
            "counted_ops": operations,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
            if cuda
            else None,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if cuda else None,
        }
        self.records.append(record)
        with (self.output / "hardware_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if index == 0:
            self.inventory = parameter_inventory(self.model)
        measured = sum(r["units"] for r in self.records if not r["excluded_from_timing"])
        self.write("complete" if measured >= self.target else "running")
        print(
            f"[hardware] measured {measured}/{self.target} {self.config['unit']} units", flush=True
        )
        if measured >= self.target:
            raise ProfileComplete()
        return result

    def write(self, status: str, **extra: Any) -> None:
        measured = [record for record in self.records if not record["excluded_from_timing"]]
        units = sum(record["units"] for record in measured)
        seconds = sum(record["adapter_wall_time_s"] for record in measured)
        per_unit_ms = [
            record["adapter_wall_time_s"] * 1000 / record["units"]
            for record in measured
            for _ in range(record["units"])
        ]
        flop_records = [record for record in self.records if record["instrumented"]]
        flop_units = sum(record["units"] for record in flop_records)
        flops = sum(record["counted_flops"] or 0 for record in flop_records)
        gflops = flops / flop_units / 1e9 if flop_units and flops > 0 else None

        def peak(key: str):
            values = [record[key] for record in measured if record[key] is not None]
            return max(values) if values else None

        report = {
            "schema_version": "rpx.hardware.v1",
            **self.config,
            "status": status,
            **extra,
            **self.inventory,
            "system": self.system,
            "model_environment": self.provenance,
            "declared_precision": getattr(self.model, "native_precision", None),
            "measured_units": units,
            "measured_calls": len(measured),
            "excluded_calls": len(self.records) - len(measured),
            "budget_policy": "at least requested units",
            "timing_scope": (
                "synchronized adapter call including adapter preprocessing/postprocessing; "
                "excludes dataset loading, weights loading, evaluation and persistence"
            ),
            "latency_kind": "adapter-call time per input frame at batch size 1",
            "latency_mean_ms": seconds * 1000 / units if units else None,
            **{
                f"latency_p{quantile}_ms": float(np.percentile(per_unit_ms, quantile))
                if per_unit_ms
                else None
                for quantile in (50, 95, 99)
            },
            "throughput_units_per_s": units / seconds if seconds > 0 else None,
            "measured_adapter_wall_time_s": seconds,
            "peak_cuda_allocated_mib": peak("peak_cuda_allocated_mib"),
            "peak_cuda_reserved_mib": peak("peak_cuda_reserved_mib"),
            "flops_g_per_unit": gflops,
            "macs_g_equivalent_per_unit": gflops / 2 if gflops is not None else None,
            "flop_status": "counted_ops_only" if gflops is not None else "unavailable",
            "flop_coverage": (
                "not certified complete: custom/fused/non-PyTorch operations may be uncounted"
            ),
            "flop_convention": (
                "PyTorch registered formulas; multiply-add=2 FLOPs; MAC equivalent=FLOPs/2"
            ),
            "flop_profiled_units": flop_units,
            "flop_profiled_calls": len(flop_records),
        }
        atomic_json(self.output / "hardware_profile.json", report)


def batch_metadata(batch: Any) -> dict:
    rows = []
    for sample in batch:
        rgb = getattr(sample, "rgb_seq", None)
        if rgb is None:
            rgb = getattr(sample, "rgb", None)
        rows.append(
            {
                "id": str(getattr(sample, "id", "")),
                "input_shape": list(rgb.shape) if rgb is not None else None,
            }
        )
    return {"batch_size": len(batch), "samples": rows}

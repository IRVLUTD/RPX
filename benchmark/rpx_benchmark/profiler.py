"""Hardware-agnostic model efficiency profiling for RPX.

Three-tier efficiency reporting
===============================

**Tier 1 — Model properties (fully hardware-agnostic):**
    Parameters (M), FLOPs (G), MACs (G), estimated DRAM traffic (GB),
    arithmetic intensity (FLOP/Byte).  Identical on any hardware.

**Tier 2 — Roofline latency bounds (hardware-parametric):**
    Given Tier-1 numbers and a :class:`GPUSpec`, compute theoretical
    lower-bound latency on any GPU via the roofline model:
    ``latency = max(FLOPs/peak_flops, traffic/peak_bw)``.
    Published for reference GPUs; anyone can recompute for their own.

**Tier 3 — Measured latency (hardware-specific):**
    Wall-clock p50/p95/p99, peak memory per backend, throughput.
    Always accompanied by a :class:`SystemCard`.

FLOPs convention (for reproducibility):
  - Single forward pass, batch_size=1
  - Input resolution: task default or caller-specified
  - Precision: FP32 for counting (hardware-agnostic)
  - API-only models: report None (shown as "N/A (API)" in tables)
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =========================================================================== #
# Tier 1: Hardware-agnostic model properties
# =========================================================================== #

@dataclass
class EfficiencyMetadata:
    """Hardware-agnostic efficiency metadata for a model.

    Tier-1 fields (identical on any hardware):
        ``params_m``, ``flops_g``, ``macs_g``, ``memory_traffic_gb``,
        ``arithmetic_intensity``.

    Tier-3 fields (hardware-specific, filled post-run):
        ``latency_*``, ``peak_*_mb``.
    """

    # --- Tier 1: hardware-agnostic ----------------------------------------
    params_m: float | None = None            # Total parameters in millions
    flops_g: float | None = None             # FLOPs (giga) at batch=1, FP32
    macs_g: float | None = None              # MACs (giga) = FLOPs / 2
    actmem_gb_fp16: float | None = None      # Activation memory at FP16 (GB)
    memory_traffic_gb: float | None = None   # Estimated DRAM read+write (GB)
    arithmetic_intensity: float | None = None  # FLOPs / Bytes (FLOP/Byte)

    latency_ms_per_sample: float | None = None  # Filled post-run by the task runner
    model_type: str = "local"                # "local" | "api"
    notes: str = ""                          # e.g. precision mode, resolution override

    # --- Tier 2: roofline bounds (computed on demand) ---------------------
    roofline: Dict[str, "RooflineBound"] | None = None

    # --- Tier 3: measured (hardware-specific) -----------------------------
    peak_cpu_mb: float | None = None
    peak_cuda_mb: float | None = None
    peak_mps_mb: float | None = None

    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None

    # --- System card (attached when measured) ------------------------------
    system_card: "SystemCard | None" = None

    def derive_tier1(self) -> None:
        """Fill in derived Tier-1 fields from primary measurements.

        Call after ``flops_g`` and ``params_m`` are populated.
        """
        if self.flops_g is not None and self.macs_g is None:
            self.macs_g = round(self.flops_g / 2.0, 3)

        if self.memory_traffic_gb is None and self.params_m is not None:
            # Conservative estimate: read all params (FP32 = 4 bytes) once
            # + read/write activations ~ 2x param bytes.  This is a
            # floor; actual traffic depends on layer structure and caching.
            param_bytes = self.params_m * 1e6 * 4  # FP32
            self.memory_traffic_gb = round(param_bytes * 3 / 1e9, 4)

        if (self.arithmetic_intensity is None
                and self.flops_g is not None
                and self.memory_traffic_gb is not None
                and self.memory_traffic_gb > 0):
            total_flops = self.flops_g * 1e9
            total_bytes = self.memory_traffic_gb * 1e9
            self.arithmetic_intensity = round(total_flops / total_bytes, 2)

    def compute_roofline(
        self,
        specs: Sequence["GPUSpec"] | None = None,
    ) -> Dict[str, "RooflineBound"]:
        """Compute Tier-2 roofline latency bounds for reference GPUs.

        Parameters
        ----------
        specs : sequence of GPUSpec, optional
            GPU specifications to compute bounds for.  Defaults to
            :data:`REFERENCE_GPUS` (A100, RTX 4090, Jetson Orin).

        Returns
        -------
        dict mapping GPU name to :class:`RooflineBound`.
        """
        if specs is None:
            specs = list(REFERENCE_GPUS.values())

        self.derive_tier1()
        if self.flops_g is None or self.memory_traffic_gb is None:
            return {}

        bounds: Dict[str, RooflineBound] = {}
        for gpu in specs:
            bounds[gpu.name] = RooflineBound.from_model_and_gpu(
                flops_g=self.flops_g,
                traffic_gb=self.memory_traffic_gb,
                gpu=gpu,
            )
        self.roofline = bounds
        return bounds

    def to_table_row(self) -> dict:
        """Produce result-table-ready dict (None -> 'N/A (API)' for API models)."""
        na = "N/A (API)" if self.model_type == "api" else None
        row = {
            "type": self.model_type,
            "params_m": self.params_m if self.params_m is not None else na,
            "flops_g": self.flops_g if self.flops_g is not None else na,
            "macs_g": self.macs_g if self.macs_g is not None else na,
            "memory_traffic_gb": self.memory_traffic_gb,
            "arithmetic_intensity": self.arithmetic_intensity,
            "actmem_gb_fp16": self.actmem_gb_fp16,
            "latency_ms_per_sample": self.latency_ms_per_sample,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "peak_cpu_mb": self.peak_cpu_mb,
            "peak_cuda_mb": self.peak_cuda_mb,
            "peak_mps_mb": self.peak_mps_mb,
        }
        if self.roofline:
            for gpu_name, bound in self.roofline.items():
                key = gpu_name.lower().replace(" ", "_").replace("-", "_")
                row[f"roofline_{key}_ms"] = bound.latency_ms
                row[f"bottleneck_{key}"] = bound.bottleneck
        if self.system_card is not None:
            row["system_card"] = self.system_card.to_dict()
        return row


# =========================================================================== #
# Tier 2: GPU specs + Roofline bounds
# =========================================================================== #

@dataclass(frozen=True)
class GPUSpec:
    """Hardware specification for a reference GPU.

    Used to compute roofline latency bounds.  All numbers are for the
    precision specified by ``precision`` (default FP32).

    Attributes
    ----------
    name : str
        Human-readable name, e.g. ``"A100-80GB"``.
    peak_tflops : float
        Peak throughput in tera-FLOP/s at the stated precision.
    memory_bw_gbps : float
        Peak DRAM bandwidth in GB/s.
    memory_gb : float
        Total device memory in GB.
    precision : str
        Precision these specs refer to (``"fp32"``, ``"fp16"``, etc.).
    """

    name: str
    peak_tflops: float
    memory_bw_gbps: float
    memory_gb: float
    precision: str = "fp32"


#: Reference GPUs covering datacenter, workstation, and edge-robot scenarios.
#: Sources: official NVIDIA data sheets.
REFERENCE_GPUS: Dict[str, GPUSpec] = {
    "A100-80GB": GPUSpec(
        name="A100-80GB",
        peak_tflops=19.5,
        memory_bw_gbps=2039.0,
        memory_gb=80.0,
    ),
    "RTX 4090": GPUSpec(
        name="RTX 4090",
        peak_tflops=82.6,
        memory_bw_gbps=1008.0,
        memory_gb=24.0,
    ),
    "Jetson Orin 64GB": GPUSpec(
        name="Jetson Orin 64GB",
        peak_tflops=5.3,
        memory_bw_gbps=204.8,
        memory_gb=64.0,
        precision="fp32",
    ),
}


@dataclass(frozen=True)
class RooflineBound:
    """Roofline-model latency bound for one model on one GPU.

    The roofline model (Williams et al., 2009) gives a lower-bound
    latency based on whichever of compute or memory bandwidth is the
    bottleneck:

    .. math::

        T_{\\text{roofline}} = \\max\\!\\Big(
            \\frac{\\text{FLOPs}}{\\text{peak FLOP/s}},\\;
            \\frac{\\text{Traffic}}{\\text{peak BW}}
        \\Big)

    Attributes
    ----------
    gpu_name : str
    compute_ms : float
        Time if fully compute-bound (FLOPs / peak_tflops).
    memory_ms : float
        Time if fully memory-bound (traffic / peak_bw).
    latency_ms : float
        ``max(compute_ms, memory_ms)`` — the roofline lower bound.
    bottleneck : str
        ``"compute"`` or ``"memory"``.
    """

    gpu_name: str
    compute_ms: float
    memory_ms: float
    latency_ms: float
    bottleneck: str

    @classmethod
    def from_model_and_gpu(
        cls,
        flops_g: float,
        traffic_gb: float,
        gpu: GPUSpec,
    ) -> "RooflineBound":
        """Compute the roofline bound for a model on a GPU.

        Parameters
        ----------
        flops_g : float
            Model forward-pass FLOPs in giga-ops.
        traffic_gb : float
            Estimated DRAM traffic in GB.
        gpu : GPUSpec
            Target GPU specification.
        """
        # peak_tflops is tera-FLOP/s = 1e12 FLOP/s
        # flops_g is giga-FLOP = 1e9 FLOP
        # compute_s = (flops_g * 1e9) / (peak_tflops * 1e12)
        #           = flops_g / (peak_tflops * 1e3)
        compute_s = flops_g / (gpu.peak_tflops * 1e3)
        memory_s = traffic_gb / gpu.memory_bw_gbps

        compute_ms = round(compute_s * 1e3, 4)
        memory_ms = round(memory_s * 1e3, 4)
        latency_ms = round(max(compute_ms, memory_ms), 4)
        bottleneck = "compute" if compute_ms >= memory_ms else "memory"

        return cls(
            gpu_name=gpu.name,
            compute_ms=compute_ms,
            memory_ms=memory_ms,
            latency_ms=latency_ms,
            bottleneck=bottleneck,
        )


# =========================================================================== #
# Tier 3: System Card (attached to measured results)
# =========================================================================== #

@dataclass
class SystemCard:
    """Full hardware/software description for reproducibility.

    Attached to every set of Tier-3 measurements so readers know
    exactly what hardware the numbers were collected on.  Follows the
    MLPerf convention of reporting system details alongside results.
    """

    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_count: int = 1
    cpu_name: str = ""
    ram_gb: float = 0.0
    cuda_version: str = ""
    pytorch_version: str = ""
    python_version: str = ""
    precision: str = "fp32"
    batch_size: int = 1
    input_resolution: str = "640x480"
    os: str = ""
    driver_version: str = ""

    @classmethod
    def auto_detect(
        cls,
        precision: str = "fp32",
        batch_size: int = 1,
        input_resolution: str = "640x480",
    ) -> "SystemCard":
        """Best-effort auto-detection of the current system.

        Fills every field it can; leaves empty strings for fields
        that require libraries not installed (e.g. torch for CUDA info).
        """
        card = cls(
            precision=precision,
            batch_size=batch_size,
            input_resolution=input_resolution,
            python_version=platform.python_version(),
            os=f"{platform.system()} {platform.release()}",
        )

        # CPU
        card.cpu_name = platform.processor() or platform.machine()

        # RAM
        try:
            import psutil  # noqa: PLC0415
            card.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except ImportError:
            pass

        # GPU (torch)
        try:
            import torch  # noqa: PLC0415
            card.pytorch_version = torch.__version__
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                card.gpu_name = props.name
                card.gpu_memory_gb = round(props.total_memory / (1024 ** 3), 1)
                card.gpu_count = torch.cuda.device_count()
                card.cuda_version = torch.version.cuda or ""
                # Driver version from nvidia-smi is not in torch; leave blank
                # unless the user fills it manually.
        except ImportError:
            pass

        return card

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_count": self.gpu_count,
            "cpu_name": self.cpu_name,
            "ram_gb": self.ram_gb,
            "cuda_version": self.cuda_version,
            "pytorch_version": self.pytorch_version,
            "python_version": self.python_version,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "input_resolution": self.input_resolution,
            "os": self.os,
            "driver_version": self.driver_version,
        }

    def summary(self) -> str:
        """One-line summary for table footers / report headers."""
        parts = []
        if self.gpu_name:
            parts.append(f"{self.gpu_name} ({self.gpu_memory_gb}GB)")
        if self.cpu_name:
            parts.append(self.cpu_name)
        parts.append(f"PyTorch {self.pytorch_version}" if self.pytorch_version else "")
        parts.append(f"CUDA {self.cuda_version}" if self.cuda_version else "")
        parts.append(self.precision)
        return " | ".join(p for p in parts if p)


# =========================================================================== #
# Memory traffic estimation
# =========================================================================== #

def estimate_memory_traffic_gb(
    model: Any,
    input_shape: Tuple[int, ...] = (3, 480, 640),
    device: str = "cpu",
    precision_bytes: int = 4,
) -> float | None:
    """Estimate total DRAM traffic (GB) for a single forward pass.

    Strategy (in order of preference):

    1. **torch.profiler** with ``profile_memory=True`` — most accurate,
       captures actual CUDA memory events.  Requires torch >= 2.0 and
       a CUDA device.
    2. **Analytical estimate** — conservative lower bound:
       ``param_bytes + 2 * activation_bytes``.  Works on any backend.

    Parameters
    ----------
    model : nn.Module or similar
        The model to profile.
    input_shape : tuple
        ``(C, H, W)`` — batch dim is prepended.
    device : str
        Device for the dummy input.
    precision_bytes : int
        Bytes per element (4 for FP32, 2 for FP16/BF16).

    Returns
    -------
    float or None
        Estimated DRAM traffic in GB, or None if estimation fails.
    """
    # Strategy 1: torch.profiler
    try:
        import torch  # noqa: PLC0415
        if device != "cpu" and torch.cuda.is_available():
            dummy = torch.zeros(1, *input_shape, device=device)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                with torch.no_grad():
                    model(dummy)

            # Sum all CUDA memory allocations as a proxy for traffic.
            # This overcounts (includes temporaries freed mid-forward)
            # but is more realistic than the analytical estimate.
            total_bytes = 0
            for evt in prof.key_averages():
                mem = getattr(evt, "cuda_memory_usage", 0) or 0
                if mem > 0:
                    total_bytes += mem
            if total_bytes > 0:
                return round(total_bytes / 1e9, 4)
    except (ImportError, RuntimeError, AttributeError):
        pass

    # Strategy 2: analytical estimate
    try:
        params_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )
        # Heuristic: activations ~ 2x param memory for typical vision
        # models (each layer reads previous activation, writes new one).
        activation_bytes = params_bytes * 2
        total = params_bytes + activation_bytes  # read params + r/w activations
        return round(total / 1e9, 4)
    except (AttributeError, TypeError):
        pass

    return None


# --------------------------------------------------------------------------- #
# Memory profiler — CPU / CUDA / MPS backends
# --------------------------------------------------------------------------- #

def _cpu_peak_rss_mb() -> float | None:
    """Current process peak RSS in MB; None if neither psutil nor resource
    can produce a number."""
    # psutil first (cross-platform, no import of ``resource``).
    try:
        import psutil  # noqa: PLC0415
        proc = psutil.Process()
        info = proc.memory_info()
        # ``peak_wset`` (Windows) or ``peak_rss`` (macOS 12+) are rare;
        # fall back to ``rss`` if the peak variant isn't available.
        peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
        if peak is not None:
            return round(peak / (1024 * 1024), 3)
        return round(info.rss / (1024 * 1024), 3)
    except ImportError:
        pass
    try:
        import resource  # noqa: PLC0415 — POSIX only
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ``ru_maxrss`` units: KB on Linux, bytes on macOS. We normalise
        # by checking the magnitude — a process using hundreds of GB is
        # exceedingly unlikely, so > 10M means the value is in bytes.
        maxrss = usage.ru_maxrss
        if maxrss > 10_000_000:
            return round(maxrss / (1024 * 1024), 3)
        return round(maxrss / 1024, 3)
    except ImportError:
        return None


def _cuda_peak_alloc_mb() -> float | None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    except Exception:
        return None


def _mps_current_alloc_mb() -> float | None:
    """Apple Silicon MPS memory — uses ``torch.mps.current_allocated_memory``.

    ``torch.mps`` has no ``max_memory_allocated`` until torch 2.3; we
    fall back to the current alloc so we still surface *something*.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    mps_mod = getattr(torch, "mps", None)
    if mps_mod is None:
        return None
    try:
        if not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return None
    except AttributeError:
        return None
    fn = getattr(mps_mod, "current_allocated_memory", None)
    if fn is None:
        return None
    try:
        return round(fn() / (1024 * 1024), 3)
    except Exception:
        return None


@dataclass
class MemoryProfiler:
    """Hardware-agnostic memory profiler covering CPU / CUDA / MPS.

    Usage::

        mp = MemoryProfiler().reset()
        # ... run inference ...
        peaks = mp.peaks()   # {"cpu_mb": ..., "cuda_mb": ..., "mps_mb": ...}

    Each backend returns ``None`` when it cannot be sampled (library
    missing, backend unavailable, measurement unsupported). The caller
    is responsible for deciding whether ``None`` is informative
    (report as ``"n/a"``) or an error.
    """

    sample_cpu: bool = True
    sample_cuda: bool = True
    sample_mps: bool = True

    def reset(self) -> "MemoryProfiler":
        """Clear accumulated peaks on every enabled backend."""
        if self.sample_cuda:
            try:
                import torch  # noqa: PLC0415
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass
        # CPU and MPS backends don't expose a reset primitive — the
        # sampled value is "current" (MPS) or "peak-since-process-start"
        # (CPU ru_maxrss). Document the limitation in the docstring.
        return self

    def peaks(self) -> dict:
        return {
            "cpu_mb": _cpu_peak_rss_mb() if self.sample_cpu else None,
            "cuda_mb": _cuda_peak_alloc_mb() if self.sample_cuda else None,
            "mps_mb": _mps_current_alloc_mb() if self.sample_mps else None,
        }


# --------------------------------------------------------------------------- #
# Latency profiler — percentiles + mean
# --------------------------------------------------------------------------- #

@dataclass
class LatencyProfiler:
    """Accumulate per-sample latencies and report percentiles.

    Percentiles use linear interpolation (``numpy.percentile`` default)
    and skip a user-configurable number of warmup samples so the
    reported numbers reflect steady-state throughput rather than the
    first-batch JIT compile.
    """

    warmup: int = 1
    _samples_ms: List[float] = field(default_factory=list)

    def add_sample_seconds(self, seconds: float) -> None:
        self._samples_ms.append(float(seconds) * 1000.0)

    def add_batch_seconds(self, batch_seconds: float, batch_size: int) -> None:
        """Amortise a batch timing over ``batch_size`` sample entries."""
        if batch_size <= 0:
            return
        per = batch_seconds / batch_size
        self._samples_ms.extend([per * 1000.0] * batch_size)

    def percentiles(self) -> dict:
        """Return ``{p50_ms, p95_ms, p99_ms, mean_ms}`` in milliseconds.

        Returns ``{k: None for k in keys}`` when fewer than one sample
        exists after warmup trimming.
        """
        trimmed = self._samples_ms[self.warmup:] or self._samples_ms
        if not trimmed:
            return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
        try:
            import numpy as np  # noqa: PLC0415
            p50, p95, p99 = np.percentile(trimmed, [50, 95, 99])
            return {
                "p50_ms": round(float(p50), 3),
                "p95_ms": round(float(p95), 3),
                "p99_ms": round(float(p99), 3),
                "mean_ms": round(float(np.mean(trimmed)), 3),
            }
        except ImportError:
            trimmed_sorted = sorted(trimmed)
            n = len(trimmed_sorted)

            def _q(frac: float) -> float:
                return trimmed_sorted[min(n - 1, int(round(frac * (n - 1))))]

            return {
                "p50_ms": round(_q(0.50), 3),
                "p95_ms": round(_q(0.95), 3),
                "p99_ms": round(_q(0.99), 3),
                "mean_ms": round(sum(trimmed) / n, 3),
            }

    def samples_ms(self) -> Sequence[float]:
        """Return the raw latency samples (post-warmup not trimmed)."""
        return list(self._samples_ms)


def count_parameters(model: Any) -> float:
    """Count trainable parameters in millions.

    Supports PyTorch nn.Module and any object with a ``parameters()`` method.
    Returns None if the model type is not supported.
    """
    try:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return round(params / 1e6, 3)
    except AttributeError:
        pass

    # JAX / Flax: model may expose a ``params`` pytree
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(model.params)
        params = sum(leaf.size for leaf in leaves)
        return round(params / 1e6, 3)
    except (AttributeError, ImportError):
        pass

    return None


def count_flops_torch(
    model: Any,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> float | None:
    """Estimate FLOPs (giga) using ``torch.utils.flop_counter`` (PyTorch >= 2.1).

    Falls back to ``fvcore`` if available.

    **OOM-safe**: when the FLOP counter runs out of CUDA memory (the
    dispatch metadata alone can consume ~3 GB on top of the model), the
    function automatically retries on CPU.  The FLOP count is identical
    regardless of device — it is a graph property, not a runtime one.

    Args:
        model: PyTorch nn.Module.
        input_shape: (C, H, W) — batch dimension is added automatically.
        device: device string for the dummy input tensor.

    Returns:
        FLOPs in giga-ops, or None if neither backend is available.
    """
    def _try_flop_count(mod: Any, dev: str) -> float | None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return None

        dummy = torch.zeros(1, *input_shape, device=dev)

        # PyTorch 2.1+ native counter
        try:
            from torch.utils.flop_counter import FlopCounterMode  # noqa: PLC0415
            with torch.no_grad():
                with FlopCounterMode(display=False) as fcm:
                    mod(dummy)
            total = _sum_nested_counts(fcm.get_flop_counts())
            if total > 0:
                return round(total / 1e9, 3)
        except ImportError:
            pass

        # fvcore fallback
        try:
            from fvcore.nn import FlopCountAnalysis  # noqa: PLC0415
            flops = FlopCountAnalysis(mod, dummy)
            return round(flops.total() / 1e9, 3)
        except ImportError:
            pass

        return None

    # Helper for nested FlopCounterMode dicts (same logic as runner.py)
    def _sum_nested_counts(node: Any) -> int:
        if isinstance(node, dict):
            return sum(_sum_nested_counts(v) for v in node.values())
        try:
            return int(node)
        except (TypeError, ValueError):
            return 0

    # Try on the requested device first
    try:
        result = _try_flop_count(model, device)
        if result is not None:
            return result
    except RuntimeError as exc:
        # Catch CUDA OOM and fall through to CPU retry
        if "out of memory" not in str(exc).lower():
            return None

    # CPU fallback: FLOPs are a graph property — the count is device-
    # independent.  Move the model to CPU, count, then move it back.
    if device != "cpu":
        try:
            import torch  # noqa: PLC0415
            original_device = next(model.parameters()).device
            model_cpu = model.cpu()
            result = _try_flop_count(model_cpu, "cpu")
            model.to(original_device)
            return result
        except Exception:
            # If anything goes wrong restoring the model, still return
            # None rather than crashing the caller.
            try:
                model.to(device)
            except Exception:
                pass
            return None

    return None


def profile_model(
    model: Any,
    input_shape: Tuple[int, ...] = (3, 480, 640),
    device: str = "cpu",
    model_type: str = "local",
    notes: str = "",
    compute_roofline: bool = True,
    attach_system_card: bool = True,
    precision: str = "fp32",
) -> EfficiencyMetadata:
    """Auto-profile a model and return EfficiencyMetadata.

    Fills all three tiers of efficiency reporting:

    - **Tier 1**: params, FLOPs, MACs, memory traffic, arithmetic
      intensity (hardware-agnostic).
    - **Tier 2**: roofline latency bounds for reference GPUs
      (hardware-parametric).
    - **Tier 3**: system card for the current machine
      (hardware-specific; measured latency is added later by the runner).

    Parameters
    ----------
    model : nn.Module or similar
        The model to profile.
    input_shape : tuple
        ``(C, H, W)`` for FLOPs counting. Default 640x480 RGB.
    device : str
        Device for the dummy input tensor.
    model_type : str
        ``"local"`` or ``"api"``.
    notes : str
        Free-text notes (e.g. ``"ViT-L/14, FP16 inference"``).
    compute_roofline : bool
        If True, compute Tier-2 roofline bounds for reference GPUs.
    attach_system_card : bool
        If True, auto-detect and attach a SystemCard.
    precision : str
        Precision string for the system card (``"fp32"``, ``"fp16"``).

    Returns
    -------
    EfficiencyMetadata
        With Tier 1/2/3 fields populated where possible.
    """
    if model_type == "api":
        return EfficiencyMetadata(model_type="api", notes=notes)

    params_m = count_parameters(model)
    flops_g = count_flops_torch(model, input_shape, device=device)
    traffic_gb = estimate_memory_traffic_gb(model, input_shape, device=device)

    h, w = input_shape[-2], input_shape[-1]
    em = EfficiencyMetadata(
        params_m=params_m,
        flops_g=flops_g,
        memory_traffic_gb=traffic_gb,
        model_type=model_type,
        notes=notes,
    )

    # Derive Tier 1 (MACs, arithmetic intensity)
    em.derive_tier1()

    # Tier 2: roofline bounds
    if compute_roofline:
        em.compute_roofline()

    # Tier 3: system card
    if attach_system_card:
        em.system_card = SystemCard.auto_detect(
            precision=precision,
            input_resolution=f"{w}x{h}",
        )

    return em

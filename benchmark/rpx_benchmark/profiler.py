"""Hardware-agnostic model efficiency profiling for RPX.

Reports: parameter count (M), FLOPs/MACs (G), optional activation memory (GB).
These metadata enable fair comparison across compute environments and are
required columns in the RPX results table per the NeurIPS D&B submission.

FLOPs convention (for reproducibility):
  - Single forward pass, batch_size=1
  - Input resolution: task default or caller-specified
  - Precision: FP32 for counting (hardware-agnostic)
  - API-only models: report None (shown as "N/A (API)" in tables)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple


@dataclass
class EfficiencyMetadata:
    """Hardware-agnostic efficiency metadata for a model.

    The dataclass grew with M3 — in addition to the original params /
    FLOPs / activation-memory fields, we now also carry measured peak
    memory per backend and latency percentiles. All new fields are
    optional so existing callers don't have to update their
    construction sites.
    """

    params_m: float | None = None            # Total parameters in millions
    flops_g: float | None = None             # FLOPs (giga) at batch=1, FP32
    actmem_gb_fp16: float | None = None      # Optional activation memory at FP16 (GB)
    latency_ms_per_sample: float | None = None  # Filled post-run by the task runner
    model_type: str = "local"                # "local" | "api"
    notes: str = ""                          # e.g. precision mode, resolution override

    # --- M3 additions: measured peak memory (per backend, MB) -------------
    peak_cpu_mb: float | None = None
    peak_cuda_mb: float | None = None
    peak_mps_mb: float | None = None

    # --- M3 additions: latency percentiles (ms) ---------------------------
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None

    def to_table_row(self) -> dict:
        """Produce result-table-ready dict (None → 'N/A (API)' for API models)."""
        na = "N/A (API)" if self.model_type == "api" else None
        return {
            "type": self.model_type,
            "params_m": self.params_m if self.params_m is not None else na,
            "flops_g": self.flops_g if self.flops_g is not None else na,
            "actmem_gb_fp16": self.actmem_gb_fp16,
            "latency_ms_per_sample": self.latency_ms_per_sample,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "peak_cpu_mb": self.peak_cpu_mb,
            "peak_cuda_mb": self.peak_cuda_mb,
            "peak_mps_mb": self.peak_mps_mb,
        }


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
    """Estimate FLOPs (giga) using ``torch.utils.flop_counter`` (PyTorch ≥ 2.1).

    Falls back to ``fvcore`` if available.

    Args:
        model: PyTorch nn.Module.
        input_shape: (C, H, W) — batch dimension is added automatically.
        device: device string for the dummy input tensor.

    Returns:
        FLOPs in giga-ops, or None if neither backend is available.
    """
    try:
        import torch
        dummy = torch.zeros(1, *input_shape, device=device)

        # PyTorch 2.1+ native counter
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fcm:
                model(dummy)
            total = sum(fcm.get_flop_counts().values())
            return round(total / 1e9, 3)
        except ImportError:
            pass

        # fvcore fallback
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(model, dummy)
            return round(flops.total() / 1e9, 3)
        except ImportError:
            pass

    except ImportError:
        pass

    return None


def profile_model(
    model: Any,
    input_shape: Tuple[int, ...] = (3, 480, 640),
    device: str = "cpu",
    model_type: str = "local",
    notes: str = "",
) -> EfficiencyMetadata:
    """Auto-profile a model and return EfficiencyMetadata.

    Args:
        model: a model object (PyTorch nn.Module recommended).
        input_shape: (C, H, W) for FLOPs counting. Default 640×480 RGB.
        device: device for dummy input tensor.
        model_type: "local" or "api".
        notes: free-text notes (e.g. "ViT-L/14, FP16 inference").

    Returns:
        EfficiencyMetadata with params_m and flops_g filled where possible.
    """
    if model_type == "api":
        return EfficiencyMetadata(model_type="api", notes=notes)

    params_m = count_parameters(model)
    flops_g = count_flops_torch(model, input_shape, device=device)

    return EfficiencyMetadata(
        params_m=params_m,
        flops_g=flops_g,
        model_type=model_type,
        notes=notes,
    )

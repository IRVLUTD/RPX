"""Unified model profiler for all RPX tasks.

One class, any model, all metrics. No per-task boilerplate.

Usage
-----
    from rpx_benchmark.model_profiler import ModelProfiler

    profiler = ModelProfiler(model)
    report = profiler.profile()

    # Hardware-agnostic (same on any machine)
    print(report.params_m)              # 524.4
    print(report.flops_g)               # 187.3

    # Measured (hardware-specific, tagged with system card)
    print(report.latency_p50_ms)        # 142.7
    print(report.memory_peak_mb)        # 3200.0

    # Deployability verdicts (novel — directly answers "can I use this?")
    print(report.realtime_at_hz(10))    # False (needs 142ms, budget is 100ms)
    print(report.realtime_at_hz(5))     # True  (needs 142ms, budget is 200ms)
    print(report.budget_fraction(10))   # 1.42  (exceeds 10Hz budget by 42%)

    # Get the EfficiencyMetadata for the runner
    eff = report.to_efficiency_metadata()

Design
------
The profiler handles two distinct phases:

**Pre-run** (called once, before inference):
    Params, FLOPs, memory traffic estimate, system card.
    These are model-intrinsic or estimated — no data needed.

**Post-run** (filled by the runner during inference):
    Latency percentiles, peak memory, FLOP counter refinement.
    The profiler returns an :class:`EfficiencyMetadata` that the runner
    mutates in-place during ``run_with_deployment_readiness``.

This split means the profiler works for any task — depth, pose, segmentation,
detection — without knowing what task it is. The task script just does::

    profiler = ModelProfiler(model_or_adapter)
    eff = profiler.pre_run_profile()
    # ... pass eff to runner, runner fills latency/memory ...
    report = profiler.full_report(eff)

Novel metric: Real-Time Budget Analysis
----------------------------------------
For robotics deployment, the key question isn't "what's the latency?" but
"does it fit in my control loop?"  RPX reports a **budget fraction**:

.. math::

    \\beta(f) = \\frac{\\tau_{\\text{model}}}{1/f}

where :math:`f` is the target control frequency (Hz) and
:math:`\\tau_{\\text{model}}` is measured inference latency.

:math:`\\beta < 1` means the model fits within the real-time budget.
:math:`\\beta > 1` means it exceeds it, and by how much.

Reported at standard robot control frequencies: 5 Hz (slow manipulation),
10 Hz (fast manipulation), 30 Hz (navigation).  No existing perception
benchmark reports this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .profiler import (
    EfficiencyMetadata,
    GPUSpec,
    LatencyProfiler,
    MemoryProfiler,
    REFERENCE_GPUS,
    RooflineBound,
    SystemCard,
    count_parameters,
    estimate_memory_traffic_gb,
)
from .logging_utils import get_logger

log = get_logger(__name__)

# Standard robot control frequencies for budget analysis.
ROBOT_CONTROL_HZ: Tuple[float, ...] = (5.0, 10.0, 30.0)
"""Slow manipulation (5 Hz), fast manipulation (10 Hz), navigation (30 Hz)."""


def _find_torch_module(model: Any) -> Any:
    """Walk common attribute paths to find an ``nn.Module`` with ``.parameters()``.

    Works for BenchmarkableModel, raw adapters, and torch modules directly.
    Returns ``None`` if no torch module is found.
    """
    # If it already has .parameters(), it IS a torch module
    if hasattr(model, "parameters") and callable(model.parameters):
        return model

    # Walk common adapter layouts
    for path in (
        "model",
        "torch_module",
        "_model",
        "_pipe.model",
        "adapter.model",
        "adapter._model",
        "adapter.torch_module",
    ):
        cur = model
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None and hasattr(cur, "parameters") and callable(cur.parameters):
            return cur

    return None


@dataclass
class ProfileReport:
    """Complete profiling result for a model.

    Tier 1 and budget analysis are the primary outputs.
    Tier 3 fields are filled after the runner completes.
    """

    # ── Tier 1: Hardware-agnostic ────────────────────────────────────────
    params_m: Optional[float] = None
    flops_g: Optional[float] = None
    macs_g: Optional[float] = None
    memory_traffic_gb: Optional[float] = None
    arithmetic_intensity: Optional[float] = None

    # ── Tier 2: Roofline (hardware-parametric) ───────────────────────────
    roofline: Optional[Dict[str, Dict[str, float]]] = None

    # ── Tier 3: Measured (hardware-specific) ─────────────────────────────
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    peak_cpu_mb: Optional[float] = None
    peak_cuda_mb: Optional[float] = None
    peak_mps_mb: Optional[float] = None
    system_card: Optional[Dict[str, Any]] = None

    # ── Budget analysis (novel) ──────────────────────────────────────────
    budget_fractions: Optional[Dict[str, float]] = None
    """Maps ``"5hz"`` → β, ``"10hz"`` → β, ``"30hz"`` → β."""

    def budget_fraction(self, hz: float) -> Optional[float]:
        """Compute β = latency / (1000/hz).  < 1 means real-time feasible."""
        if self.latency_p50_ms is None or hz <= 0:
            return None
        budget_ms = 1000.0 / hz
        return round(self.latency_p50_ms / budget_ms, 3)

    def realtime_at_hz(self, hz: float) -> Optional[bool]:
        """Can this model sustain ``hz`` frames per second?"""
        beta = self.budget_fraction(hz)
        if beta is None:
            return None
        return beta <= 1.0

    def compute_budget_fractions(self) -> Dict[str, float]:
        """Compute budget fractions at standard robot control frequencies."""
        out: Dict[str, float] = {}
        for hz in ROBOT_CONTROL_HZ:
            beta = self.budget_fraction(hz)
            if beta is not None:
                out[f"{hz:.0f}hz"] = beta
        self.budget_fractions = out
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Structured output for JSON serialization."""
        return {
            "tier1_hardware_agnostic": {
                "params_m": self.params_m,
                "flops_g": self.flops_g,
                "macs_g": self.macs_g,
                "memory_traffic_gb": self.memory_traffic_gb,
                "arithmetic_intensity": self.arithmetic_intensity,
            },
            "tier2_roofline": self.roofline,
            "tier3_measured": {
                "latency_p50_ms": self.latency_p50_ms,
                "latency_p95_ms": self.latency_p95_ms,
                "latency_p99_ms": self.latency_p99_ms,
                "peak_cpu_mb": self.peak_cpu_mb,
                "peak_cuda_mb": self.peak_cuda_mb,
                "peak_mps_mb": self.peak_mps_mb,
                "system_card": self.system_card,
            },
            "realtime_budget": self.budget_fractions,
        }

    def summary_line(self) -> str:
        """One-line summary for terminal output."""
        parts = []
        if self.params_m is not None:
            parts.append(f"{self.params_m:.1f}M params")
        if self.flops_g is not None:
            parts.append(f"{self.flops_g:.1f}G FLOPs")
        if self.latency_p50_ms is not None:
            parts.append(f"{self.latency_p50_ms:.1f}ms p50")
        if self.budget_fractions:
            for hz_key, beta in self.budget_fractions.items():
                symbol = "✓" if beta <= 1.0 else "✗"
                parts.append(f"{symbol}@{hz_key}")
        return " | ".join(parts) if parts else "no profile data"


class ModelProfiler:
    """Unified profiler for any RPX model.  Works across all tasks.

    Parameters
    ----------
    model : Any
        A BenchmarkableModel, adapter, torch nn.Module, or callable.
        The profiler auto-discovers the torch module for Tier 1 metrics.
    input_shape : tuple
        ``(C, H, W)`` for FLOPs estimation.  Default: RPX standard 640×480 RGB.
    device : str
        Device for profiling.  Default ``"cuda"``.

    Example
    -------
    ::

        profiler = ModelProfiler(my_adapter)

        # Pre-run: get Tier 1 metrics + EfficiencyMetadata for the runner
        eff = profiler.pre_run_profile()

        # ... runner.run_with_deployment_readiness(efficiency=eff) ...
        # (runner fills latency, memory, FLOPs into eff)

        # Post-run: build the full report from the now-populated eff
        report = profiler.full_report(eff)
        print(report.summary_line())
        # → "524.4M params | 187.3G FLOPs | 142.7ms p50 | ✗@10hz | ✓@5hz"
    """

    def __init__(
        self,
        model: Any,
        input_shape: Tuple[int, ...] = (3, 480, 640),
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self._torch_mod = _find_torch_module(model)

    @property
    def has_torch_module(self) -> bool:
        return self._torch_mod is not None

    def pre_run_profile(self) -> EfficiencyMetadata:
        """Compute Tier 1 metrics and return an EfficiencyMetadata for the runner.

        The returned object is passed to
        ``runner.run_with_deployment_readiness(efficiency=eff)``.
        The runner will fill in Tier 3 fields (latency, memory, FLOPs)
        during the benchmark run.
        """
        eff = EfficiencyMetadata(
            model_type="local" if self._torch_mod is not None else "classical",
        )

        if self._torch_mod is not None:
            try:
                eff.params_m = count_parameters(self._torch_mod)
            except Exception as e:
                log.warning("params count failed: %s", e)

            try:
                eff.memory_traffic_gb = estimate_memory_traffic_gb(
                    self._torch_mod, self.input_shape, device=self.device,
                )
            except Exception as e:
                log.warning("memory traffic estimate failed: %s", e)

        try:
            eff.system_card = SystemCard.auto_detect(
                input_resolution=f"{self.input_shape[-1]}x{self.input_shape[-2]}",
            )
        except Exception as e:
            log.warning("system card detection failed: %s", e)

        return eff

    def full_report(self, eff: EfficiencyMetadata) -> ProfileReport:
        """Build a complete ProfileReport from a (now runner-populated) EfficiencyMetadata.

        Call this AFTER the runner has completed — ``eff`` will have
        latency, memory, and FLOPs filled in by the runner.
        """
        # Ensure derived fields are populated
        eff.derive_tier1()
        if eff.flops_g is not None and eff.roofline is None:
            eff.compute_roofline()

        # Build roofline dict for serialization
        roofline_dict = None
        if eff.roofline:
            roofline_dict = {
                name: {
                    "compute_ms": b.compute_ms,
                    "memory_ms": b.memory_ms,
                    "latency_ms": b.latency_ms,
                    "bottleneck": b.bottleneck,
                }
                for name, b in eff.roofline.items()
            }

        report = ProfileReport(
            # Tier 1
            params_m=eff.params_m,
            flops_g=eff.flops_g,
            macs_g=eff.macs_g,
            memory_traffic_gb=eff.memory_traffic_gb,
            arithmetic_intensity=eff.arithmetic_intensity,
            # Tier 2
            roofline=roofline_dict,
            # Tier 3
            latency_p50_ms=eff.latency_p50_ms,
            latency_p95_ms=eff.latency_p95_ms,
            latency_p99_ms=eff.latency_p99_ms,
            peak_cpu_mb=eff.peak_cpu_mb,
            peak_cuda_mb=eff.peak_cuda_mb,
            peak_mps_mb=eff.peak_mps_mb,
            system_card=eff.system_card.to_dict() if eff.system_card else None,
        )

        # Compute real-time budget analysis
        report.compute_budget_fractions()

        return report

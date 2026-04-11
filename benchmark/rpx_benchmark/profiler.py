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

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class EfficiencyMetadata:
    """Hardware-agnostic efficiency metadata for a model."""

    params_m: float | None = None            # Total parameters in millions
    flops_g: float | None = None             # FLOPs (giga) at batch=1, FP32
    actmem_gb_fp16: float | None = None      # Optional activation memory at FP16 (GB)
    latency_ms_per_sample: float | None = None  # Filled post-run by the task runner
    model_type: str = "local"                # "local" | "api"
    notes: str = ""                          # e.g. precision mode, resolution override

    def to_table_row(self) -> dict:
        """Produce result-table-ready dict (None → 'N/A (API)' for API models)."""
        na = "N/A (API)" if self.model_type == "api" else None
        return {
            "type": self.model_type,
            "params_m": self.params_m if self.params_m is not None else na,
            "flops_g": self.flops_g if self.flops_g is not None else na,
            "actmem_gb_fp16": self.actmem_gb_fp16,
            "latency_ms_per_sample": self.latency_ms_per_sample,
        }


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

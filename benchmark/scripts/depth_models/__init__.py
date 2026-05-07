"""Reference depth-model adapters for the RPX benchmark.

Each adapter is a *callable* class with the contract:

* single-image: ``adapter(rgb: np.ndarray (H, W, 3) uint8) -> np.ndarray (H', W') float32`` (metres)
* batched:     ``adapter(list[np.ndarray]) -> list[np.ndarray]``

Plus three optional class-level attributes the runner reads:

* ``native_alignment: str`` — one of ``"none"`` (metric), ``"median"``,
  ``"ls_affine"``, ``"ls_disparity"``. The runner picks alignment by
  default; ``--alignment`` flag still overrides per-run.
* ``torch_module`` — exposed so the profiler walker can count parameters /
  estimate memory traffic. Optional but recommended.
* ``DEFAULT_MODEL_ID: str`` — the canonical HF / github checkpoint id.

The :data:`MODEL_REGISTRY` below maps short names (``--model X``) to a
zero-arg builder that returns the constructed adapter. Adding a new
model is one line.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


# Builder: ``builder(device: str, batch_size: int, **kwargs) -> adapter``.
ModelBuilder = Callable[..., Any]


def _build_zoedepth(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .zoedepth import ZoeDepth
    return ZoeDepth(device=device, batch_size=batch_size, **kwargs)


def _build_depth_pro(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .depth_pro import DepthPro
    return DepthPro(device=device, batch_size=batch_size, **kwargs)


def _build_da_v2_indoor(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .depth_anything_v2 import DepthAnythingV2Metric
    # HF-transformers-format checkpoint (the non-'-hf' variants are
    # PyTorch state-dicts that require the depth_anything_v2 python
    # package; the '-hf' variants ship the config.json the auto-pipeline
    # needs). 'Indoor' is the Hypersim-trained head; 'Outdoor' is VKITTI.
    return DepthAnythingV2Metric(
        model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        device=device, batch_size=batch_size, **kwargs,
    )


def _build_da_v2_outdoor(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .depth_anything_v2 import DepthAnythingV2Metric
    return DepthAnythingV2Metric(
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        device=device, batch_size=batch_size, **kwargs,
    )


def _build_unidepth_v2(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .unidepth_v2 import UniDepthV2
    return UniDepthV2(device=device, batch_size=batch_size, **kwargs)


# ── Turn B: relative-depth HF pipeline adapters (all share HFDepthEstimationAdapter) ──

def _build_da_v2_relative(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Depth Anything V2, *relative* head (up-to-scale; needs alignment)."""
    from .hf_pipeline import HFDepthEstimationAdapter
    return HFDepthEstimationAdapter(
        model_id="depth-anything/Depth-Anything-V2-Large-hf",
        device=device, batch_size=batch_size,
        native_alignment="ls_affine", **kwargs,
    )


def _build_da_v1(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Depth Anything V1 — predecessor baseline. Relative depth."""
    from .hf_pipeline import HFDepthEstimationAdapter
    return HFDepthEstimationAdapter(
        model_id="LiheYoung/depth-anything-large-hf",
        device=device, batch_size=batch_size,
        native_alignment="ls_affine", **kwargs,
    )


def _build_midas_v31(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """MiDaS v3.1 (DPT-BEiT-L) — classic relative-depth baseline."""
    from .hf_pipeline import HFDepthEstimationAdapter
    return HFDepthEstimationAdapter(
        model_id="Intel/dpt-beit-large-384",
        device=device, batch_size=batch_size,
        native_alignment="ls_affine", **kwargs,
    )


def _build_distill_any_depth(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Distill-Any-Depth — multi-teacher distillation, relative depth."""
    from .hf_pipeline import HFDepthEstimationAdapter
    return HFDepthEstimationAdapter(
        model_id="xingyang1/Distill-Any-Depth-Large-hf",
        device=device, batch_size=batch_size,
        native_alignment="ls_affine", **kwargs,
    )


#: Public name → builder. ``--model X`` resolves through this table.
#: Each entry is the canonical short id (snake_case) the team uses in
#: result.json paths and the Box upload tree, so the same key shows up
#: across the codebase, the brief PDF tables, and the dashboard.
MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    # ── Metric (native_alignment="none") ─────────────────────────────────
    "zoedepth":              _build_zoedepth,
    "depth_pro":             _build_depth_pro,
    "da_v2_metric_indoor":   _build_da_v2_indoor,
    "da_v2_metric_outdoor":  _build_da_v2_outdoor,
    "unidepth_v2":           _build_unidepth_v2,
    # ── Relative / aligned (native_alignment="ls_affine") ────────────────
    "da_v2_relative":        _build_da_v2_relative,
    "da_v1":                 _build_da_v1,
    "midas_v31":             _build_midas_v31,
    "distill_any_depth":     _build_distill_any_depth,
    # ── Pending ─────────────────────────────────────────────────────────
    # Turn C (diffusion): marigold, marigold_lcm, lotus_2, geowizard
    # Turn D (Microsoft): moge_2, moge_v1
    # Turn E (specialised HF): patchfusion, hyden_metric, hyden_relative
    # Turn F (github):  metric3d_v2, metricsolver
}


#: Display names for tables and reports — preserves capitalisation /
#: punctuation that the snake_case keys lose.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "zoedepth":              "ZoeDepth_NK",
    "depth_pro":             "DepthPro",
    "da_v2_metric_indoor":   "DA-V2-Metric-Indoor-L",
    "da_v2_metric_outdoor":  "DA-V2-Metric-Outdoor-L",
    "unidepth_v2":           "UniDepth-V2-ViTL14",
    "da_v2_relative":        "DA-V2-Relative-L",
    "da_v1":                 "DA-V1-Large",
    "midas_v31":             "MiDaS-v3.1-DPT-BEiT-L",
    "distill_any_depth":     "Distill-Any-Depth-L",
}


def list_models() -> list[str]:
    """Return the sorted list of registered model keys."""
    return sorted(MODEL_REGISTRY.keys())


__all__ = [
    "MODEL_REGISTRY",
    "MODEL_DISPLAY_NAMES",
    "list_models",
]

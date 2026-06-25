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
        device=device,
        batch_size=batch_size,
        **kwargs,
    )


def _build_da_v2_outdoor(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    from .depth_anything_v2 import DepthAnythingV2Metric

    return DepthAnythingV2Metric(
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        device=device,
        batch_size=batch_size,
        **kwargs,
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
        device=device,
        batch_size=batch_size,
        native_alignment="ls_affine",
        **kwargs,
    )


def _build_da_v1(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Depth Anything V1 — predecessor baseline. Relative depth."""
    from .hf_pipeline import HFDepthEstimationAdapter

    return HFDepthEstimationAdapter(
        model_id="LiheYoung/depth-anything-large-hf",
        device=device,
        batch_size=batch_size,
        native_alignment="ls_affine",
        **kwargs,
    )


def _build_midas_v31(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """MiDaS v3.1 (DPT-BEiT-L) — classic relative-depth baseline."""
    from .hf_pipeline import HFDepthEstimationAdapter

    return HFDepthEstimationAdapter(
        model_id="Intel/dpt-beit-large-384",
        device=device,
        batch_size=batch_size,
        native_alignment="ls_affine",
        **kwargs,
    )


def _build_distill_any_depth(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Distill-Any-Depth — multi-teacher distillation, relative depth."""
    from .hf_pipeline import HFDepthEstimationAdapter

    return HFDepthEstimationAdapter(
        model_id="xingyang1/Distill-Any-Depth-Large-hf",
        device=device,
        batch_size=batch_size,
        native_alignment="ls_affine",
        **kwargs,
    )


# ── Diffusion adapters (relative; ls_affine alignment in the runner) ──


def _build_marigold(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Marigold v1.1 — diffusion. Default ensemble_size=1 (raw). Override
    with ensemble_size=10 for the published-best supplementary number."""
    from .marigold import Marigold

    return Marigold(
        model_id="prs-eth/marigold-depth-v1-1",
        device=device,
        batch_size=batch_size,
        ensemble_size=1,
        **kwargs,
    )


def _build_marigold_lcm(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Marigold-LCM — latent-consistency-distilled, single-step inference."""
    from .marigold import Marigold

    return Marigold(
        model_id="prs-eth/marigold-depth-lcm-v1-0",
        device=device,
        batch_size=batch_size,
        ensemble_size=1,
        **kwargs,
    )


def _build_lotus_2(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Lotus-2 — single-step diffusion-prior depth.

    The earlier draft pinned to ``jingheya/Lotus-2`` which doesn't
    exist on HF (404). The live distribution is the v2-0-disparity
    direct model; ``jingheya/lotus-depth-g-v2-1-disparity`` is the
    generative variant — pass ``model_id=...`` to override.
    """
    from .lotus import Lotus

    return Lotus(
        model_id="jingheya/lotus-depth-d-v2-0-disparity",
        device=device,
        batch_size=batch_size,
        **kwargs,
    )


def _build_geowizard(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """GeoWizard — diffusion, joint depth + normal (we consume depth only)."""
    from .geowizard import GeoWizard

    return GeoWizard(
        model_id="lemonaddie/geowizard", device=device, batch_size=batch_size, **kwargs
    )


# ── Microsoft pair (MoGe-2 metric, MoGe v1 affine-invariant) ──


def _build_moge_2(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """MoGe-2 — geometry-aware metric depth + normal."""
    from .moge import MoGe

    return MoGe(
        model_id="Ruicheng/moge-2-vitl-normal",
        device=device,
        batch_size=batch_size,
        native_alignment="none",  # metric
        **kwargs,
    )


def _build_moge_v1(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """MoGe v1 — predecessor; affine-invariant depth."""
    from .moge import MoGe

    return MoGe(
        model_id="Ruicheng/moge-vitl",
        device=device,
        batch_size=batch_size,
        native_alignment="ls_affine",  # affine-invariant
        **kwargs,
    )


# ── Specialised HF adapters ──


def _build_patchfusion(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """PatchFusion ZoeDepth — tile-based hi-res metric (CVPR'24)."""
    from .patchfusion import PatchFusion

    return PatchFusion(
        model_id="zhyever/patchfusion_zoedepth", device=device, batch_size=batch_size, **kwargs
    )


def _build_hyden_metric(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """HyDen — Meta's metric-depth head (ICLR'26)."""
    from .hyden import HyDen

    return HyDen(
        model_id="facebook/hyden-da2-metric-depth",
        device=device,
        batch_size=batch_size,
        native_alignment="none",
        **kwargs,
    )


def _build_hyden_relative(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """HyDen — Meta's relative-depth head (ICLR'26)."""
    from .hyden import HyDen

    return HyDen(
        model_id="facebook/hyden-da2-relative-depth",
        device=device,
        batch_size=batch_size,
        native_alignment="ls_affine",
        **kwargs,
    )


# ── github-vendored ──


def _build_metric3d_v2(*, device: str = "cuda", batch_size: int = 1, **kwargs):
    """Metric3D V2 — universal metric depth via torch.hub."""
    from .metric3d_v2 import Metric3DV2

    return Metric3DV2(device=device, batch_size=batch_size, **kwargs)


#: Public name → builder. ``--model X`` resolves through this table.
#: Each entry is the canonical short id (snake_case) the team uses in
#: result.json paths and the Box upload tree, so the same key shows up
#: across the codebase, the brief PDF tables, and the dashboard.
MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    # ── Metric (native_alignment="none") ─────────────────────────────────
    "zoedepth": _build_zoedepth,
    "depth_pro": _build_depth_pro,
    "da_v2_metric_indoor": _build_da_v2_indoor,
    "da_v2_metric_outdoor": _build_da_v2_outdoor,
    "unidepth_v2": _build_unidepth_v2,
    "moge_2": _build_moge_2,
    "patchfusion": _build_patchfusion,
    "hyden_metric": _build_hyden_metric,
    "metric3d_v2": _build_metric3d_v2,
    # ── Relative / aligned (native_alignment="ls_affine") ────────────────
    "da_v2_relative": _build_da_v2_relative,
    "da_v1": _build_da_v1,
    "midas_v31": _build_midas_v31,
    "distill_any_depth": _build_distill_any_depth,
    "marigold": _build_marigold,
    "marigold_lcm": _build_marigold_lcm,
    "lotus_2": _build_lotus_2,
    "geowizard": _build_geowizard,
    "moge_v1": _build_moge_v1,
    "hyden_relative": _build_hyden_relative,
    # ── Dropped (no clean release as of May 2026) ────────────────────────
    # MetricSolver — kept off the table until a maintained checkpoint
    # surfaces; revisit in a future minor release.
}


#: Display names for tables and reports — preserves capitalisation /
#: punctuation that the snake_case keys lose.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    # Metric
    "zoedepth": "ZoeDepth_NK",
    "depth_pro": "DepthPro",
    "da_v2_metric_indoor": "DA-V2-Metric-Indoor-L",
    "da_v2_metric_outdoor": "DA-V2-Metric-Outdoor-L",
    "unidepth_v2": "UniDepth-V2-ViTL14",
    "moge_2": "MoGe-2-ViTL",
    "patchfusion": "PatchFusion-ZoeDepth",
    "hyden_metric": "HyDen-DA2-Metric",
    "metric3d_v2": "Metric3D-V2-ViT-Giant",
    # Relative
    "da_v2_relative": "DA-V2-Relative-L",
    "da_v1": "DA-V1-Large",
    "midas_v31": "MiDaS-v3.1-DPT-BEiT-L",
    "distill_any_depth": "Distill-Any-Depth-L",
    "marigold": "Marigold-v1.1",
    "marigold_lcm": "Marigold-LCM",
    "lotus_2": "Lotus-2",
    "geowizard": "GeoWizard",
    "moge_v1": "MoGe-v1-ViTL",
    "hyden_relative": "HyDen-DA2-Relative",
}


def list_models() -> list[str]:
    """Return the sorted list of registered model keys."""
    return sorted(MODEL_REGISTRY.keys())


# --------------------------------------------------------------------------- #
# Canonical roster bridge
# --------------------------------------------------------------------------- #
# The paper's canonical roster lives at
# ``rpx_benchmark.adapters.depth_scaffold.DEPTH_MODEL_CARDS`` and uses
# kebab-case keys (``da-v2-large``, ``depth-pro``, ...). This module's
# ``MODEL_REGISTRY`` predates that roster and uses snake_case keys
# (``da_v2_metric_indoor``, ``depth_pro``, ...). The bridge below maps
# canonical → registry so the team can call adapters by their
# paper-table names via ``scripts/run_depth.py --model da-v2-large``.
#
# Each canonical key maps to ONE registry adapter; multi-variant models
# (DA-V2 indoor/outdoor) default to the indoor head since the bulk of
# RPX scenes are indoor — pass ``--head outdoor`` to override
# (handled at the CLI layer for the few adapters that support it).
#
# Three canonical entries have no registry adapter today
# (DA3 Metric-L, FE2E, DepthLM): they map to ``None`` so a clean
# resolution error is raised in lieu of a confusing KeyError.
CANONICAL_TO_LEGACY: Dict[str, str | None] = {
    # Working
    "da-v2-large":   "da_v2_metric_indoor",
    "depth-pro":     "depth_pro",
    "unidepth-v2":   "unidepth_v2",
    "metric3d-v2":   "metric3d_v2",
    "moge-2-vit-l":  "moge_2",
    "hyden":         "hyden_metric",
    "lotus-2":       "lotus_2",
    # No upstream implementation today — install when the team has one
    "da3-metric-l":  None,
    "fe2e":          None,
    "depthlm":       None,
}


def resolve_model_key(name: str) -> str:
    """Resolve a CLI ``--model`` value to a registry key.

    Accepts both naming conventions:

    * **Canonical roster** kebab-case keys
      (:data:`~rpx_benchmark.adapters.depth_scaffold.DEPTH_MODEL_CARDS`):
      ``da-v2-large``, ``depth-pro``, etc. Routed via
      :data:`CANONICAL_TO_LEGACY`.

    * **Legacy registry** snake_case keys: ``da_v2_metric_indoor``,
      ``depth_pro``, etc. Passed through unchanged.

    Raises
    ------
    SystemExit
        Clean error if the name resolves to no known adapter, with the
        full list of valid canonical + legacy names.
    """
    if name in MODEL_REGISTRY:
        return name
    if name in CANONICAL_TO_LEGACY:
        legacy = CANONICAL_TO_LEGACY[name]
        if legacy is None:
            from rpx_benchmark.adapters.depth_scaffold import DEPTH_MODEL_CARDS

            card = DEPTH_MODEL_CARDS.get(name)
            hint = (
                f"Install hint: {card.install_hint}" if card else
                "No upstream implementation on this branch."
            )
            raise SystemExit(
                f"--model {name!r} is in the canonical roster but has no "
                f"adapter under scripts/depth_models/ yet. {hint}"
            )
        return legacy
    canonical = sorted(CANONICAL_TO_LEGACY)
    legacy = sorted(MODEL_REGISTRY)
    raise SystemExit(
        f"unknown --model {name!r}. "
        f"Canonical roster names: {canonical}. "
        f"Legacy registry names: {legacy}."
    )


__all__ = [
    "MODEL_REGISTRY",
    "MODEL_DISPLAY_NAMES",
    "CANONICAL_TO_LEGACY",
    "list_models",
    "resolve_model_key",
]

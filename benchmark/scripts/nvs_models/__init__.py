"""Reference NVS-model adapters for the RPX benchmark — NVS axis.

Each adapter is a callable class with the contract:

* ``adapter(context_rgbs, context_depths, context_poses, target_pose) -> dict``
  - ``context_rgbs``: ``list[np.ndarray (H, W, 3) uint8]``
  - ``context_depths``: ``list[np.ndarray (H, W) float32]`` (metres; may be empty)
  - ``context_poses``: ``list[np.ndarray (4, 4) float64]`` (camera-to-world)
  - ``target_pose``: ``np.ndarray (4, 4) float64`` (camera-to-world, query view)
  - returns: ``{"rgb": (H, W, 3) uint8, "depth": (H, W) float32 | None}``

Plus optional class-level attributes the runner reads:

* ``native_precision: str`` — fp32 / fp16 / bf16. Drives the OperatingPoint precision tag.
* ``torch_module`` — exposed so the profiler walker can count parameters.

The :data:`MODEL_REGISTRY` below maps short names (``--model X``) to a
zero-arg builder. Adding a new model is one line.

Until each upstream is wired up the builders raise a clear
``AdapterError`` with the install hint and the upstream URL — the
registry loads cleanly on a bare environment so ``--help`` and
``identity_passthrough`` keep working.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

ModelBuilder = Callable[..., Any]


# ── Reference baselines ──────────────────────────────────────────────────────


def _build_identity_passthrough(*, device: str = "cpu", **kwargs: Any) -> Any:
    """Trivial baseline: return the context view with the closest pose to target."""
    from .identity_passthrough import IdentityPassthroughNVS

    return IdentityPassthroughNVS(device=device, **kwargs)


def _build_splatter_image(*, device: str = "cuda", **kwargs: Any) -> Any:
    """Splatter Image (CVPR 2024) — single-view feed-forward 3DGS.

    The adapter scaffolds the real upstream wiring (GaussianSplatPredictor +
    render_predicted from the szymanowiczs/splatter-image repo) and raises
    a clear AdapterError if the upstream isn't on PYTHONPATH. End-to-end
    smoke against real RPX scenes is **pending** an upstream `pip install
    -e .` and is tracked in `benchmark/SHARED_CONTEXT.md`.
    """
    from .splatter_image import SplatterImage

    return SplatterImage(device=device, **kwargs)


def _build_depthsplat(*, device: str = "cuda", **kwargs: Any) -> Any:
    """DepthSplat (CVPR 2025), released two-view 256x256 checkpoint."""
    from .depthsplat import DepthSplatNVS

    return DepthSplatNVS(device=device, **kwargs)


# ── Feed-forward 3DGS / pointmap models ──────────────────────────────────────


def _not_yet_wired(name: str, upstream: str, install_hint: str) -> Callable[..., Any]:
    """Builder factory for adapters whose upstream isn't installed yet.

    Raises ``AdapterError`` with the install hint when the user invokes
    the adapter, but importing this module stays clean — so ``--help``,
    ``list_models()``, and ``identity_passthrough`` continue to work on
    a bare environment.
    """

    def _builder(**_kwargs: Any) -> Any:
        from rpx_benchmark.exceptions import AdapterError  # noqa: PLC0415

        raise AdapterError(
            f"{name} adapter is not yet wired into RPX.",
            hint=f"Upstream: {upstream}. {install_hint}",
        )

    return _builder


# ── 10-model NVS slate from the parallel-session paper draft ─────────────────
#
#   DepthSplat · MVSplat · pixelSplat · NoPoSplat · Splatt3R · AnySplat ·
#   PF3plat · Flash3D · Splatter Image · FLARE
#
# Each pending until its upstream is integrated; the AdapterError carries the
# repo URL + install hint so a contributor knows exactly what to clone.

_PENDING_MODELS: Dict[str, Dict[str, str]] = {
    "mvsplat": {
        "upstream": "https://github.com/donydchen/mvsplat",
        "hint":     "clone the repo + download the re10k-trained checkpoint.",
    },
    "pixelsplat": {
        "upstream": "https://github.com/dcharatan/pixelsplat",
        "hint":     "clone the repo and follow its conda setup.",
    },
    "nopo_splat": {
        "upstream": "https://github.com/cvg/NoPoSplat",
        "hint":     "clone the repo; needs torch + torchvision matching their pin.",
    },
    "splatt3r": {
        "upstream": "https://github.com/btsmart/splatt3r",
        "hint":     "clone the repo + pull weights via huggingface_hub.",
    },
    "anysplat": {
        "upstream": "https://github.com/OpenRobotLab/AnySplat",
        "hint":     "clone the repo, `pip install -e .`, fetch their checkpoint.",
    },
    "pf3plat": {
        "upstream": "https://github.com/cvg/PF3plat",
        "hint":     "clone the repo and run `pip install -e .` from a fresh env.",
    },
    "flash3d": {
        "upstream": "https://github.com/eldar/flash3d",
        "hint":     "clone + checkpoint via huggingface_hub.",
    },
    "flare": {
        "upstream": "https://github.com/ant-research/FLARE",
        "hint":     "clone the repo; needs pytorch3d and the FLARE checkpoint.",
    },
}


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "identity_passthrough": _build_identity_passthrough,
    "depthsplat":           _build_depthsplat,
    "splatter_image":       _build_splatter_image,
    **{
        name: _not_yet_wired(name, info["upstream"], info["hint"])
        for name, info in _PENDING_MODELS.items()
    },
}


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "identity_passthrough": "IdentityPassthrough",
    "depthsplat":           "DepthSplat",
    "mvsplat":              "MVSplat",
    "pixelsplat":           "pixelSplat",
    "nopo_splat":           "NoPoSplat",
    "splatt3r":             "Splatt3R",
    "anysplat":             "AnySplat",
    "pf3plat":              "PF3plat",
    "flash3d":              "Flash3D",
    "splatter_image":       "SplatterImage",
    "flare":                "FLARE",
}


def list_models() -> List[str]:
    """Return all registered model keys, sorted alphabetically."""
    return sorted(MODEL_REGISTRY.keys())


__all__ = ["MODEL_REGISTRY", "MODEL_DISPLAY_NAMES", "list_models"]

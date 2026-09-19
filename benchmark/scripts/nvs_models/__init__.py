"""NVS adapters and model metadata for the RPX benchmark."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from .external_bridge import build_external
from .specs import MODEL_SPECS, PAPER_MODEL_KEYS, NVSModelSpec, models_by_track

ModelBuilder = Callable[..., Any]


def _build_identity_passthrough(*, device: str = "cpu", **kwargs: Any) -> Any:
    from .identity_passthrough import IdentityPassthroughNVS

    return IdentityPassthroughNVS(device=device, **kwargs)


def _build_depthsplat(*, device: str = "cuda", **kwargs: Any) -> Any:
    from .depthsplat import DepthSplatNVS

    return DepthSplatNVS(device=device, **kwargs)


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    key: (_build_depthsplat if key == "depthsplat" else partial(build_external, spec))
    for key, spec in MODEL_SPECS.items()
}
MODEL_REGISTRY["identity_passthrough"] = _build_identity_passthrough

MODEL_DISPLAY_NAMES: dict[str, str] = {
    key: spec.display_name for key, spec in MODEL_SPECS.items()
}
MODEL_DISPLAY_NAMES["identity_passthrough"] = "IdentityPassthrough"


def list_models(*, include_diagnostics: bool = True) -> list[str]:
    keys = list(PAPER_MODEL_KEYS)
    if include_diagnostics:
        keys.append("identity_passthrough")
    return sorted(keys)


__all__ = [
    "MODEL_DISPLAY_NAMES",
    "MODEL_REGISTRY",
    "MODEL_SPECS",
    "NVSModelSpec",
    "PAPER_MODEL_KEYS",
    "list_models",
    "models_by_track",
]

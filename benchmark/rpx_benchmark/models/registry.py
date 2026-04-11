"""Name → factory registry for RPX model adapters.

A factory is ``(device: str, **kwargs) -> BenchmarkableModel``. The
actual import of the factory module is deferred until
:func:`resolve` is called, so the top-level package can be imported
without torch, transformers, or any model-specific dependency.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Tuple

from ..adapters import BenchmarkableModel
from ..exceptions import ConfigError

# name -> (module suffix under rpx_benchmark.models, factory callable name)
# Organised by family; runnable first, deferred stubs last. Stubs raise
# a clear NotImplementedError so ``rpx models`` shows the intended slate.
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # ----- Monocular absolute depth: HuggingFace transformers family -----
    "depth_anything_v2_metric_indoor_small": (
        "depth_anything_v2", "depth_anything_v2_metric_indoor_small"),
    "depth_anything_v2_metric_indoor_base": (
        "depth_anything_v2", "depth_anything_v2_metric_indoor_base"),
    "depth_anything_v2_metric_indoor_large": (
        "depth_anything_v2", "depth_anything_v2_metric_indoor_large"),
    "depth_pro": (
        "depth_pro", "depth_pro"),
    "zoedepth_nyu": (
        "zoedepth", "zoedepth_nyu"),

    # ----- Monocular absolute depth: native packages -----
    "unidepth_v2_vitb": (
        "unidepth_v2", "unidepth_v2_vitb"),
    "unidepth_v2_vitl": (
        "unidepth_v2", "unidepth_v2_vitl"),
    "metric3d_v2_vit_small": (
        "metric3d_v2", "metric3d_v2_vit_small"),
    "metric3d_v2_vit_large": (
        "metric3d_v2", "metric3d_v2_vit_large"),
    "metric3d_v2_vit_giant2": (
        "metric3d_v2", "metric3d_v2_vit_giant2"),

    # ----- Deferred (registered for visibility; raise on resolve) -----
    "video_depth_anything_large": (
        "_deferred", "video_depth_anything_large"),
    "prompt_depth_anything_vits": (
        "_deferred", "prompt_depth_anything_vits"),
    "depth_anything_3": (
        "_deferred", "depth_anything_3"),
}

# Runnable subset, for CLI choices that should exclude deferred stubs.
DEFERRED_MODELS = frozenset({
    "video_depth_anything_large",
    "prompt_depth_anything_vits",
    "depth_anything_3",
})


def available_models(include_deferred: bool = False) -> List[str]:
    """Return sorted registered model names.

    By default excludes deferred stubs so the CLI's ``--model`` choice
    list is runnable-only. Pass ``include_deferred=True`` to list the
    full intended slate.
    """
    names = sorted(MODEL_REGISTRY.keys())
    if include_deferred:
        return names
    return [n for n in names if n not in DEFERRED_MODELS]


def get_factory(name: str) -> Callable[..., BenchmarkableModel]:
    """Return the factory function registered under ``name`` (lazy import).

    Parameters
    ----------
    name : str
        Registered model name. Use :func:`available_models` to list
        the current slate.

    Returns
    -------
    Callable
        The factory function. Instantiate the model by calling it
        with the appropriate device / kwargs.

    Raises
    ------
    ConfigError
        If ``name`` is not in the registry. The error lists every
        currently registered model so typos are obvious.
    """
    if name not in MODEL_REGISTRY:
        raise ConfigError(
            f"Unknown model {name!r}.",
            hint=(
                "Registered models: "
                + ", ".join(available_models(include_deferred=True))
            ),
        )
    module_suffix, factory_name = MODEL_REGISTRY[name]
    module = importlib.import_module(f"rpx_benchmark.models.{module_suffix}")
    return getattr(module, factory_name)


def resolve(name: str, *, device: str = "cuda", **kwargs) -> BenchmarkableModel:
    """Look up ``name`` and call the factory with ``device`` + extra kwargs."""
    factory = get_factory(name)
    return factory(device=device, **kwargs)


def register(name: str, module_suffix: str, factory_name: str) -> None:
    """Third-party code can register additional model factories at runtime."""
    MODEL_REGISTRY[name] = (module_suffix, factory_name)

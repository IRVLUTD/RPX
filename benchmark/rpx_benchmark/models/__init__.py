"""Registered model factories for RPX benchmarking.

Each factory is a callable ``(device: str, **kwargs) -> BenchmarkableModel``
that composes an input adapter, a model, and an output adapter from
:mod:`rpx_benchmark.adapters`. Users can register their own with
:func:`rpx_benchmark.models.registry.register`.
"""

from .registry import (
    MODEL_REGISTRY,
    available_models,
    get_factory,
    register,
    resolve,
)

__all__ = [
    "MODEL_REGISTRY",
    "available_models",
    "get_factory",
    "register",
    "resolve",
]

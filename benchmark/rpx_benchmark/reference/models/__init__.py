"""Reference model factories registered against the monocular-depth slate.

Each module here is a thin factory over a reference adapter in
:mod:`rpx_benchmark.reference.adapters`. The top-level
:data:`rpx_benchmark.models.registry.MODEL_REGISTRY` maps user-facing
model names to ``(module_suffix, factory_name)`` tuples that are
resolved against this package.
"""

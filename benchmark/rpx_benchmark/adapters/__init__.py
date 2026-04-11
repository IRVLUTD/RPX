"""Formal adapter framework for bringing models into the RPX benchmark.

The contract is intentionally minimal: preprocess the RPX ``Sample`` into
whatever the model wants, invoke the model, postprocess the output into
the RPX prediction contract. Users composing their own adapter stack
only need to touch the *model* -- the input and output adapters shipped
here handle the plumbing for common model families.

See :mod:`rpx_benchmark.adapters.base` for the core types and
:mod:`rpx_benchmark.adapters.depth_hf` for a reference implementation
that works with any HuggingFace ``AutoModelForDepthEstimation``
checkpoint.
"""

from .base import (
    BenchmarkableModel,
    InputAdapter,
    ModelInvoker,
    OutputAdapter,
    PreparedInput,
    default_invoker,
    make_numpy_depth_model,
    make_numpy_mask_model,
)

__all__ = [
    "BenchmarkableModel",
    "InputAdapter",
    "ModelInvoker",
    "OutputAdapter",
    "PreparedInput",
    "default_invoker",
    "make_numpy_depth_model",
    "make_numpy_mask_model",
]

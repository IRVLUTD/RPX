"""Reference adapters and model factories shipped with RPX.

This subpackage holds concrete, opinionated implementations for a
handful of popular perception models (HuggingFace depth/segmentation,
UniDepth V2, Metric3D V2, Depth Anything V2, etc.). They exist so
you have working code to copy from — *not* as a fixed slate that
every RPX user must depend on.

The framework surface (adapter protocols, the ``BenchmarkableModel``
container, the per-task metric/task registries, the profiler, the
loader) lives in the top-level :mod:`rpx_benchmark` namespace. Those
modules never depend on anything in :mod:`rpx_benchmark.reference`.
The separation keeps the "bring your model, we bring the harness"
story crisp:

- **Framework** = `rpx_benchmark.{api,adapters,loader,metrics,tasks,
  runner,profiler,...}` — stable public surface, no deep-learning
  dependencies in the default install.
- **Reference** = `rpx_benchmark.reference.{adapters,models}` —
  depends on torch / transformers / backend-specific libraries;
  pulled in via extras (``[depth-hf]``, ``[depth-unidepth]``,
  ``[depth-metric3d]``).

Existing imports such as
``from rpx_benchmark.adapters.depth_hf import make_hf_depth_model``
continue to work via shim modules that re-export from here and emit a
``DeprecationWarning``. Update your code to the new path:
``from rpx_benchmark.reference.adapters.depth_hf import make_hf_depth_model``.
"""

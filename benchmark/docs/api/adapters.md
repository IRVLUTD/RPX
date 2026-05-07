# Adapters (`rpx_benchmark.adapters`)

The formal adapter framework: `InputAdapter` / `OutputAdapter`
protocols, `PreparedInput`, `BenchmarkableModel`, and the
true-batched-dispatch wrappers.

## Base framework

::: rpx_benchmark.adapters.base
    options:
      show_root_toc_entry: false

## True-batched-dispatch base + reference subclasses

The toolkit ships a generic batched-dispatch `BenchmarkModel` so any
task can opt into single-call GPU batching without reimplementing the
loop, save logic, and prediction-wrapping ceremony.

::: rpx_benchmark.adapters.batched_depth
    options:
      show_root_toc_entry: false

::: rpx_benchmark.adapters.batched_multimodal
    options:
      show_root_toc_entry: false

## Reference depth adapters

Reference implementations of nine monocular-depth adapters live in
`scripts/depth_models/` (alongside the run scripts), not under
`rpx_benchmark.adapters`. See
[`scripts/depth_models/README.md`](https://github.com/IRVLUTD/RPX/tree/main/benchmark/scripts/depth_models)
for the full list and the per-model registry.

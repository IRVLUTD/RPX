# Profiler (`rpx_benchmark.profiler`)

Hardware-agnostic efficiency metadata: parameter count, FLOPs, and
optional activation-memory accounting.

The runner does the FLOPs measurement via
`torch.utils.flop_counter.FlopCounterMode` around the first real
inference batch, so the count reflects the adapter's actual
preprocessing pipeline (not a hand-picked dummy input that breaks
patch-aligned ViT inputs).

::: rpx_benchmark.profiler
    options:
      show_root_toc_entry: false

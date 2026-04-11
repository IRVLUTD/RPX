# UI (`rpx_benchmark.ui`)

Terminal output helper used by the CLI: a rich-based backend when
the `rich` package is available, a plain-text backend otherwise.
Both backends share the same interface (header, stage, result
table, phase score table, efficiency table, progress callback,
footer), so library code stays agnostic to which one is active.

::: rpx_benchmark.ui
    options:
      show_root_toc_entry: false
      filters:
        - "!^_"

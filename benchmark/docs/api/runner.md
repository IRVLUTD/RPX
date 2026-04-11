# Runner (`rpx_benchmark.runner`)

The runner iterates a dataset, calls the model, wraps the first
batch in `FlopCounterMode` for FLOPs, measures per-batch wall-clock
for median latency, and builds a `DeploymentReadinessReport`.

::: rpx_benchmark.runner
    options:
      show_root_toc_entry: false

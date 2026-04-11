# Models registry (`rpx_benchmark.models`)

Named factory registry for turning `"some_model_name"` into a
[`BenchmarkableModel`][rpx_benchmark.adapters.base.BenchmarkableModel]
via lazy module imports. The CLI's `--model` choice list comes from
the `available_models()` function.

## Registry

::: rpx_benchmark.models.registry
    options:
      show_root_toc_entry: false
      members_order: source

## Shipped factories

::: rpx_benchmark.models.depth_anything_v2
    options:
      show_root_toc_entry: false

::: rpx_benchmark.models.depth_pro
    options:
      show_root_toc_entry: false

::: rpx_benchmark.models.zoedepth
    options:
      show_root_toc_entry: false

::: rpx_benchmark.models.unidepth_v2
    options:
      show_root_toc_entry: false

::: rpx_benchmark.models.metric3d_v2
    options:
      show_root_toc_entry: false

## Deferred stubs

::: rpx_benchmark.models._deferred
    options:
      show_root_toc_entry: false
      filters:
        - "!^_deferred$"

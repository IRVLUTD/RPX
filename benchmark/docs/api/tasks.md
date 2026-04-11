# Tasks (`rpx_benchmark.tasks`)

End-to-end benchmark pipelines plus the task plugin registry. Nine
of ten tasks have a runnable pipeline; object tracking is deferred
until a sequence-per-sample protocol is decided.

## Registry

::: rpx_benchmark.tasks.registry
    options:
      show_root_toc_entry: false

## Shared pipeline helper

::: rpx_benchmark.tasks._pipeline
    options:
      show_root_toc_entry: false
      filters:
        - "!^_"

## Monocular absolute depth

::: rpx_benchmark.tasks.monocular_depth
    options:
      show_root_toc_entry: false

## Object segmentation

::: rpx_benchmark.tasks.segmentation
    options:
      show_root_toc_entry: false

## Object detection + open-vocab detection

::: rpx_benchmark.tasks.detection
    options:
      show_root_toc_entry: false

## Visual grounding

::: rpx_benchmark.tasks.visual_grounding
    options:
      show_root_toc_entry: false

## Relative camera pose

::: rpx_benchmark.tasks.relative_pose
    options:
      show_root_toc_entry: false

## Keypoint matching

::: rpx_benchmark.tasks.keypoint_matching
    options:
      show_root_toc_entry: false

## Sparse depth

::: rpx_benchmark.tasks.sparse_depth
    options:
      show_root_toc_entry: false

## Novel view synthesis

::: rpx_benchmark.tasks.novel_view_synthesis
    options:
      show_root_toc_entry: false

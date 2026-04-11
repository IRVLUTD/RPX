# Changelog

All notable changes to `rpx-benchmark` are recorded here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project aims to follow [Semantic Versioning](https://semver.org/)
once `1.0.0` is cut.

## [Unreleased]

### Added
- `rpx` CLI with auto-discovered task subcommands (`rpx bench <task>`).
- **Nine end-to-end task pipelines**: monocular_depth,
  object_segmentation, object_detection, open_vocab_detection,
  visual_grounding, relative_camera_pose, keypoint_matching,
  sparse_depth, novel_view_synthesis. Every one plugs through a
  shared :func:`rpx_benchmark.tasks._pipeline.run_pipeline` helper
  so new tasks are a ~100-line subclass + model resolver + TaskSpec.
- `BenchmarkableModel` adapter framework (`InputAdapter` /
  `OutputAdapter` / `PreparedInput` / default invoker).
- **Numpy fast paths for every task**: `make_numpy_depth_model`,
  `make_numpy_mask_model`, `make_numpy_detection_model`,
  `make_numpy_grounding_model`, `make_numpy_pose_model`,
  `make_numpy_keypoint_model`, `make_numpy_sparse_depth_model`,
  `make_numpy_nvs_model`. Each normalises dict-or-tuple return
  values and raises :class:`AdapterError` with a `hint` on shape
  mismatches.
- HuggingFace fast-paths: `make_hf_depth_model`,
  `make_hf_instance_seg_model`.
- Native-package adapters: `make_unidepth_v2_model`,
  `make_metric3d_v2_model`.
- Metric plugin registry (`rpx_benchmark.metrics`) — one class +
  `@register_metric` to add a new metric to any task.
- Task plugin registry (`rpx_benchmark.tasks.registry`) — one file to
  add a whole new benchmark task.
- Model plugin registry (`rpx_benchmark.models.registry`) — lazy-imported
  factories + deferred-stub support.
- Exception hierarchy (`RPXError` base + `ConfigError`, `ManifestError`,
  `DownloadError`, `ModelError`, `AdapterError`, `MetricError`) with
  `hint` and `details` on every raise.
- Structured logging (`rpx_benchmark.logging_utils`) with rich backend
  fallback and `RPX_LOG_LEVEL` env var.
- `BenchmarkRunner` integration with `torch.utils.flop_counter`
  (first-batch FLOPs) and median per-sample latency (skip warmup).
- Per-sample metadata (`id`, `phase`, `difficulty`) attached to
  `result.per_sample` without leaking into aggregate means.
- Claude-Code-style terminal UI (`rich` backend with plain-text
  fallback) showing progress bar, phase score table, efficiency table.
- Report writers: `write_json` + `format_markdown_summary` with
  JSON + markdown output per run.
- Task pipelines: monocular absolute depth + object segmentation.
- Hosted MkDocs site built automatically from numpydoc docstrings via
  `mkdocstrings`, plus a GitHub Actions workflow to deploy to Pages.
- 120-test offline pytest suite covering adapters, registries,
  metrics (all 10 tasks), deployment algebra, loader error paths,
  ground-truth loaders, hub error paths, CLI integration, exceptions
  and logging, and end-to-end synthetic pipelines for both task
  runners.
- Continuous integration workflows (`tests.yml`, `docs.yml`) across
  Python 3.10 / 3.11 / 3.12 with `ruff` lint.

### Deferred
- Video Depth Anything (sequence model; needs temporal eval mode).
- Prompt Depth Anything (needs sparse-depth prompt; belongs in a
  prompted-depth task).
- Depth Anything 3 (not yet in `transformers`).
- SAM2-native segmentation adapter (waiting for the native package
  dependency decision).

[Unreleased]: https://github.com/IRVLUTD/RPX

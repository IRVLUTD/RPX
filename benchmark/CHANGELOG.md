# Changelog

All notable changes to `rpx-benchmark` are recorded here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project aims to follow [Semantic Versioning](https://semver.org/)
once `1.0.0` is cut.

## [Unreleased]

## [0.3.0] - 2026-04-13

"Bring your own model" for real: the toolkit no longer ships any
models, no reference adapters, no model registry, and no CLI. The
split is now crisp — we provide the dataset, per-task dataloaders,
metric calculators, and hardware-agnostic profiling; you bring a
`BenchmarkableModel`. Roughly 3,000 lines of shipped-model / CLI /
reference-adapter code deleted; 215 tests passing; coverage ↑ to
77.3%.

### Removed
- **Entire `rpx_benchmark.reference` subpackage** — all reference
  adapters (`depth_hf`, `depth_unidepth`, `depth_metric3d`, `seg_hf`)
  and all reference model factories (`depth_anything_v2`, `depth_pro`,
  `zoedepth`, `unidepth_v2`, `metric3d_v2`, `_deferred`).
- **Entire `rpx_benchmark.models` subpackage** (name registry, factory
  dispatcher). There is no model registry any more.
- **CLI (`rpx_benchmark.cli`, `banner`, `ui`)** and the `rpx` console
  script. The toolkit is a library; benchmarks are run from Python.
- **Back-compat adapter shims** at the old `rpx_benchmark.adapters.{depth_hf,
  depth_unidepth, depth_metric3d, seg_hf}` paths.
- **Lazy facades** (`_schemas_lazy.py`, `_data_lazy.py`) — `import
  rpx_benchmark` now best-effort-imports `schemas` and `data` with a
  plain `try: import`.
- Install extras `depth`, `depth-hf`, `depth-unidepth`, `depth-metric3d`,
  `depth-all`, `ui` (all tied to the removed reference / CLI code).
- Docs: the whole `docs/reference/` section, `docs/api/models.md`,
  `docs/api/ui.md`, `docs/guides/adding-a-model.md`.
- Tests covering the removed features: `test_reference_relocation.py`,
  `test_registry.py` (model registry), `test_cli.py`, `test_banner.py`.

### Changed
- **`TaskSpec` slimmed**: dropped `build_config` and
  `add_cli_arguments` (CLI-only). A spec now carries identification,
  metric, modalities, `run`, and the two optional deployment-readiness
  hooks.
- **`TaskRunConfig` slimmed**: `model` is the only selector (required);
  `model_name`, `hf_checkpoint`, and `model_kwargs` are gone. Every
  task module shrank to ~30 lines.
- **Package layout** consolidated: the top-level `__init__.py` no
  longer re-exports any reference / CLI / registry symbols; only
  stable framework surface remains.
- Docs top-to-bottom rewrite to match the new simpler surface: README,
  index, quickstart, installation, BYO-model, architecture (overview /
  adapters / pipeline / registries), guides (adding-a-task,
  contributing).

### Kept
- Every data contract (`TaskType`, `Sample`, all GT / Prediction
  dataclasses, `BenchmarkModel` ABC).
- The nine `make_numpy_<task>_model(fn)` framework factories.
- Pydantic manifest schemas, HF `datasets` integration (`load_hf`,
  `RPXHFBridge`, `row_to_sample`), HF hub snapshot downloads.
- Runner with TaskSpec-dispatched deployment hooks, LatencyProfiler,
  MemoryProfiler, EfficiencyMetadata.
- Determinism helpers, pre-commit, mypy strict overlay, ruff + black
  config, CI matrix + release automation workflows.

## [0.2.0] - 2026-04-12

Full M0 → M5 production-grade refactor: Pydantic manifest schemas,
`reference/` subpackage relocation, HuggingFace `datasets`
integration, 10-task parity (including the new object-tracking
runner), runner generalized via TaskSpec hooks, profiler backends for
latency percentiles + CPU/CUDA/MPS peak memory, deterministic seeding,
and a CI / release pipeline with Trusted-Publishing PyPI automation.
Tests grew from 154 → 251 (+97); no behaviour regressions.

### Added
- **Deterministic seeding** via the new `rpx_benchmark.determinism`
  module: `seed_all(int)` seeds Python `random`, NumPy, PyTorch
  (CPU + CUDA + MPS), and `transformers.set_seed` where present;
  `deterministic(int)` is a context manager that restores the outer
  `random` + `numpy` RNG state on exit so nested blocks don't leak.
  Both degrade gracefully when torch / transformers aren't installed.
  Re-exported at the top level as `rpx.seed_all` / `rpx.deterministic`.
- Docs: new `api/determinism.md` + a Determinism section in the
  contributing guide.

### CI / release
- **CI matrix expanded** (`.github/workflows/tests.yml`):
  - pytest matrix now installs the `schemas` + `hf-datasets` extras
    so schema and HF-bridge tests run instead of importorskip-ing.
  - Coverage reporting via `pytest-cov`, floor set to 70% (current
    baseline; rachet up as coverage grows).
  - New `ruff (lint + format)` job runs the pyproject-configured
    rules (`E9`/`F63`/`F7`/`F82`/`F401`/`I`/`B`) plus advisory
    format checks for `ruff format` and `black`.
  - New `mypy (strict modules)` job runs the `[tool.mypy]`
    overrides against the stable-surface modules.
- **Release automation** (`.github/workflows/release.yml`) triggered
  by `v*.*.*` tag push: builds wheel + sdist, smoke-imports the
  wheel, publishes to PyPI via **Trusted Publishing** (OIDC — no
  API token), creates a GitHub release with notes extracted from
  the matching `CHANGELOG.md` section.
- New `[tool.coverage]` configuration in `pyproject.toml` excluding
  the reference adapters (they require torch + network to exercise
  meaningfully).

### Docs
- **Bring Your Own Model** guide rewritten around `load_hf` and a
  four-path structure: zero-code CLI, numpy callable, custom adapter
  stack, and a **new API / cloud-model template** showing how to
  wrap a remote inference service and mark it
  `EfficiencyMetadata(model_type="api")`.
- **Reference adapter section** added at `docs/reference/` with its
  own top-level navigation group. Each shipped adapter (HF depth,
  UniDepth V2, Metric3D V2, HF segmentation) gets a page with the
  install extra, usage snippet, and mkdocstrings-rendered API. The
  overview page explicitly labels these as examples — "framework vs
  reference" is now a first-class concept in the docs.
- `bring-your-own-model.md` cross-links to `reference/index.md` for
  the "copy-from-template" workflow; the framework-vs-reference
  table matches the admonition added in
  `architecture/overview.md`.

### Changed
- **Runner generalised: deployment hooks replace task branching**
  (`rpx_benchmark.runner`). `TaskSpec` now carries two optional
  callables — `temporal_stability_fn(preds, samples, poses)` and
  `geometric_coherence_fn(preds, samples)` — that the runner invokes
  uniformly. The previous hardcoded branches for depth (TS) and
  segmentation (TS + SGC) moved to the corresponding task modules,
  where new tasks can add their own hooks without touching
  `runner.py`. Existing behaviour is preserved end-to-end (same TS /
  SGC results) and verified by the pre-existing deployment tests.

### Added
- **Profiler backends** in `rpx_benchmark.profiler`:
  - `LatencyProfiler(warmup=1)` — accumulates per-sample timings and
    reports `p50`, `p95`, `p99`, and `mean` (ms), trimming warmup.
  - `MemoryProfiler()` — peak CPU RSS via `psutil` / POSIX
    `resource`, CUDA via `torch.cuda.max_memory_allocated`, MPS via
    `torch.mps.current_allocated_memory`. Each backend degrades
    gracefully to `None` when the runtime is unavailable; no new
    hard dependencies.
  - `EfficiencyMetadata` gains `peak_cpu_mb`, `peak_cuda_mb`,
    `peak_mps_mb`, `latency_p50_ms`, `latency_p95_ms`,
    `latency_p99_ms` (all optional; existing call sites unchanged).
  - Runner populates the new fields on the report's efficiency
    object; `latency_ms_per_sample` is now the p50 of the warmed
    samples (was median; same values in practice).
- Docs: `docs/architecture/overview.md` picked up a **Deployment-
  readiness hooks** section and a **Profiling** section covering
  the new backends.
- **Object tracking task reaches parity with the other nine**:
  - New `rpx_benchmark.tasks.tracking` module with
    `ObjectTrackingRunConfig`, `run_object_tracking`, and a registered
    `TASK_SPEC` (primary metric: `mota`, higher-is-better).
  - New `rpx_benchmark.make_numpy_tracking_model(fn)` factory with
    four accepted return shapes (`TrackletPrediction`, list of
    `Tracklet`, list of dicts, `{"tracks": [...]}`).
  - `TASK_SPEC` self-registers; CLI auto-picks it up so
    `rpx bench object_tracking --split hard` is wired end-to-end.
- **10-task parity guardrail** (`tests/test_all_tasks_parity.py`)
  parametrized across every `TaskType` asserting: a `TaskSpec`
  registered, at least one metric calculator registered, an HF
  `Features` schema, and a pydantic `SampleEntry` schema. A missing
  registration for a new TaskType now fails an obvious test rather
  than silently dropping coverage.
- **Tracking tests** (`tests/test_tracking.py`): numpy-adapter
  roundtrip for dict and Tracklet return shapes, MOTA/IDF1 golden
  values (perfect-recall and all-misses corners), registry dispatch,
  and an end-to-end pipeline smoke test with a tiny synthetic manifest.
- **HuggingFace `datasets` integration** in the new
  `rpx_benchmark.data` subpackage:
  - `load_hf(task, split)` — one-liner over `datasets.load_dataset`
    returning an `RPXHFBridge` that yields `list[Sample]` batches
    indistinguishable from `RPXDataset`'s output to the runner.
  - `RPXHFBridge(hf_dataset, task)` + `row_to_sample(row, task)` —
    programmatic bridge for users who build their own `datasets.Dataset`
    objects.
  - `RPX_FEATURES` + `features_for_task(task)` — canonical HF
    `Features` schemas per TaskType, shared between the bridge and
    shard generators.
  - `rpx_benchmark.data.torch_dataloader.to_torch_dataloader(...)` —
    PyTorch DataLoader adapter with a collate that preserves the
    `list[Sample]` contract.
  - `RPXDataset.from_hf(hf_ds, task=...)` classmethod delegating to
    the bridge.
- **Shard generation script** `scripts/build_hf_shards.py`: turns the
  legacy JSON-manifest + on-disk RPX scene tree into Parquet shards
  plus a `README.md` with the required `configs:` frontmatter for
  `datasets.load_dataset` to discover the task configs and splits.
- New `[hf-datasets]` install extra (`pip install
  'rpx-benchmark[hf-datasets]'`) pulling `datasets>=2.18`.
- `rpx_benchmark.data` lazy facade on the top-level package so
  `import rpx_benchmark` doesn't pull pyarrow/datasets.
- Docs: `docs/getting-started/load-the-dataset.md` covering the
  `datasets.load_dataset` path, the `huggingface_hub` snapshot path,
  PyTorch DataLoader integration, shard-building, and the canonical
  per-task schema.

### Changed
- **Reference adapters and model factories relocated** to the new
  `rpx_benchmark.reference` subpackage. Affected moves:
  - `rpx_benchmark.adapters.{depth_hf,depth_unidepth,depth_metric3d,seg_hf}`
    → `rpx_benchmark.reference.adapters.*`
  - `rpx_benchmark.models.{depth_anything_v2,depth_pro,zoedepth,
    unidepth_v2,metric3d_v2,_deferred}`
    → `rpx_benchmark.reference.models.*`
  The move sharpens the "framework vs reference" split: the framework
  surface (`adapters/base.py`, `models/registry.py`, loaders, metrics,
  tasks, runner, profiler) has no dependency on torch/transformers,
  while the `reference/` subpackage pulls them in via install extras.
- `rpx_benchmark.models.registry.get_factory` now resolves
  ``module_suffix`` entries against `rpx_benchmark.reference.models`
  by default. Third parties can pass an absolute module path
  (containing a dot) to register factories in any package.

### Deprecated
- Importing from the old `rpx_benchmark.adapters.{depth_hf,
  depth_unidepth,depth_metric3d,seg_hf}` paths continues to work via
  shim modules that re-export from `rpx_benchmark.reference.adapters`
  and emit a `DeprecationWarning`. The shims will be removed in the
  next minor release — update to the new import paths before then.

### Tooling
- **Inline type info** via a PEP 561 `py.typed` marker in the wheel,
  so downstream `mypy`/`pyright` runs pick up our annotations.
- **mypy configuration** with a strict overlay scoped to the stable
  public-surface modules (`api`, `schemas`, `_schemas_lazy`,
  `exceptions`). The rest of the package runs permissive for now and
  migrates module-by-module through later milestones.
- **Ruff lint slate expanded** with isort import ordering (`I`) and
  bugbear (`B`); added first-party grouping for `rpx_benchmark`.
- **Black** formatting configuration (100 chars, py310–py312) aligned
  with ruff so both tools are interchangeable.
- **Pre-commit** configuration (`.pre-commit-config.yaml`) running
  ruff + black + mypy (scoped) + standard hygiene checks on staged
  files.
- **`dev` extra** now bundles `mypy`, `black`, `pre-commit`, and
  `pytest-cov` alongside the existing `pytest` / `ruff` entries. Also
  pulls `pydantic` so type-checkers can resolve the new schema module
  in dev envs.
- New `docs/guides/contributing.md` covering dev install, the tooling
  slate, type-checking policy, and the commit workflow.

### Added
- **Strict manifest validation** via the new
  `rpx_benchmark.schemas` module: Pydantic v2 models for the top-level
  `Manifest` and a per-`TaskType` `SampleEntry` class for all 10
  tasks. Expose `schemas.validate_manifest(...)`,
  `schemas.dump_json_schema(task?)`, and a lazy facade on the
  top-level package so the pydantic dependency stays fully optional.
- `RPXDataset.from_manifest(..., validate=True)` and
  `RPXDataset.from_dict(..., validate=True)` opt into strict
  validation. Field-level errors surface in
  `ManifestError.details["pydantic_errors"]` with full `loc` paths
  (e.g. `("samples", 3, "depth")`).
- New `[schemas]` install extra (`pip install
  'rpx-benchmark[schemas]'`) pulling `pydantic>=2.5`.
- Docs: `api/schemas.md` + architecture/overview update covering the
  tolerant vs strict validation tiers.
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

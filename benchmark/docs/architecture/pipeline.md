# Pipeline

Every `rpx bench <task>` call follows the same five steps, regardless
of task. This page walks through exactly what happens.

## 1. Config construction

```python
cfg = MonocularDepthRunConfig(
    hf_checkpoint="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    split="hard",
    device="cuda",
)
```

The dataclass's `__post_init__` validates every field and raises
[`ConfigError`][rpx_benchmark.exceptions.ConfigError] with a `hint`
line if anything is wrong:

- Zero or multiple model selectors set.
- Unknown difficulty split.
- `batch_size < 1`.

Errors are raised at construction time — the user sees the mistake
before any weights are downloaded.

## 2. Device fallback

```python
cfg.device = _resolve_device(cfg.device)  # "cuda" → "cpu" if unavailable
```

If the user asked for `cuda` and `torch.cuda.is_available()` is
`False`, the pipeline emits a `WARNING` log line and falls back to
`cpu` **before** any model download. Users on CPU-only hosts are not
punished with a multi-GB torch download and a cryptic
`.to('cuda')` crash.

## 3. Dataset download

```python
manifest_path = download_split(task, split=cfg.split, repo_id=...)
dataset = RPXDataset.from_manifest(manifest_path, batch_size=cfg.batch_size)
```

[`hub.download_split`][rpx_benchmark.hub.download_split]:

1. Fetches the small `manifests/<task>/<split>.json` file first.
2. Extracts the unique `(scene, phase)` pairs referenced in the
   manifest.
3. Builds a list of HuggingFace `allow_patterns` globs scoped to
   **only the modalities the task needs** (e.g. `rgb/*` and
   `depth/*` for monocular depth; no masks, no poses, no QA).
4. Calls `huggingface_hub.snapshot_download(...)` with those
   patterns. The HF content-addressed cache means switching tasks
   on the same scenes later re-uses everything already on disk.
5. Writes a resolved manifest (with `root` pointing at the local
   snapshot) to `~/.cache/rpx_benchmark/` outside the HF cache so
   subsequent runs can pick it up without hitting the network.

All failures are wrapped as
[`DownloadError`][rpx_benchmark.exceptions.DownloadError] or
[`ManifestError`][rpx_benchmark.exceptions.ManifestError] with
actionable hints.

## 4. Model resolution

The task runner picks **exactly one** of:

- `cfg.model` — already-constructed `BenchmarkableModel` (takes
  precedence).
- `cfg.model_name` — name looked up in
  [`rpx_benchmark.models.registry`][rpx_benchmark.models.registry].
- `cfg.hf_checkpoint` — passed to `make_hf_*_model(checkpoint, ...)`
  for the fast path.

```python
model = _resolve_model(cfg)
model.setup()                          # load weights, warm CUDA
efficiency = _count_params_only(model) # static param count only
```

Setup is called **once** here so the runner can be invoked with
`call_setup=False` and the first batch is a "warm" forward pass, not
a load-plus-first-forward.

## 5. Benchmark runner

```python
runner = BenchmarkRunner(
    model=model,
    dataset=dataset,
    metric_suite=MetricSuite.for_task(task),
    call_setup=False,
)
result, dr_report = runner.run_with_deployment_readiness(
    primary_metric="absrel",
    model_name=display_name,
    efficiency=efficiency,
    compute_ts=True,
    compute_sgc_flag=False,
    progress=cfg.progress,
)
```

Inside the runner, per batch:

```text
t0 = time.perf_counter()
if first_batch:
    flops_g, predictions = FlopCounterMode(model.predict, batch)
else:
    predictions = model.predict(batch)
batch_seconds = time.perf_counter() - t0
per_sample_seconds.extend([batch_seconds / len(batch)] * len(batch))

for sample, pred in zip(batch, predictions):
    validate_prediction(task, pred, sample)
    metrics = metric_suite.evaluate(pred, sample.ground_truth)
    metrics.update(_sample_meta(sample))     # id/phase/difficulty
    per_sample_metrics.append(metrics)
```

After the loop:

```text
latency_ms = median(per_sample_seconds[1:])     # skip warmup batch
wps  = compute_weighted_phase_score(...)        # ESD-weighted per-phase
str_ = compute_str(phase_scores)                # interaction drop + recovery
ts   = compute_temporal_stability_depth(...)    # optional
```

The result is two objects:

- **[`BenchmarkResult`][rpx_benchmark.metrics.registry.BenchmarkResult]**
  with `per_sample` (numeric metrics **+** metadata),
  `aggregated` (numeric-only means), and `num_samples`.
- **[`DeploymentReadinessReport`][rpx_benchmark.deployment.DeploymentReadinessReport]**
  with WPS, STR, TS, FLOPs, median latency, and parameter count.

## 6. Reports

Both outputs are written to
`./rpx_results/<model>/<split>/` via `write_json` and
`format_markdown_summary` in
[`rpx_benchmark.reports`](../api/reports.md).

The terminal UI renders the same data live through panels, tables,
and a progress bar (`rich`), or plain text when rich is unavailable
or `--plain` is passed.

# Pipeline

Every `run_<task>(cfg)` call follows the same five steps. The
task-specific bits (`TaskType`, primary metric, optional
deployment-readiness hooks) live on the task's `TaskSpec`; the
plumbing lives in `rpx_benchmark.tasks._pipeline.run_pipeline`.

```text
cfg  ─►  download split  ─►  RPXDataset  ─►  BenchmarkRunner  ─►  reports
            (hub)             (loader)         (runner + profilers)
```

## The five steps

```python
from rpx_benchmark import MonocularDepthRunConfig, make_numpy_depth_model, run_monocular_depth

def my_depth(rgb):      # H x W x 3 uint8 -> H x W float32 (metres)
    ...

cfg = MonocularDepthRunConfig(
    model=make_numpy_depth_model(my_depth, name="my_depth"),
    split="hard",
    device="cpu",
)

result, report, paths = run_monocular_depth(cfg)
```

What `run_monocular_depth` does, step by step:

1. **Resolve device.** `cuda` requests fall back to CPU when
   `torch.cuda.is_available()` is False.
2. **Download the split.** `rpx_benchmark.hub.download_split` pulls
   only the modalities this task needs (RGB + depth for monocular
   depth, RGB + mask for segmentation, ...).
3. **Load.** `RPXDataset.from_manifest(path)` yields
   `list[Sample]` batches.
4. **Run.** `BenchmarkRunner.run_with_deployment_readiness`:
   - wraps the first batch in torch's `FlopCounterMode` for FLOPs,
   - records per-sample latency through `LatencyProfiler`
     (p50/p95/p99/mean, skipping warmup),
   - samples peak CPU / CUDA / MPS memory via `MemoryProfiler`,
   - scores each sample through the task's registered
     `MetricCalculator`(s),
   - dispatches optional deployment-readiness hooks on the task's
     `TaskSpec`.
5. **Write reports.** `result.json` + `summary.md` under
   `output_dir` (defaults to `./rpx_results/<model>/<split>/`).

## Extending

To add a task, write a ~30-line `tasks/<my_task>.py` module mirroring
the existing ones — a `TaskRunConfig` subclass (usually empty), a
`run_<task>` wrapper around `run_pipeline`, and a `TASK_SPEC` that
self-registers on import. See
[Adding a Task](../guides/adding-a-task.md) for the full recipe.

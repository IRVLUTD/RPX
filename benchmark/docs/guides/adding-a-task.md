# Adding a New Task

Adding a new task is a **one-file change**: clone
`rpx_benchmark/tasks/monocular_depth.py`, swap the task-specific
knobs, and register a `TaskSpec`. The CLI auto-discovers the new
task at parser-build time and exposes `rpx bench <your_task>` with
zero edits to `cli.py`.

## Minimum viable task module

```python
# rpx_benchmark/tasks/my_task.py
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..adapters import BenchmarkableModel
from ..api import Difficulty, TaskType
from ..deployment import DeploymentReadinessReport
from ..exceptions import ConfigError
from ..hub import download_split
from ..loader import RPXDataset
from ..logging_utils import get_logger
from ..metrics.registry import BenchmarkResult, MetricSuite
from ..profiler import EfficiencyMetadata, count_parameters
from ..reports import format_markdown_summary, write_json
from ..runner import BenchmarkRunner, ProgressCallback
from .registry import TaskSpec, register_task

log = get_logger(__name__)
PRIMARY_METRIC = "my_metric"       # key your MetricCalculator emits


@dataclass
class MyTaskRunConfig:
    model: Optional[BenchmarkableModel] = None
    model_name: Optional[str] = None
    hf_checkpoint: Optional[str] = None
    split: Difficulty | str = Difficulty.HARD
    repo_id: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    batch_size: int = 1
    device: str = "cuda"
    output_dir: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[ProgressCallback] = None

    def __post_init__(self) -> None:
        given = [x for x in (self.model, self.model_name, self.hf_checkpoint) if x]
        if len(given) != 1:
            raise ConfigError(
                "MyTaskRunConfig requires exactly one of model / model_name / hf_checkpoint.",
            )
        if isinstance(self.split, str):
            self.split = Difficulty(self.split)
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")


def run_my_task(cfg: MyTaskRunConfig) -> Tuple[BenchmarkResult, DeploymentReadinessReport, Dict[str, Path]]:
    task = TaskType.MY_TASK_TYPE        # add to api.py if it's genuinely new

    # 1. Dataset
    manifest_path = download_split(task, split=cfg.split,
                                   repo_id=cfg.repo_id or "IRVLUTD/rpx-benchmark",
                                   cache_dir=cfg.cache_dir, revision=cfg.revision)
    dataset = RPXDataset.from_manifest(manifest_path, batch_size=cfg.batch_size)

    # 2. Model
    model = _resolve_model(cfg)         # same three-way dispatch as depth task
    model.setup()
    efficiency = EfficiencyMetadata(params_m=count_parameters(model.model))

    # 3. Runner
    runner = BenchmarkRunner(model=model, dataset=dataset,
                             metric_suite=MetricSuite.for_task(task),
                             call_setup=False)
    result, dr_report = runner.run_with_deployment_readiness(
        primary_metric=PRIMARY_METRIC,
        model_name=getattr(model, "name", "model"),
        efficiency=efficiency,
        compute_ts=True, compute_sgc_flag=False,
        progress=cfg.progress,
    )

    # 4. Reports
    split_name = cfg.split.value if isinstance(cfg.split, Difficulty) else str(cfg.split)
    out_dir = Path(cfg.output_dir or f"./rpx_results/{model.name}/{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    write_json(json_path, task=task.value, model_name=model.name,
               split=split_name, repo_id=cfg.repo_id, result=result, dr_report=dr_report)
    md_path.write_text(format_markdown_summary(...), encoding="utf-8")

    return result, dr_report, {"json": json_path, "markdown": md_path}


# CLI glue
def _add_cli_arguments(p: argparse.ArgumentParser) -> None:
    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument("--model")
    mg.add_argument("--hf-checkpoint")
    p.add_argument("--split", required=True,
                   choices=[d.value for d in Difficulty])
    p.add_argument("--repo", default="IRVLUTD/rpx-benchmark")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None)


def _build_config(args: argparse.Namespace) -> MyTaskRunConfig:
    return MyTaskRunConfig(
        model_name=args.model,
        hf_checkpoint=args.hf_checkpoint,
        split=args.split,
        repo_id=args.repo,
        device=args.device,
        output_dir=args.output_dir,
    )


TASK_SPEC = TaskSpec(
    task=TaskType.MY_TASK_TYPE,
    display_name="My Task",
    description="One-line description shown in `rpx bench --help`.",
    primary_metric=PRIMARY_METRIC,
    required_modalities=["rgb", "..."],
    higher_is_better=True,
    build_config=_build_config,
    run=run_my_task,
    add_cli_arguments=_add_cli_arguments,
)

register_task(TASK_SPEC)
```

## Register the module in the tasks package

Add one import line to `rpx_benchmark/tasks/__init__.py`:

```python
from .my_task import MyTaskRunConfig, run_my_task, TASK_SPEC as MY_TASK_SPEC
```

That triggers `register_task(TASK_SPEC)` at package import time, and
the CLI picks up `rpx bench my_task` automatically.

## Checklist

- [ ] `TaskType.MY_TASK_TYPE` exists in `rpx_benchmark/api.py`
- [ ] At least one `MetricCalculator` is registered for that task
      (see [Adding a Metric](adding-a-metric.md))
- [ ] Ground-truth dataclass matches what the loader produces for the
      manifest field names you use
- [ ] `run_my_task` exercises the download → load → run → report
      flow without importing `cli.py` (task runners must be
      importable from Python, not just from the CLI)
- [ ] Tests under `tests/test_my_task_pipeline.py` using
      [`make_numpy_depth_model`-style helpers](../api/adapters.md)
      exercise the end-to-end path on a synthetic dataset fixture.

See
[`rpx_benchmark/tasks/monocular_depth.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/tasks/monocular_depth.py)
and
[`rpx_benchmark/tasks/segmentation.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/tasks/segmentation.py)
for two complete examples.

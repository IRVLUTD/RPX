# Toolkit capabilities

Everything the toolkit can do, and where it lives. The benchmark runners and analysis APIs ship in the PyPI package; annotation tools, the paper's model runners and their containers are in the repository.

<figure class="rpx-workflow-figure"><a href="../assets/toolkit-overview.svg"><img src="../assets/toolkit-overview.svg" alt="RPX toolkit capabilities from scene data and adapters through metrics, profiling and extensibility." loading="lazy"></a><figcaption>How the pieces fit together, illustrated with scene 012. Open the SVG to zoom.</figcaption></figure>

| Capability | Where to start | Where it lives |
| --- | --- | --- |
| Discover, download and load data | [Data contracts](../data/README.md), [Hub API](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/hub.html) | PyPI; Hub extra for downloads/Parquet |
| Validate manifests and units | [Schemas API](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/schemas.html) | PyPI `schemas` extra |
| Six paper task interfaces | [Task benchmark examples](../benchmarks/README.md) | PyPI; pretrained dependencies separate |
| Use local or API models | [Bring your own model](../models/README.md) | PyPI adapters and VQA callable contract |
| Annotate and refine GT masks | [Mask pipeline](../annotation/README.md) | Source checkout; CUDA and review UI |
| Measure latency and memory | [Hardware profiler](../profiling/README.md) | PyPI; PyTorch/CUDA for GPU measurement |
| Compute phase robustness and desirability | [Φ/JEDI calculator](../analysis/README.md) | PyPI; raw paired metrics required |
| Register new metrics or runners | [Add metrics](../metrics/README.md), [Add tasks](../tasks/README.md) | PyPI runtime registration; new identities require source changes |
| Score scene difficulty | [ESD extraction](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/esd.html), [ESD scoring](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/data/esd_scoring.html) | PyPI; analysis/optional decoding extras |
| Build manifests and tar shards | [Dataset hub CLI](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/dataset_hub/cli.html) | PyPI Hub tools |
| Inspect scenes interactively | [Rerun scene tool](https://github.com/IRVLUTD/RPX/blob/main/benchmark/scripts/visualize_rerun.py) | Repository script; `viz` extra |
| Detection, segmentation, sparse depth, keypoint matching | [Task API inventory](../api/README.md) | Additional registered PyPI task runners |
| Canonical question generation and VQA validation | [VQA source](https://github.com/IRVLUTD/RPX/tree/main/data/mask_annotation/visual_grounding_gt/vqa_gt), [VQA APIs](../api/README.md#vqa-contracts-and-evaluation) | Generation in source; evaluation in PyPI |
| Per-sample/cell logs and structured reports | [Cell log API](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/cell_log.html), [Reports API](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/reports.html) | PyPI |
| Batch adapters and depth alignment | [Adapter inventory](../api/README.md#model-adapters), [Alignment API](https://irvlutd.github.io/RPX/toolkit-docs/rpx_benchmark/metrics/depth_alignment.html) | PyPI; model support/protocol dependent |

## Discover the installed registries

```python
from rpx_benchmark.tasks import available_tasks
from rpx_benchmark.metrics import available_metrics
print([task.value for task in available_tasks()])
print(available_metrics())
```

Registered tasks are reusable APIs; the paper's six tasks each add a frozen evaluation protocol on top. A new model function or metric doesn't change those protocols.

[Get started](../getting-started/README.md) · [Full API reference](../api/README.md) · [Citation and licenses](../citation/README.md)

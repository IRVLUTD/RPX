# Load the dataset

RPX ships two paths for pulling the benchmark data. Pick the one that
fits your workflow:

| Path | Backend | Best for |
|---|---|---|
| **`datasets.load_dataset`** | [`hf-datasets`](https://huggingface.co/docs/datasets/) extra | Quick starts, streaming, PyTorch / JAX training loops |
| **`rpx_benchmark.hub`** | [`hub`](https://huggingface.co/docs/huggingface_hub) extra | Modality-aware bulk snapshots, offline reuse |

Both paths yield the same `rpx_benchmark.Sample` contract downstream —
so adapters and metrics don't care which you chose.

## Option 1 — `datasets` one-liner (recommended)

```bash
pip install 'rpx-benchmark[hf-datasets]'
```

```python
from rpx_benchmark.data import load_hf

ds = load_hf("monocular_depth", split="hard")
for batch in ds:
    for sample in batch:           # list[Sample]
        # sample.rgb           -> H×W×3 uint8
        # sample.ground_truth  -> DepthGroundTruth(depth_map: H×W float32 metres)
        # sample.phase         -> Phase.CLUTTER / .INTERACTION / .CLEAN
        # sample.difficulty    -> Difficulty.EASY / .MEDIUM / .HARD
        ...
```

`load_hf` is a thin wrapper over `datasets.load_dataset` plus our
`RPXHFBridge`. It understands:

- **Task configs** — `"monocular_depth"`, `"object_segmentation"`, ...
  (the enum values of `TaskType`).
- **Splits** — `"easy"`, `"medium"`, `"hard"` (the `Difficulty` enum).
- **Streaming** — `load_hf(..., streaming=True)` forwards to the
  `datasets` streaming path so you don't need the full shards on disk.
- **Pinned revisions** — `load_hf(..., revision="v0.3.0")` pins the
  dataset release you want.

### Already have a `datasets.Dataset`?

If you built your own `Dataset` (for example during data engineering
experiments), wrap it directly:

```python
from rpx_benchmark import RPXDataset
from rpx_benchmark.api import TaskType

bridge = RPXDataset.from_hf(my_hf_dataset, task=TaskType.MONOCULAR_DEPTH)
```

### PyTorch DataLoader

```python
from rpx_benchmark.data.torch_dataloader import to_torch_dataloader

dl = to_torch_dataloader(ds, num_workers=4, pin_memory=True)
```

The DataLoader yields `list[Sample]` so the collate stays trivial —
task-specific tensor stacking happens inside your `InputAdapter`.

## Option 2 — modality-aware snapshot (full-dataset workflows)

```bash
pip install 'rpx-benchmark[hub]'
```

```python
import rpx_benchmark as rpx

ds = rpx.hub.load("monocular_depth", "hard")  # RPXDataset
```

`rpx_benchmark.hub.load` calls `huggingface_hub.snapshot_download` with
an `allow_patterns` list derived from the task: RGB + depth for
`monocular_depth`, RGB + mask for `object_segmentation`, and so on.
Re-running a second task on the same scenes reuses the cached RGB
frames and pulls only the new label files.

## Building the Parquet shards yourself

If you are publishing your own RPX-style dataset to the Hub, use the
shard-generation script as the entry point:

```bash
python scripts/build_hf_shards.py \
    --source-root   /mnt/rpx/ \
    --manifests-root /mnt/rpx/manifests/ \
    --out           build/hf_shards/
```

The script writes one Parquet shard per `(task, split)` pair plus a
`README.md` with the required `configs:` frontmatter so
`datasets.load_dataset` can discover them. Upload with:

```bash
huggingface-cli upload my-org/my-rpx-dataset build/hf_shards/ .
```

## Canonical schema (for tool builders)

The shared column layout per task is exported as a
[`datasets.Features`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Features)
dict:

```python
from rpx_benchmark.data.features import features_for_task
from rpx_benchmark.api import TaskType

features_for_task(TaskType.MONOCULAR_DEPTH)
# Features({
#   'id': Value('string'),
#   'scene': Value('string'),
#   'phase': Value('string'),
#   'difficulty': Value('string'),
#   'rgb': Image(decode=True, ...),
#   'camera_pose': Sequence(Value('float32'), length=16),
#   'depth': Image(decode=True, ...),
# })
```

Use this schema when writing your own shard generators so the
`RPXHFBridge` can decode rows without task-specific configuration.

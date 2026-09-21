# RPX Benchmark Toolkit

Load RPX data, run reference or custom perception models, and compare accuracy,
scene-change robustness and compute cost with Python 3.10–3.12:

```bash
git clone <ANONYMOUS_REPOSITORY_URL>
cd RPX
python -m pip install -e './benchmark[hub,schemas]'
python benchmark/examples/run_depth.py
```

The example is a CPU-only installation check using synthetic data. Reference
checkpoints need their own runtime, weights and GPU resources. The package
installation alone does not install every model's dependencies.

## Reference models

Use isolated environments because model dependencies conflict. The linked
READMEs contain build, checkpoint, cache and execution instructions. Run a
model's small inference gate before a full benchmark, and retain its dataset
revision and runtime metadata with the results.

| Evaluation | Models and commands | Environment |
|---|---|---|
| Image depth | Ten canonical models, including DA3, DA-V2 Large, Depth Pro, UniDepth V2, Metric3D V2, MoGe-2, HyDen, Lotus 2, FE2E and ZipDepth; `scripts/run_depth.py` | [Depth](../docker/depth-smoke/README.md), with dedicated [FE2E](../docker/depth-fe2e/README.md) and [ZipDepth](../docker/depth-zipdepth/README.md) overlays |
| Video depth | DA3, DepthCrafter, Video Depth Anything, ChronoDepth, RollingDepth, DVD, GEMDepth, ViGeo, MonST3R and VGGT; `scripts/run_video_depth.py` | [Depth](../docker/depth-smoke/README.md), [DVD](../docker/depth-dvd/README.md), [GEMDepth](../docker/depth-gemdepth/README.md) |
| Relative pose | VGGT, DA3, CUT3R, DUSt3R, MASt3R, MUSt3R, Reloc3r, Pi3X, Fast3R and MonST3R; additional baseline adapters are retained | [Relative pose](../docker/rcpe-smoke/README.md) |
| Tracking | Mask-, box- and text-initialized adapters, with MOS and Ego protocols; `scripts/run_tracking.py` and `scripts/run_tracking_paper.py` | [Tracking](../docker/tracking-smoke/README.md) |
| VQA / bbox grounding | Twenty roster entries with native Transformers or vLLM backends, one- and two-image inputs; `scripts/run_vqa_benchmark.py` | [VQA](../docker/vqa-smoke/README.md) |

For segmentation, detection, open-vocabulary detection, generic grounding,
sparse depth and keypoint matching, the toolkit supplies the callable APIs,
loaders and evaluators below. There is no bundled full pretrained-model sweep
for those tasks. Mask annotation models produce dataset labels; they are not
a substitute for a benchmark adapter.

Example, **inside the selected model environment**, from `benchmark/`:

```bash
python scripts/run_depth.py --model da-v2-large --split easy \
  --repo <REVIEW_DATASET_REPO> --device cuda --max-samples 10 \
  --output-dir ../rpx_results/da-v2-large/easy-smoke
python scripts/run_video_depth.py --help
python scripts/run_relative_pose.py --help
python scripts/run_tracking.py --help
```

`python benchmark/scripts/run_depth_smoke_matrix.py --list-models` lists the
canonical depth roster without loading weights. Docker matrix files and adapter
registries are the source of model identifiers and pinned checkpoint revisions.
Model availability and successful inference are different checks: the offline
suite tests adapters with controlled inputs, while GPU gates run real weights.

## Bring your own model

Wrap a callable and pass it to a public task runner. The callable may invoke a
local model or your inference service; convert its response to the documented
NumPy output. Authentication and service configuration belong in your wrapper,
not in committed source.

```python
import rpx_benchmark as rpx
from my_model import predict_depth  # your implementation: RGB -> depth in metres

model = rpx.make_numpy_depth_model(predict_depth, name="my-depth")
result, report, paths = rpx.run_monocular_depth(
    rpx.MonocularDepthRunConfig(
        model=model,
        split="easy",
        repo_id="<REVIEW_DATASET_REPO>",
        revision=rpx.DEFAULT_REVISION,
        device="cpu",  # configure your callable's actual device yourself
        output_dir="rpx_results/my-depth/easy",
    )
)
print(result.aggregated)
print(paths["json"])
```

For a local dataset, supply `manifest_path="/data/manifests/monocular_depth/easy.json"`
to skip downloading. Manifests name a task and dataset root and list samples
with modality paths, scene, phase and difficulty. Local examples and all ten
public workflows are exercised in [tests/test_user_workflows.py](tests/test_user_workflows.py).
For relative depth, set `depth_output_kind="relative"` on the factory; never
label arbitrary relative values as metric depth.

| Task | Factory (under `rpx`) | Callable input → output | Public runner |
|---|---|---|---|
| Monocular depth | `make_numpy_depth_model` | RGB `(H,W,3)` → depth `(H,W)` | `run_monocular_depth` |
| Video depth | `make_numpy_video_depth_model` | RGB `(T,H,W,3)` → depth `(T,H,W)` | `run_video_depth` |
| Segmentation | `make_numpy_mask_model` | RGB → integer instance mask; 0 is background | `run_segmentation` |
| Detection | `make_numpy_detection_model` | RGB → boxes, scores, labels | `run_object_detection` |
| Open-vocabulary detection | `make_numpy_detection_model(..., task=rpx.TaskType.OPEN_VOCAB_DETECTION)` | RGB → boxes, scores, labels; bind your vocabulary in the callable | `run_open_vocab_detection` |
| Tracking | `make_numpy_tracking_model` | RGB → tracks with persistent IDs and boxes | `run_object_tracking` |
| Grounding | `make_numpy_grounding_model` | RGB, text → boxes and scores | `run_visual_grounding` |
| Relative pose | `make_numpy_pose_model` | RGB A, RGB B → rotation `(3,3)`, translation `(3,)` | `run_relative_pose` |
| Sparse depth | `make_numpy_sparse_depth_model` | RGB, pixel coordinates → depth values | `run_sparse_depth` |
| Keypoint matching | `make_numpy_keypoint_model` | RGB A, RGB B → points0, points1 | `run_keypoint_matching` |

Factory docstrings specify shapes and units. For custom preprocessing, batching
or framework-specific outputs, implement `BenchmarkModel` or compose
`BenchmarkableModel` with input/output adapters. The generic tracking API is a
frame-based contract; use the production tracking runner for the full clip-based
paper protocol and its initialization/scoring rules.

### Custom VQA model or API

VQA uses JSONL question manifests rather than the core task manifests. A custom
callable receives only ordered image paths and the public prompt; ground-truth
answers and boxes are kept inside the evaluator.

```python
from rpx_benchmark.vqa import evaluate_vqa
from my_model import predict_vqa

# predict_vqa(image_paths, prompt, max_new_tokens, output_kind) -> raw text
# image_paths: (target,) or (reference_crop, target), in that order.
report = evaluate_vqa(
    "vqa_manifest.jsonl", predict_vqa,
    model_name="my-vlm", model_revision="my-checkpoint-or-service-version",
    image_cache=".cache/rpx-vqa", output_dir="rpx_results/my-vlm",
)
print(report["metrics"])
```

Return one JSON object with `label` and `bbox` (XYXY coordinates normalized to
0–1000); binary questions require `yes` or `no`. Parsing failures and inference
errors remain in the denominator. Runs save raw outputs, errors, timing,
manifest hash, model revision and scored results. Use a fresh output directory.
The [VQA runtime README](../docker/vqa-smoke/README.md) explains manifest
preparation and the reference-model benchmark. The preserved Molmo adapter in
`scripts/vqa_models/molmo_backend.py` is optional and outside the frozen roster.

## Data and results

The Hub loader fetches only required modalities and reuses its cache. Pin a
dataset commit for comparable runs; do not mix results from moving revisions.
The [data README](data/README.md) identifies the local canonical splits and
metadata. Dataset publishing tools are documented separately in
[rpx_benchmark/dataset_hub/README.md](rpx_benchmark/dataset_hub/README.md).

Generic frame runs write `result.json`, `summary.md`, `cells.parquet` and
`per_sample_metrics.parquet`. Video runs write clip/cell results. Hardware,
precision and batch size belong in each cell; unavailable measurements stay
unavailable. Custom API wall time includes transport and is not a GPU-only
latency measurement. Production runners add prediction caches, protocol
metrics and provenance records.

## Testing

From `benchmark/`, in a clean virtual environment:

```bash
python -m pip install -e '.[dev,hub,schemas,hf-datasets,analysis,ar-pose-audit]' \
  imageio imageio-ffmpeg av transformers build
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps 'git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a'
pytest tests -q --cov=rpx_benchmark --cov-report=term-missing
pytest ../data/mask_annotation/visual_grounding_gt/vqa_gt/tests -q
ruff check rpx_benchmark tests scripts
mypy rpx_benchmark
python -m build
```

Tests cover task contracts, local datasets, public custom-model workflows,
metrics, reports, download behavior, model-call signatures and VQA parsing.
They do not download model checkpoints. Run the linked CUDA inference gates
on the target machine to validate real models, available memory and dataset
access. Sensor capture and annotation UI checks require their own hardware.

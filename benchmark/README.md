<p align="center">
  <img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires." width="100%"/>
</p>


<h3 align="center">Same Scene, Different Story:<br>Evaluating Robot Perception Across Scene Phases in the Wild</h3>

<p align="center">
  Score a perception model on 100 real scenes recorded before, during and after manipulation,<br>
  and see where it breaks with phase robustness Φ and worst-phase quality 𝒥<sub>min</sub>.
</p>

<p align="center">
  <a href="https://pypi.org/project/rpx-benchmark/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/rpx-benchmark?label=pypi&color=4F46E5&logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/rpx-benchmark/"><img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white"></a>
  <a href="https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests&logo=github"></a>
  <a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img alt="Toolkit docs" src="https://img.shields.io/badge/docs-toolkit-0d9488?logo=materialformkdocs&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img alt="Dataset on Hugging Face" src="https://img.shields.io/badge/dataset-IRVLUTD%2FRPX-ffcc4d?logo=huggingface&logoColor=white"></a>
  <a href="https://hub.docker.com/r/irvlutd/rpx"><img alt="Docker images" src="https://img.shields.io/badge/docker-irvlutd%2Frpx-2496ED?logo=docker&logoColor=white"></a>
  <a href="https://github.com/IRVLUTD/RPX"><img alt="Source on GitHub" src="https://img.shields.io/badge/source-IRVLUTD%2FRPX-24292f?logo=github"></a>
</p>

## Getting started

The `rpx-benchmark` package provides the data, an adapter per task, the paper's benchmark
protocol, and the Φ and 𝒥<sub>min</sub> calculator. You write one function that runs your model.

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

The example below scores an image-depth model on a 6-frame trial. The download fetches only this
task's files (about 160 MB) and needs no login.

```python
import numpy as np
import rpx_benchmark as rpx

# 1. Download the task's data
manifest = rpx.download_split("monocular_depth", "easy", max_samples=6)

# 2. Your model: RGB image (H, W, 3) in, depth (H, W) in metres out
def predict(rgb):
    return np.ones(rgb.shape[:2], np.float32)

# 3. Wrap it in the task's adapter and run the benchmark
model = rpx.make_numpy_depth_model(predict, name="my-model")
result, report, paths = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(
    model=model, manifest_path=str(manifest), output_dir="rpx_results/my-model",
))
print(result.aggregated["absrel"], result.aggregated["delta1"])
```

Φ and 𝒥<sub>min</sub> compare phases, so they need all three phases of many scenes. Drop
`max_samples` to download the full tier, run the benchmark again, then:

```python
from rpx_benchmark.examples.summarize_metrics import calculate, read_records

metrics = ["absrel", "rmse", "silog", "delta1"]
summary = calculate(read_records(paths["json"]), metrics)
```

To try any task before writing code, `rpx.benchmark("T2", rpx.download("T2", max_samples=2))`
runs it with a placeholder model.

## Tasks

Every task follows the steps above with its own download name, adapter and runner.

| Task | Download | Adapter and runner | Paper metrics |
|:--|:--|:--|:--|
| T1 Image depth | `"monocular_depth"` | `make_numpy_depth_model`<br>`run_monocular_depth` | AbsRel, RMSE, SILog, δ1 |
| T2 Video depth | `"video_depth"` | `make_numpy_video_depth_model`<br>`run_video_depth` | T1's, plus TGM, TGSE |
| T3 Object tracking | `"object_tracking"` | `make_numpy_tracking_model`<br>`run_object_tracking` | HOTA, MOTA, IDSW |
| T4 Relative camera pose | `"relative_camera_pose"`* | `make_numpy_pose_model`<br>`run_relative_pose` | AUC at 5°, 10°, 20° |
| T5 VQA | `rpx.download("T5")` | `vqa.evaluate_vqa` | Acc@0.5, GIoU, center-in-GT |
| T6 In-context VQA | `rpx.download("T6")` | `vqa.evaluate_vqa` | Acc@0.5, GIoU, center-in-GT |

<sub>* Pass `manifest_name="relative_pose"` as well. Functions are under `rpx`; `vqa` is
`rpx_benchmark.vqa`, and its models return text, so they need no adapter.
[`dummy_models.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/examples/dummy_models.py) shows the inputs and
outputs of every task.</sub>

<details>
<summary>Prefer the command line?</summary>

```bash
python -m rpx_benchmark.examples.benchmark_tasks \
    --task T1 --split easy --max-samples 6 --model my_model:predict

# Check the install without downloading anything
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke
```

</details>

To add a task or a metric, see the toolkit docs on
[tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) and
[metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/).

## Model contracts and paper models

The table lists what your function receives and returns for each task, and the Docker image that contains
every model the paper evaluated on it.

| Task | Input → output | Paper's models, `irvlutd/rpx:` |
|:--|:--|:--|
| T1 Image depth | RGB `(H, W, 3)` → depth `(H, W)` in metres | `depth-zipdepth-latest` (10) |
| T2 Video depth | clip `(T, H, W, 3)` → depth `(T, H, W)` | `depth-zipdepth-latest` (8)<br>`depth-dvd-latest` (DVD) |
| T3 Object tracking | frame `(H, W, 3)` → `[{"track_id", "boxes"}]` | `tracking-dam4sam-rpx-latest` (8) |
| T4 Relative camera pose | `(rgb_a, rgb_b)` → `{"rotation", "translation"}` | `rcpe-monst3r-rpx-latest` (10) |
| T5 VQA | image path + question → JSON `{"label", "bbox"}` | `vqa-vllm` (12) |
| T6 In-context VQA | reference crop + image + question → same | `vqa-vllm` (12) |

[`dummy_models.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/examples/dummy_models.py)
has a runnable placeholder for each row. Models that predict relative depth pass
`depth_output_kind="relative"` to the adapter. The [model guide](https://irvlutd.github.io/RPX/toolkit-docs/models/)
documents the full input contracts, and the [benchmark pages](https://irvlutd.github.io/RPX/toolkit-docs/benchmarks/)
walk through each task with figures.

For T5 and T6, `rpx.download("T5")` returns the packaged question list. Pass it, or your own list, to
`evaluate_vqa`; [the next section](#custom-vqa-model-or-api) covers models behind an API.

Always pull a specific tag: the bare `irvlutd/rpx` image is an unrelated legacy image. The
[Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md#which-image-to-pull) also covers the
box- and text-prompted trackers and GemDepth, which ships as a build recipe only.

The generic T3 and T4 runners report per-sample errors. The paper's HOTA and pooled pose AUC come from the
clip- and pair-level reference runners in
[`benchmark/scripts`](https://github.com/IRVLUTD/RPX/tree/main/benchmark/scripts).

## Custom VQA model or API

Any local model or hosted API works as long as it returns text:

```python
from rpx_benchmark.vqa import evaluate_vqa

def predict_vqa(image_paths, prompt, max_new_tokens, output_kind):
    # image_paths is (target,) or (reference_crop, target).
    # Return your model's raw text.
    ...

report = evaluate_vqa(
    "vqa_manifest.jsonl", predict_vqa,
    model_name="my-vlm", model_revision="v1",
    image_cache=".cache/rpx-vqa", output_dir="rpx_results/my-vlm",
)
print(report["metrics"]["bbox_accuracy_at_0_5"])
```

The reply should be one JSON object with `label` and `bbox` (XYXY, normalized to 0–1000). The model never sees
the answers, and replies that fail to parse count as misses. The
[VQA guide](https://github.com/IRVLUTD/RPX/blob/main/docker/vqa-smoke/README.md) explains how to prepare the
manifest.

## Results

In the paper, quality drops once hands enter the scene, and none of the 65 evaluations is both accurate and
robust across phases.

<p align="center">
  <img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-hero-light.webp" alt="RPX: the same real scene before, during and after manipulation and from an egocentric view; mean quality per view for video depth, tracking and VQA drops when hands enter the scene; none of the 65 evaluations in the paper reaches both worst-phase quality 0.75 and phase robustness 0.86; three steps: pip install rpx-benchmark, run your model, get phi and J-min." width="100%"/>
</p>

## Further reading

- [Φ and JEDI calculator](https://irvlutd.github.io/RPX/toolkit-docs/analysis/): compute Φ and 𝒥<sub>min</sub> from existing results
- [Hardware profiler](https://irvlutd.github.io/RPX/toolkit-docs/profiling/): latency, throughput and memory
- [Mask annotation pipeline](https://irvlutd.github.io/RPX/toolkit-docs/annotation/): label your own scenes the way RPX was labelled
- [Metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/) and [tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/): add your own
- [Dataset card](https://huggingface.co/datasets/IRVLUTD/RPX) and [data guide](https://irvlutd.github.io/RPX/toolkit-docs/data/): 100 scenes, 70 objects, 133,121 frames

## Testing

From `benchmark/` in a [source checkout](https://github.com/IRVLUTD/RPX):

```bash
python -m pip install -e '.[dev,hub,schemas,hf-datasets,analysis,ar-pose-audit]' \
    imageio imageio-ffmpeg av transformers build
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
TRACKEVAL=12c8791b303e0a0b50f753af204249e622d0281a
python -m pip install --no-deps \
    "git+https://github.com/JonathonLuiten/TrackEval.git@$TRACKEVAL"

pytest tests -q
ruff check rpx_benchmark tests scripts
mypy rpx_benchmark
python -m build
```

The tests never download checkpoints. Real-model checks run through the per-image GPU gates in the
[Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md).

## Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across
            Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and
            Kamath, Srinanditha and Allu, Sai Haneesh and Rangappa, Govind Tyagi and
            Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

The toolkit is MIT-licensed and the dataset is CC BY 4.0. Pretrained models keep their own licenses.

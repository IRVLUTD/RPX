<p align="center">
  <img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires." width="100%"/>
</p>

<h3 align="center">Same Scene, Different Story:<br> Evaluating Robot Perception Across Scene Phases in the Wild</h3>

<p align="center">
  Score any perception model on 100 real scenes recorded <b>before, during and after manipulation</b>,<br/>
  and see where it breaks with two numbers: <b>Φ</b> (phase robustness) and <b>𝒥<sub>min</sub></b> (worst-phase quality).
</p>

<p align="center">
  <a href="https://pypi.org/project/rpx-benchmark/"><img src="https://img.shields.io/pypi/v/rpx-benchmark?color=4F46E5" alt="PyPI"></a>
  <a href="https://pypi.org/project/rpx-benchmark/"><img src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue" alt="Python 3.10–3.12"></a>
  <a href="https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img src="https://img.shields.io/badge/docs-toolkit-0d9488" alt="Docs"></a>
  <a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-IRVLUTD%2FRPX-yellow" alt="Dataset"></a>
  <a href="https://hub.docker.com/r/irvlutd/rpx"><img src="https://img.shields.io/badge/docker-irvlutd%2Frpx-2496ED" alt="Docker"></a>
  <a href="https://github.com/IRVLUTD/RPX"><img src="https://img.shields.io/badge/github-IRVLUTD%2FRPX-24292f" alt="GitHub"></a>
</p>

---

RPX gives you the data, the benchmark and the metrics from the paper. You supply a function that runs your
model; an adapter converts between it and RPX.

| Component | Purpose | Entry point |
|:--|:--|:--|
| Data | 100 real scenes in three phases, downloaded per task | `rpx.download_split` |
| Adapters | connect your model's inputs and outputs to RPX | `rpx.make_numpy_*_model` |
| Benchmark | the six paper tasks with the paper's protocol and metrics | `rpx.run_*` |
| Metrics | phase robustness Φ and worst-phase quality 𝒥<sub>min</sub> | `calculate` |

To add a task or a metric, see the toolkit docs on [tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/)
and [metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/).

## Getting started

Install the package:

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

Download the data for one task. This fetches only that task's files, about 160 MB for a 6-frame trial, and
needs no login:

```python
import numpy as np
import rpx_benchmark as rpx

manifest = rpx.download_split("monocular_depth", "easy", max_samples=6)
```

Wrap your model in the task's adapter and run the benchmark:

```python
def predict(rgb):                               # RPX passes an RGB image (H, W, 3)
    return np.ones(rgb.shape[:2], np.float32)   # return depth (H, W) in metres; replace with your model

model = rpx.make_numpy_depth_model(predict, name="my-model")
result, report, paths = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(
    model=model, manifest_path=str(manifest), output_dir="rpx_results/my-model",
))
print(result.aggregated["absrel"], result.aggregated["delta1"])
```

Φ and 𝒥<sub>min</sub> compare phases, so they need all three phases of many scenes. Drop `max_samples` above to
download the full tier, then:

```python
from rpx_benchmark.examples.summarize_metrics import calculate, read_records

summary = calculate(read_records(paths["json"]), ["absrel", "rmse", "silog", "delta1"])
```

The other tasks follow the same pattern:

| Task | Data: `download_split(…)` | Adapter | Benchmark | Paper metrics |
|:--|:--|:--|:--|:--|
| T1 Image depth | `"monocular_depth"` | `make_numpy_depth_model` | `run_monocular_depth` | AbsRel, RMSE, SILog, δ1 |
| T2 Video depth | `"video_depth"` | `make_numpy_video_depth_model` | `run_video_depth` | T1's, plus TGM and TGSE |
| T3 Tracking | `"object_tracking"` | `make_numpy_tracking_model` | `run_object_tracking` | HOTA, MOTA, IDSW |
| T4 Camera pose | `"relative_camera_pose"`, `manifest_name="relative_pose"` | `make_numpy_pose_model` | `run_relative_pose` | AUC at 5°, 10°, 20° |
| T5 VQA | `rpx.download("T5")` (packaged questions) | none; your function returns text | `rpx_benchmark.vqa.evaluate_vqa` | Acc@0.5, GIoU, center-in-GT |
| T6 In-context VQA | `rpx.download("T6")` | none | `rpx_benchmark.vqa.evaluate_vqa` | same as T5 |

Functions are under `rpx` unless written out in full. To try a task with a placeholder model before writing
any code, run `rpx.benchmark("T2", rpx.download("T2", max_samples=2))`.

<details>
<summary>Prefer the command line?</summary>

```bash
python -m rpx_benchmark.examples.benchmark_tasks --task T1 --split easy --max-samples 6 --model my_model:predict
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke   # check the install, no download
```

</details>

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
    # image_paths is (target,) or (reference_crop, target); return your model's raw text.
    ...

report = evaluate_vqa(
    "vqa_manifest.jsonl", predict_vqa, model_name="my-vlm", model_revision="v1",
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
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps \
  'git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a'
pytest tests -q
ruff check rpx_benchmark tests scripts && mypy rpx_benchmark && python -m build
```

The tests never download checkpoints. Real-model checks run through the per-image GPU gates in the
[Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md).

## Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and Kamath, Srinanditha and
            Allu, Sai Haneesh and Rangappa, Govind Tyagi and Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

The toolkit is MIT-licensed and the dataset is CC BY 4.0. Pretrained models keep their own licenses.

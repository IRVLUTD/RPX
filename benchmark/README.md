<p align="center">
  <img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires." width="100%"/>
</p>

<h3 align="center">Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild</h3>

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

## Get started in three steps

**1. Install** (Python 3.10–3.12; CPU is enough to start)

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

**2. Check the install**: all six evaluators on synthetic inputs, in a few seconds, with no download.

```bash
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke --output rpx-check
```

<details>
<summary>What you should see</summary>

```text
RPX smoke check: synthetic inputs, CPU, real evaluators
  T1  Image depth              1 samples   AbsRel 0.050 · δ1 1.000            → rpx-check/T1/results
  T2  Video depth              1 samples   AbsRel 0.026 · TGM 0.000           → rpx-check/T2/results
  T3  Object tracking          1 samples   MOTA 1.000 · IDSW 0.000            → rpx-check/T3/results
  T4  Relative camera pose     1 samples   Rot° 0.000 · Transl° 0.000         → rpx-check/T4/results
  T5  VQA grounding            1 samples   Acc@0.5 1.000 · GIoU 1.000         → rpx-check/T5/results
  T6  In-context VQA           1 samples   Acc@0.5 1.000 · GIoU 1.000         → rpx-check/T6/results
Done: 6 task(s) ran end to end; full results under rpx-check/.
```

</details>

**3. Score your model on real RPX scenes.** Swap in your own function; RPX loads the data and applies the paper's metrics.

```python
import numpy as np
import rpx_benchmark as rpx

# Your model: RGB image (H, W, 3) -> metric depth in metres (H, W).
def predict_depth(rgb):
    return np.ones(rgb.shape[:2], np.float32)

model = rpx.make_numpy_depth_model(predict_depth, name="my-depth")
result, report, paths = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(
    model=model, split="easy", repo_id="IRVLUTD/RPX",
    max_samples=6, output_dir="rpx_results/my-depth",
))
print(result.aggregated["absrel"], result.aggregated["delta1"])
```

The first run fetches about 160 MB, only the files this task needs, and finishes in under a minute. No Hugging Face
login is required.

## Get the paper's diagnosis

Remove `max_samples` to score the full Easy tier (33 scenes × 3 phases), then turn the per-sample results into Φ and
𝒥<sub>min</sub>:

```bash
python -m rpx_benchmark.examples.summarize_metrics \
  --input rpx_results/my-depth/result.json \
  --metrics absrel rmse silog delta1 --output rpx_results/my-depth/phi.json
# prints: my-depth / monocular_depth: Φ …   Jmin …
#         (33 scenes; J per phase: clutter …  interaction …  clean …)
```

Φ needs all three phases of many scenes, so a 6-frame trial cannot produce it.

## Tasks and reference models

Wrap your model with the task's factory and run it, or pull the Docker image that holds every model the paper
evaluated on that task.

| Task | Wrap your model with | Run with | Paper's models, `irvlutd/rpx:` |
|---|---|---|---|
| **T1** Image depth | `make_numpy_depth_model` | `run_monocular_depth` | `depth-zipdepth-latest` (10) |
| **T2** Video depth | `make_numpy_video_depth_model` | `run_video_depth` | `depth-zipdepth-latest` (8)<br>`depth-dvd-latest` (DVD) |
| **T3** Object tracking | `make_numpy_tracking_model` | `run_object_tracking` | `tracking-dam4sam-rpx-latest` (8) |
| **T4** Relative camera pose | `make_numpy_pose_model` | `run_relative_pose` | `rcpe-monst3r-rpx-latest` (10) |
| **T5** VQA grounding | a `predict_vqa` callable | `vqa.evaluate_vqa` | `vqa-vllm` (12) |
| **T6** In-context VQA | the same callable, two images | `vqa.evaluate_vqa` | `vqa-vllm` (12) |

- **Inputs, shapes and units** for each factory are in its docstring and the [model guide](https://irvlutd.github.io/RPX/toolkit-docs/models/). Each task
  also has an [illustrated walkthrough](https://irvlutd.github.io/RPX/toolkit-docs/benchmarks/).
- **Paper metrics**: the generic T3 and T4 runners report per-sample errors; the paper's HOTA and pooled pose AUC come
  from the clip- and pair-level reference runners in the [repository](https://github.com/IRVLUTD/RPX/tree/main/benchmark/scripts).
- **Docker**: always name a tag, since bare `irvlutd/rpx` is an unrelated legacy image. Box- and text-prompted
  trackers and GemDepth (build recipe only) are covered in the [Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md#which-image-to-pull).
- **More tasks**: segmentation, detection, open-vocabulary detection, grounding, sparse depth and keypoint matching
  runners are also included; `rpx_benchmark.tasks.available_tasks()` lists them.

## Custom VQA model or API

Any local model or hosted API works, as long as it returns text:

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

Return one JSON object with `label` and `bbox` (XYXY, normalized to 0–1000). Your model never sees the answers, and
unparseable replies count as misses. The [VQA guide](https://github.com/IRVLUTD/RPX/blob/main/docker/vqa-smoke/README.md) explains how to
prepare the manifest.

## What RPX reveals

Quality drops when hands enter the scene, and no model in the paper is both accurate and robust.

<p align="center">
  <img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-hero-light.webp" alt="RPX: the same real scene before, during and after manipulation and from an egocentric view; mean quality per view for video depth, tracking and VQA drops when hands enter the scene; none of the 65 evaluations in the paper reaches both worst-phase quality 0.75 and phase robustness 0.86; three steps: pip install rpx-benchmark, run your model, get phi and J-min." width="100%"/>
</p>

## Go further

| I want to… | Go to |
|---|---|
| Compute Φ and 𝒥<sub>min</sub> from results I already have | [Φ and JEDI calculator](https://irvlutd.github.io/RPX/toolkit-docs/analysis/) |
| Profile latency, throughput and memory | [Hardware profiler](https://irvlutd.github.io/RPX/toolkit-docs/profiling/) |
| Label my own scenes the way RPX was labelled | [Mask annotation pipeline](https://irvlutd.github.io/RPX/toolkit-docs/annotation/) |
| Add a metric or a new task | [Metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/) · [Tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) |
| Explore the data: 100 scenes, 70 objects, 133,121 frames | [Dataset card](https://huggingface.co/datasets/IRVLUTD/RPX) · [Data guide](https://irvlutd.github.io/RPX/toolkit-docs/data/) |

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

The tests never download checkpoints; real-model checks run through the per-image GPU gates in the
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

Toolkit code: MIT. Dataset: CC BY 4.0. Pretrained models keep their own licenses.

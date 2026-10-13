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

## What you get

| Piece | What it gives you | Call |
|:--|:--|:--|
| 📦 **Data** | 100 real scenes in three phases, downloaded per task | `rpx.download` |
| 🧪 **Benchmark** | the six paper tasks with the paper's protocol and metrics | `rpx.benchmark` |
| 🧮 **Metrics calculator** | Φ (phase robustness) and 𝒥<sub>min</sub> (worst-phase quality) | `summarize_metrics` |

**Your part is one adapter**: a function that turns RPX's input into your model's input, and your model's output
into RPX's format. Adding a new task or metric? See the toolkit docs on [tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) and
[metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/).

## Get started

**1 · Install**

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

**2 · Wrap your model in an adapter.** For T1 image depth:

```python
import numpy as np

def my_model(x):                              # stand-in for your network
    return np.ones(x.shape[-2:], np.float32)

def predict(rgb):                             # RPX gives: RGB image (H, W, 3), uint8
    x = rgb.transpose(2, 0, 1)[None] / 255.0  # input adapter: RPX -> your model
    depth = my_model(x)
    return np.asarray(depth, np.float32)      # output adapter: your model -> depth (H, W) in metres
```

The [dummy models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/examples/dummy_models.py) show what each task's adapter receives and returns.

**3 · Download the data and run the benchmark**

```python
import rpx_benchmark as rpx

data = rpx.download("T1", split="easy", max_samples=6)  # only this task's files, ~160 MB
result = rpx.benchmark("T1", data, predict, name="my-model")
```

**4 · Calculate Φ and 𝒥<sub>min</sub>.** These need all three phases of many scenes, so drop max_samples to download
the full tier first.

```bash
python -m rpx_benchmark.examples.summarize_metrics --input rpx_results/T1/my-model/result.json \
  --metrics absrel rmse silog delta1 --output rpx_results/T1/my-model/phi.json
```

Every paper task works the same way: **T1** image depth · **T2** video depth · **T3** tracking · **T4** camera pose ·
**T5** VQA grounding · **T6** in-context VQA. The paper's metric list for each task is in the
[calculator guide](https://irvlutd.github.io/RPX/toolkit-docs/analysis/).

<details>
<summary>Prefer the command line?</summary>

```bash
python -m rpx_benchmark.examples.benchmark_tasks --task T1 --split easy --max-samples 6 --model my_model:predict
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke   # check the install, no download
```

</details>

## Tasks and reference models

Every task uses the same two calls; only the task ID and your function change. The Docker column holds every model
the paper evaluated on that task.

| Task | ID | Your function receives → returns | Paper's models, `irvlutd/rpx:` |
|:--|:-:|:--|:--|
| Image depth | `T1` | RGB `(H, W, 3)` → depth `(H, W)` in metres | `depth-zipdepth-latest` (10) |
| Video depth | `T2` | clip `(T, H, W, 3)` → depth `(T, H, W)` | `depth-zipdepth-latest` (8)<br>`depth-dvd-latest` (DVD) |
| Object tracking | `T3` | frame `(H, W, 3)` → `[{"track_id", "boxes"}]` | `tracking-dam4sam-rpx-latest` (8) |
| Relative camera pose | `T4` | `(rgb_a, rgb_b)` → `{"rotation", "translation"}` | `rcpe-monst3r-rpx-latest` (10) |
| VQA grounding | `T5` | image path + question → JSON `{"label", "bbox"}` | `vqa-vllm` (12) |
| In-context VQA | `T6` | reference crop + image + question → same | `vqa-vllm` (12) |

- **Details**: relative-depth models pass `depth_output_kind="relative"` to `rpx.benchmark`; full input contracts are in the
  [model guide](https://irvlutd.github.io/RPX/toolkit-docs/models/), with an
  [illustrated walkthrough](https://irvlutd.github.io/RPX/toolkit-docs/benchmarks/) per task.
- **T5 / T6** download a packaged set of questions; pass your own VQA question list file as `data` instead,
  and see [custom VQA model or API](#custom-vqa-model-or-api) for API-backed models.
- **Docker**: always name a tag, since bare `irvlutd/rpx` is an unrelated legacy image. Box- and text-prompted
  trackers and GemDepth (build recipe only) are covered in the [Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md#which-image-to-pull).
- **Paper metrics**: the generic T3 and T4 runners report per-sample errors; the paper's HOTA and pooled pose AUC
  come from the clip- and pair-level reference runners in the [repository](https://github.com/IRVLUTD/RPX/tree/main/benchmark/scripts).

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

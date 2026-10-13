<p align="center">
  <img src="benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires." width="100%">
</p>

<h3 align="center">Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild</h3>

<p align="center">
  Score any perception model on 100 real scenes recorded <b>before, during and after manipulation</b>,<br/>
  and see where it breaks with two numbers: <b>Φ</b> (phase robustness) and <b>𝒥<sub>min</sub></b> (worst-phase quality).
</p>

<p align="center">
  <a href="https://pypi.org/project/rpx-benchmark/"><img src="https://img.shields.io/pypi/v/rpx-benchmark?color=4F46E5" alt="PyPI"></a>
  <a href="https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img src="https://img.shields.io/badge/docs-toolkit-0d9488" alt="Docs"></a>
  <a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-IRVLUTD%2FRPX-yellow" alt="Dataset"></a>
  <a href="https://hub.docker.com/r/irvlutd/rpx"><img src="https://img.shields.io/badge/docker-irvlutd%2Frpx-2496ED" alt="Docker"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#get-started-in-three-steps">Get started</a> ·
  <a href="#pick-your-task">Tasks and models</a> ·
  <a href="#what-rpx-reveals">Findings</a> ·
  <a href="#the-dataset">Dataset</a> ·
  <a href="#citation">Cite</a>
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

The first run fetches about 160 MB, only the files this task needs, and finishes in under a minute. No
Hugging Face login is required.

> [!TIP]
> **Next:** remove `max_samples` to score the full Easy tier (33 scenes × 3 phases), then
> [compute Φ and 𝒥<sub>min</sub>](benchmark/README.md#get-the-papers-diagnosis) from the results.
> Φ needs all three phases of many scenes, so a 6-frame trial cannot produce it.

## Pick your task

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

- **Inputs, shapes and units** for each factory: [model guide](https://irvlutd.github.io/RPX/toolkit-docs/models/). Each task also has an
  [illustrated walkthrough](https://irvlutd.github.io/RPX/toolkit-docs/benchmarks/).
- **VLMs and APIs** (T5, T6): `vqa` is `rpx_benchmark.vqa`; see [custom VQA model or API](benchmark/README.md#custom-vqa-model-or-api).
- **Docker**: always name a tag, since bare `irvlutd/rpx` is an unrelated legacy image. Box- and text-prompted
  trackers and GemDepth (build recipe only) are covered in the [Docker guide](docker/README.md#which-image-to-pull).

## What RPX reveals

Quality drops when hands enter the scene, and no model in the paper is both accurate and robust.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="benchmark/rpx_benchmark/guides/assets/rpx-hero-dark.webp">
    <img src="benchmark/rpx_benchmark/guides/assets/rpx-hero-light.webp" alt="RPX: the same real scene before, during and after manipulation and from an egocentric view; mean quality per view for video depth, tracking and VQA drops when hands enter the scene; none of the 65 evaluations in the paper reaches both worst-phase quality 0.75 and phase robustness 0.86; three steps: pip install rpx-benchmark, run your model, get phi and J-min." width="100%">
  </picture>
</p>

## Go further

| I want to… | Go to |
|---|---|
| Compute Φ and 𝒥<sub>min</sub> from results I already have | [Φ and JEDI calculator](https://irvlutd.github.io/RPX/toolkit-docs/analysis/) |
| Profile latency, throughput and memory | [Hardware profiler](https://irvlutd.github.io/RPX/toolkit-docs/profiling/) |
| Label my own scenes the way RPX was labelled | [Mask annotation pipeline](https://irvlutd.github.io/RPX/toolkit-docs/annotation/) |
| Add a metric or a new task | [Metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/) · [Tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) |
| See everything else the toolkit can do | [All capabilities](https://irvlutd.github.io/RPX/toolkit-docs/capabilities/) |

## The dataset

- **Scenes:** 100 multi-object scenes (50 indoor, 50 outdoor) in Easy, Medium and Hard tiers of 33, 33 and 34
- **Phases:** Clutter (before), Interaction (during) and Clean (after) manipulation, plus an egocentric view during Interaction
- **Objects:** 70 everyday objects, each also captured alone in 360°
- **Sensors:** D435 RGB-D, T265 fisheye stereo and 6-DoF pose, head-mounted GoPro
- **Labels:** instance masks with one identity per object across phases and views, VQA grounding, object questionnaires
- **Size:** 133,121 frames; the toolkit downloads only what a task needs

Hosted on [Hugging Face](https://huggingface.co/datasets/IRVLUTD/RPX) under CC BY 4.0. File formats and loaders are in
the [data guide](https://irvlutd.github.io/RPX/toolkit-docs/data/).

<details>
<summary><b>Terms used in RPX</b></summary>

| Term | Meaning |
|---|---|
| **Phase** | One recording of a scene: Clutter (before manipulation), Interaction (hands moving objects) or Clean (after). The egocentric view is recorded during Interaction |
| **MOS / SOS / Ego** | Multi-object scenes; single-object scans (each object alone, 360°); the head-mounted egocentric stream |
| **𝒥 and 𝒥<sub>min</sub>** | Quality on a common 0–1 scale: each metric is mapped to a desirability, combined per scene and phase, and averaged. 𝒥<sub>min</sub> is the worst phase's quality |
| **Φ** | Phase robustness, 1 minus the phase effect size from a repeated-measures analysis. High Φ means a model behaves the same across phases, not that it is accurate |
| **Easy / Medium / Hard** | Difficulty tiers, stratified by how much correction the scene's masks needed |
| **Manifest** | A JSON list of the samples a task evaluates; the loader reads it, you rarely write one |
| **Revision** | The dataset commit a run used. The toolkit pins one by default so runs stay comparable |

</details>

<details>
<summary><b>Repository layout</b></summary>

| Folder | What it holds |
|---|---|
| [`benchmark/`](benchmark/README.md) | The `rpx-benchmark` package: loaders, task runners, metrics, Φ/JEDI, examples and tests. **Start here** |
| [`docker/`](docker/README.md) | Reproducible environments for every reference model |
| [`data/capture/`](data/capture/README.md) | Sensor capture tools (D435 + T265, GoPro) |
| [`data/mask_annotation/`](data/mask_annotation/README.md) | SAM 2 mask propagation with human correction, grounding and VQA annotation |
| [`tracking/metadata/`](tracking/metadata/) | Versioned text prompts for text-initialized tracking |
| [`tools/`](tools/) | Dataset release, QA and documentation builders |

</details>

## Contributing

Issues and pull requests are welcome. Before submitting, run the [test suite](benchmark/README.md#testing); it never
downloads checkpoints. Real-model checks run through the GPU gates in [`docker/`](docker/README.md).

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

Code is MIT ([LICENSE](LICENSE)); the dataset is CC BY 4.0. Pretrained models keep their own licenses. GitHub's
*Cite this repository* button uses [CITATION.cff](CITATION.cff).

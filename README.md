<div align="center">

# 🤖 RPX

### Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild

<br>

<a href="https://jishnujayakumar.github.io">Jishnu Jaykumar P</a><sup>&ast;</sup> &nbsp;·&nbsp;
<a href="https://github.com/Optimax14">Itay Kadosh</a><sup>&ast;</sup> &nbsp;·&nbsp;
<a href="https://narendhiranv04.github.io">Narendhiran Vijayakumar</a><sup>&ast;</sup> &nbsp;·&nbsp;
<a href="https://github.com/srinandithak">Srinanditha Kamath</a> &nbsp;·&nbsp;
<a href="https://saihaneeshallu.github.io/">Sai Haneesh Allu</a>
<br>
<a href="https://github.com/GovindRangappa">Govind Tyagi Rangappa</a> &nbsp;·&nbsp;
<a href="https://github.com/AnimeshMaheshwari22">Animesh Maheshwari</a> &nbsp;·&nbsp;
<a href="https://jwroboticsvision.github.io/">Jikai Wang</a> &nbsp;·&nbsp;
<a href="https://yuxng.github.io/">Yu Xiang</a>

<sub><sup>&ast;</sup>Equal contribution · UT Dallas · UF Gainesville · IIIT Hyderabad · UT Austin</sub>

<br><br>

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-IRVLUTD%2FRPX-ffcc4d?style=for-the-badge)](https://huggingface.co/datasets/IRVLUTD/RPX)
[![Docs](https://img.shields.io/badge/Docs-Toolkit-0d9488?style=for-the-badge)](https://irvlutd.github.io/RPX/toolkit-docs/)
[![PyPI](https://img.shields.io/pypi/v/rpx-benchmark?style=for-the-badge&color=4F46E5&label=PyPI)](https://pypi.org/project/rpx-benchmark/)
[![Docker](https://img.shields.io/badge/Docker-irvlutd%2Frpx-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/irvlutd/rpx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br>

<img src="benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" width="95%" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires.">
<br>
<sub><i>Today each perception task is benchmarked on its own data. RPX records the same real scenes before (Clutter), during (Interaction) and after (Clean) manipulation, plus an egocentric view, and scores six tasks on them with two diagnostics: phase robustness Φ and worst-phase quality 𝒥<sub>min</sub>.</i></sub>

</div>

<br>

**RPX** tells you which perception model **keeps working while a robot changes the scene**: the same 100 real scenes,
recorded before, during and after manipulation, scored on six tasks.

<br>

<div align="center">

| 🎯 Accurate *and* robust | 📉 Video depth while acting | 🔁 Tracker identity switches | 💬 Best in-context grounding |
|:-:|:-:|:-:|:-:|
| **0 / 65**<br><sub>evaluations in the golden zone</sub> | **2–4×**<br><sub>near-range error during Interaction</sub> | **5–11×**<br><sub>more during Interaction</sub> | **𝒥<sub>min</sub> = 0.23**<br><sub>across 12 VLM configurations</sub> |

</div>

<br>

<br>

## 🚀 Get started

```bash
pip install 'rpx-benchmark[hub,schemas]'
python -m rpx_benchmark.examples.benchmark_tasks --task T1 --split easy --max-samples 6
```

```text
Downloading Image depth data (easy, 6 samples) ...
Scoring Image depth ...
  T1  Image depth              6 samples   AbsRel 0.295 · δ1 0.440            → rpx_results/T1
```

That downloads one task's data (only what it needs), then scores a demo model on it. No GPU, login or code needed.
Swap `T1` for `T2` video depth, `T3` tracking or `T4` camera pose.

**Score your own model:** put one function in `my_model.py` and add `--model`.

```python
# my_model.py  (T1: RGB image (H, W, 3) -> metric depth in metres (H, W))
import numpy as np

def predict(rgb):
    return np.ones(rgb.shape[:2], np.float32)  # replace with your model
```

```bash
python -m rpx_benchmark.examples.benchmark_tasks --task T1 --split easy --max-samples 6 --model my_model:predict
```

<br>

## 📚 Learn more

- 📘 [**Toolkit docs**](https://irvlutd.github.io/RPX/toolkit-docs/): what `predict` takes for each task, the full protocol, Φ and 𝒥<sub>min</sub>, metrics, annotation tools
- 🤗 [**Dataset**](https://huggingface.co/datasets/IRVLUTD/RPX): 100 scenes, 70 objects, 133K frames, CC BY 4.0
- 🐳 [**Paper's models**](docker/README.md): ready-to-run Docker images for the models evaluated in the paper
- 🐍 [**Python API**](benchmark/README.md): the `rpx-benchmark` package, also on [PyPI](https://pypi.org/project/rpx-benchmark/)

<br>

## 📝 Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran and Kamath, Srinanditha and
            Allu, Sai Haneesh and Rangappa, Govind Tyagi and Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

<br>

<div align="center">
<sub>
Code: <a href="LICENSE">MIT</a> · Dataset: CC BY 4.0 ·
Questions: <a href="https://github.com/IRVLUTD/RPX/issues">open an issue</a> or reach
<a href="https://jishnujayakumar.github.io/">Jishnu</a>,
<a href="https://github.com/Optimax14">Itay</a> or
<a href="https://narendhiranv04.github.io">Naren</a>
<br><br>
Built at the <a href="https://labs.utdallas.edu/irvl/">Intelligent Robotics and Vision Lab</a>, The University of Texas at Dallas
</sub>
</div>

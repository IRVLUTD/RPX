<div align="center">

# 🤖 RPX

### Same Scene, Different Story:<br> Evaluating Robot Perception Across Scene Phases in the Wild

<br>

<a href="https://jishnujayakumar.github.io">Jishnu Jaykumar P</a><sup>1&ast;</sup> &nbsp;·&nbsp;
<a href="https://github.com/Optimax14">Itay Kadosh</a><sup>2&ast;</sup> &nbsp;·&nbsp;
<a href="https://narendhiranv04.github.io">Narendhiran Vijayakumar</a><sup>3&ast;</sup> &nbsp;·&nbsp;
<a href="https://github.com/srinandithak">Srinanditha Kamath</a><sup>4</sup> &nbsp;·&nbsp;
<a href="https://saihaneeshallu.github.io/">Sai Haneesh Allu</a><sup>1</sup>
<br>
<a href="https://github.com/GovindRangappa">Govind Tyagi Rangappa</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://github.com/AnimeshMaheshwari22">Animesh Maheshwari</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://jwroboticsvision.github.io/">Jikai Wang</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://yuxng.github.io/">Yu Xiang</a><sup>1</sup>

<sup>1</sup>UT Dallas &nbsp;&nbsp; <sup>2</sup>UF &nbsp;&nbsp; <sup>3</sup>IIIT Hyderabad &nbsp;&nbsp; <sup>4</sup>UT Austin
<br>
<sub><sup>&ast;</sup>Equal contribution</sub>

<p>
  <a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-IRVLUTD%2FRPX-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=white"></a>&nbsp;
  <a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img alt="Docs" src="https://img.shields.io/badge/Docs-Toolkit-0d9488?style=for-the-badge&logo=materialformkdocs&logoColor=white"></a>&nbsp;
  <a href="https://pypi.org/project/rpx-benchmark/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rpx-benchmark?style=for-the-badge&color=4F46E5&label=PyPI&logo=pypi&logoColor=white"></a>&nbsp;
  <a href="https://hub.docker.com/r/irvlutd/rpx"><img alt="Docker" src="https://img.shields.io/badge/Docker-irvlutd%2Frpx-2496ED?style=for-the-badge&logo=docker&logoColor=white"></a>&nbsp;
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-3da639?style=for-the-badge&logo=opensourceinitiative&logoColor=white"></a>
</p>

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

## 🧰 What you get

| Piece | What it gives you | Function |
|:--|:--|:--|
| 📦 **Data** | 100 real scenes in three phases, downloaded per task | `rpx.download_split` |
| 🔌 **Adapters** | connect your model's inputs and outputs to RPX | `rpx.make_numpy_*_model` |
| 🧪 **Benchmark** | the six paper tasks with the paper's protocol and metrics | `rpx.run_*` |
| 🧮 **Metrics calculator** | Φ (phase robustness) and 𝒥<sub>min</sub> (worst-phase quality) | `calculate` |

**Your part is one function** that runs your model; the adapter handles RPX's inputs and outputs around it. Adding a
new task or metric? See the toolkit docs on [tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) and [metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/).

<br>

## 🚀 Get started

**1 · Install**

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

**2 · Download one task's data** (only its files: about 160 MB for this 6-frame trial, no login)

```python
import numpy as np
import rpx_benchmark as rpx

manifest = rpx.download_split("monocular_depth", "easy", max_samples=6)
```

**3 · Wrap your model in the task's adapter and run the benchmark**

```python
def predict(rgb):                               # RPX passes an RGB image (H, W, 3)
    return np.ones(rgb.shape[:2], np.float32)   # return depth (H, W) in metres; replace with your model

model = rpx.make_numpy_depth_model(predict, name="my-model")
result, report, paths = rpx.run_monocular_depth(rpx.MonocularDepthRunConfig(
    model=model, manifest_path=str(manifest), output_dir="rpx_results/my-model",
))
print(result.aggregated["absrel"], result.aggregated["delta1"])
```

**4 · Calculate Φ and 𝒥<sub>min</sub>.** These need all three phases of many scenes: drop max_samples in step 2 to get
the full tier, then

```python
from rpx_benchmark.examples.summarize_metrics import calculate, read_records

summary = calculate(read_records(paths["json"]), ["absrel", "rmse", "silog", "delta1"])
```

| Task | Data: `download_split(…)` | Adapter | Benchmark | Paper metrics for the calculator |
|:--|:--|:--|:--|:--|
| **T1** Image depth | `"monocular_depth"` | `make_numpy_depth_model` | `run_monocular_depth` | absrel · rmse · silog · delta1 |
| **T2** Video depth | `"video_depth"` | `make_numpy_video_depth_model` | `run_video_depth` | T1's + tgm · tgse |
| **T3** Tracking | `"object_tracking"` | `make_numpy_tracking_model` | `run_object_tracking` | HOTA · MOTA · IDSW |
| **T4** Camera pose | `"relative_camera_pose"`, `manifest_name="relative_pose"` | `make_numpy_pose_model` | `run_relative_pose` | AUC @ 5° · 10° · 20° |
| **T5** VQA | `rpx.download("T5")` (packaged questions) | none: your function returns text | `rpx_benchmark.vqa.evaluate_vqa` | Acc@0.5 · GIoU · center-in-GT |
| **T6** In-context VQA | `rpx.download("T6")` | none | `rpx_benchmark.vqa.evaluate_vqa` | same as T5 |

All functions are under `rpx` unless named in full. The [dummy models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/examples/dummy_models.py) show what each task's function
receives and returns. To try any task with a ready-made dummy model before writing
code, `rpx.benchmark("T2", rpx.download("T2", max_samples=2))` runs steps 2–3 in one go.

<br>

## 📚 Learn more

- 📘 [**Toolkit docs**](https://irvlutd.github.io/RPX/toolkit-docs/): what predict takes for each task, the full protocol, Φ and 𝒥<sub>min</sub>, metrics, annotation tools
- 🤗 [**Dataset**](https://huggingface.co/datasets/IRVLUTD/RPX): 100 scenes, 70 objects, 133K frames, CC BY 4.0
- 🐳 [**Paper's models**](docker/README.md): ready-to-run Docker images for the models evaluated in the paper
- 🐍 [**Python API**](benchmark/README.md): the rpx-benchmark package, also on [PyPI](https://pypi.org/project/rpx-benchmark/)

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

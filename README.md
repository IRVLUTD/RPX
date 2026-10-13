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

## 🚀 Get started

**1 · Install**

```bash
pip install 'rpx-benchmark[hub,schemas]'
```

**2 · Download a task's data.** Only the files that task needs: about 160 MB here, no login.

```python
import rpx_benchmark as rpx

data = rpx.download("T1", split="easy", max_samples=6)
```

**3 · Run the benchmark** with the paper's metrics.

```python
result = rpx.benchmark("T1", data, rpx.dummy_models.image_depth)
```

To score your model, pass your own function in place of the dummy one. The [dummy models](https://github.com/IRVLUTD/RPX/blob/main/benchmark/rpx_benchmark/examples/dummy_models.py) show what each
task's function receives and returns. Every paper task works the same way: **T1** image depth · **T2** video depth ·
**T3** tracking · **T4** camera pose · **T5** VQA grounding · **T6** in-context VQA. Results are saved in rpx_results.

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

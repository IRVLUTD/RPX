<div align="center">

# RPX

### Same Scene, Different Story:<br>Evaluating Robot Perception Across Scene Phases in the Wild

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

<br>

<a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img alt="Dataset on Hugging Face" src="https://img.shields.io/badge/dataset-IRVLUTD%2FRPX-ffcc4d?logo=huggingface&logoColor=white"></a>
<a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img alt="Toolkit docs" src="https://img.shields.io/badge/docs-toolkit-0d9488?logo=materialformkdocs&logoColor=white"></a>
<a href="https://pypi.org/project/rpx-benchmark/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/rpx-benchmark?label=pypi&color=4F46E5&logo=pypi&logoColor=white"></a>
<a href="https://hub.docker.com/r/irvlutd/rpx"><img alt="Docker images" src="https://img.shields.io/badge/docker-irvlutd%2Frpx-2496ED?logo=docker&logoColor=white"></a>
<a href="https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests&logo=github"></a>
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-3da639"></a>

<br>

<img src="benchmark/rpx_benchmark/guides/assets/rpx-overview.webp" width="95%" alt="RPX overview: today each perception task is benchmarked on its own data; RPX captures every real-world scene four ways (Clutter, Interaction, Clean and an egocentric view) and scores six tasks on them (image depth, video depth, tracking, relative camera pose, general VQA, in-context VQA) with two diagnostics, phase robustness and worst-phase quality; 100 multi-object scenes, 70 single objects scanned in 360 degrees, and a D435 RGB-D, T265 stereo and pose, and GoPro egocentric rig with masks and object questionnaires.">

<sub>The same real scenes before, during and after manipulation, scored on six perception tasks.</sub>

</div>

<br>

RPX is a benchmark and toolkit for robot perception in scenes that a robot is changing. We recorded
100 real scenes before (Clutter), during (Interaction) and after (Clean) manipulation, plus an
egocentric view, and score six tasks on the same frames. Two diagnostics summarize every model:
phase robustness Φ and worst-phase quality 𝒥<sub>min</sub>.

<div align="center">

| **0 of 65** | **2–4×** | **5–11×** | **0.23** |
|:-:|:-:|:-:|:-:|
| evaluations both accurate<br>and robust across phases | near-range video depth<br>error during Interaction | more tracker identity<br>switches during Interaction | best worst-phase score<br>for in-context grounding |

<sub>Results across 49 models and 65 evaluations in the paper.</sub>

<br>

[Citation](#citation) &nbsp;·&nbsp;
[Getting started](#getting-started) &nbsp;·&nbsp;
[Tasks](#tasks) &nbsp;·&nbsp;
[Repository layout](#repository-layout) &nbsp;·&nbsp;
[Documentation](#documentation)

</div>

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
[`dummy_models.py`](benchmark/rpx_benchmark/examples/dummy_models.py) shows the inputs and
outputs of every task.</sub>

To add a task or a metric, see the toolkit docs on
[tasks](https://irvlutd.github.io/RPX/toolkit-docs/tasks/) and
[metrics](https://irvlutd.github.io/RPX/toolkit-docs/metrics/).

## Repository layout

| Folder | Contents |
|:--|:--|
| [`benchmark/`](benchmark/README.md) | the `rpx-benchmark` package, its tests and the docs source |
| [`docker/`](docker/README.md) | images with the models evaluated in the paper |
| [`data/capture/`](data/capture/README.md) | capture rig and recording scripts |
| [`data/mask_annotation/`](data/mask_annotation/README.md) | the instance-mask annotation pipeline |
| [`tracking/`](tracking/README.md) | text prompts for text-prompted trackers (T3) |
| [`tools/`](tools/dataset/README.md) | dataset card, dataset QA, docs and figure builders |

## Documentation

- [Toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/): model contracts per task, the evaluation protocol, Φ and 𝒥<sub>min</sub>, annotation tools
- [Dataset](https://huggingface.co/datasets/IRVLUTD/RPX): 100 scenes, 70 objects and 133K frames on Hugging Face
- [Docker images](docker/README.md): which image holds which of the paper's models
- [Python package](benchmark/README.md): the `rpx-benchmark` API, also on [PyPI](https://pypi.org/project/rpx-benchmark/)

---

<div align="center">
<sub>
Code under the <a href="LICENSE">MIT License</a>, dataset under CC BY 4.0.
Questions: <a href="https://github.com/IRVLUTD/RPX/issues">open an issue</a> or contact
<a href="https://jishnujayakumar.github.io/">Jishnu</a>,
<a href="https://github.com/Optimax14">Itay</a> or
<a href="https://narendhiranv04.github.io">Naren</a>.
<br>
Built at the <a href="https://labs.utdallas.edu/irvl/">Intelligent Robotics and Vision Lab</a>,
The University of Texas at Dallas.
</sub>
</div>

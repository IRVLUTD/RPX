<div align="center">

# RPX

### Same Scene, Different Story:<br> Evaluating Robot Perception Across Scene Phases in the Wild

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

RPX measures how well robot perception models hold up while a robot changes the scene. We recorded 100 real
scenes before, during and after manipulation, and score six tasks on the same frames, so a model's drop from
one phase to the next is directly visible.

Across the 49 models we evaluated:

- None of the 65 evaluations is both accurate and robust across phases.
- Video depth error at close range grows 2–4× while the scene is being manipulated.
- Trackers switch identities 5–11× more often during interaction.
- The best of 12 VLM configurations for in-context grounding reaches a worst-phase score of only 𝒥<sub>min</sub> = 0.23.

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

## What's in this repository

The `rpx-benchmark` package gives you the data, the benchmark and the metrics from the paper. You supply a
function that runs your model; an adapter converts between it and RPX.

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

Functions are under `rpx` unless written out in full.
[`dummy_models.py`](benchmark/rpx_benchmark/examples/dummy_models.py) shows what each task passes to your
function and expects back. To try a task with a placeholder model before writing any code, run
`rpx.benchmark("T2", rpx.download("T2", max_samples=2))`.

## Documentation

- [Toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/): per-task model contracts, the evaluation protocol, Φ and 𝒥<sub>min</sub>, and the annotation tools
- [Dataset](https://huggingface.co/datasets/IRVLUTD/RPX) on Hugging Face: 100 scenes, 70 objects, 133K frames
- [Docker images](docker/README.md) for the models evaluated in the paper
- [Python package](benchmark/README.md), also on [PyPI](https://pypi.org/project/rpx-benchmark/)

## License and contact

The code is released under the [MIT License](LICENSE) and the dataset under CC BY 4.0. For questions, open an
[issue](https://github.com/IRVLUTD/RPX/issues) or contact [Jishnu](https://jishnujayakumar.github.io/),
[Itay](https://github.com/Optimax14) or [Naren](https://narendhiranv04.github.io).

RPX was built at the [Intelligent Robotics and Vision Lab](https://labs.utdallas.edu/irvl/), The University of
Texas at Dallas.

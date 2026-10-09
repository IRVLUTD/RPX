<p align="center">
  <img src="https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-hero.jpg" alt="RPX: the same real scene before, during and after manipulation and from an egocentric view; mean quality per view for video depth, tracking and VQA drops when hands enter the scene; none of the 65 evaluations in the paper reaches both worst-phase quality 0.75 and phase robustness 0.86; three steps: pip install rpx-benchmark, run your model, get phi and J-min." width="100%"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/rpx-benchmark/"><img src="https://img.shields.io/pypi/v/rpx-benchmark?color=4F46E5" alt="PyPI"></a>
  <a href="https://pypi.org/project/rpx-benchmark/"><img src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue" alt="Python 3.10–3.12"></a>
  <a href="https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://irvlutd.github.io/RPX/toolkit-docs/"><img src="https://img.shields.io/badge/docs-toolkit-0d9488" alt="Docs"></a>
  <a href="https://huggingface.co/datasets/IRVLUTD/RPX"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-IRVLUTD%2FRPX-yellow" alt="Dataset"></a>
  <a href="https://hub.docker.com/r/irvlutd/rpx"><img src="https://img.shields.io/badge/docker-irvlutd%2Frpx-2496ED" alt="Docker"></a>
  <a href="https://github.com/IRVLUTD/RPX/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  <b>Score any perception model on the same real scenes, before, during and after manipulation.</b><br/>
  Six tasks, one evaluator, and the paper's robustness diagnosis, in a few lines of Python.
</p>

---

## Why RPX?

| The usual way | With RPX |
|---|---|
| Each task is benchmarked on its own static images | Six tasks are scored on the **same 100 real scenes** |
| A scene is recorded once, in one state | Every scene is recorded **before, during and after** manipulation, plus an **egocentric** view |
| One accuracy number hides where a model breaks | **Φ** (phase robustness) and **𝒥<sub>min</sub>** (worst-phase quality) show it |
| Comparing a new model means re-implementing protocols | Wrap your function or API; RPX applies the paper's data, splits and metrics |

---

## Quick start

```bash
pip install 'rpx-benchmark[hub,schemas]'
python -m rpx_benchmark.examples.benchmark_tasks --task all --smoke --output rpx-check
```

Runs all six evaluators on tiny synthetic inputs in about a second, on CPU:

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

### Score your own model on real RPX scenes

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

The first run downloads about 160 MB and takes under a minute. Only the files this task needs are fetched, never the whole
237 GB dataset. Remove `max_samples` to score the full Easy tier (33 scenes × 3 phases).

### Get the paper's diagnosis

Run a full tier, then turn its per-sample results into Φ and 𝒥<sub>min</sub>:

```bash
python -m rpx_benchmark.examples.summarize_metrics \
  --input rpx_results/my-depth/result.json \
  --metrics absrel rmse silog delta1 --output rpx_results/my-depth/phi.json
# prints:  my-depth / monocular_depth: Φ …   Jmin …
#          (33 scenes; J per phase: clutter …  interaction …  clean …)
```

Φ needs all three phases of many scenes, so it is not available from a 6-frame trial run.

---

## Six tasks

| | Task | Wrap your model with | Run it with | Paper metrics |
|---|---|---|---|---|
| T1 | Image depth | `make_numpy_depth_model` | `run_monocular_depth` | AbsRel · RMSE · SILog · δ<sub>1</sub> |
| T2 | Video depth | `make_numpy_video_depth_model` | `run_video_depth` | T1 metrics + TGM · TGSE |
| T3 | Object tracking | `make_numpy_tracking_model` | `run_object_tracking` | HOTA · MOTA · IDSW |
| T4 | Relative camera pose | `make_numpy_pose_model` | `run_relative_pose` | AUC@5° · @10° · @20° |
| T5 | VQA grounding | a `predict_vqa` callable | `rpx_benchmark.vqa.evaluate_vqa` | Acc@0.5 · GIoU · Center-in-GT |
| T6 | In-context VQA | the same callable, two images | `rpx_benchmark.vqa.evaluate_vqa` | Acc@0.5 · GIoU · Center-in-GT |

Everything is under `rpx` unless named in full. The generic T3 and T4 runners report per-sample tracking and pose
errors; the paper's HOTA and pooled AUC come from the clip- and pair-level reference runners in the
[repository](https://github.com/IRVLUTD/RPX/tree/main/benchmark/scripts). Each task has an illustrated walkthrough in
the [benchmark samples](https://irvlutd.github.io/RPX/toolkit-docs/benchmarks/).

## Reference models

Every model in the paper except GemDepth (build recipe only) ships in a ready-to-pull Docker image. Pull the final
image for your task; it already contains every earlier model of that task.

| Task | Image | Models |
|---|---|---|
| T1, T2 | `irvlutd/rpx:depth-zipdepth-latest` | all 10 image-depth and 8 video-depth models |
| T2 | `irvlutd/rpx:depth-dvd-latest` | DVD |
| T3 | `irvlutd/rpx:tracking-dam4sam-rpx-latest` | 8 mask-initialized trackers (box- and text-prompted runs use their own images) |
| T4 | `irvlutd/rpx:rcpe-monst3r-rpx-latest` | all 10 pose models |
| T5, T6 | `irvlutd/rpx:vqa-vllm` | all 12 vision-language models |

The [Docker guide](https://github.com/IRVLUTD/RPX/blob/main/docker/README.md#which-image-to-pull) lists every image and
model, and how to run each one.

## Bring your own model

A model is any Python function, local or a call to an inference API. Wrap it with the task's factory (table above);
RPX loads the inputs, calls your function, and scores the outputs exactly as in the paper. Shapes and units are in
each factory's docstring and in the [model guide](https://irvlutd.github.io/RPX/toolkit-docs/models/). The toolkit
also includes segmentation, detection, open-vocabulary detection, grounding, sparse-depth and keypoint-matching
runners: `rpx_benchmark.tasks.available_tasks()` lists them.

## Custom VQA model or API

```python
from rpx_benchmark.vqa import evaluate_vqa

def predict_vqa(image_paths, prompt, max_new_tokens, output_kind):
    ...  # image_paths is (target,) or (reference_crop, target); return your model's raw text

report = evaluate_vqa(
    "vqa_manifest.jsonl", predict_vqa, model_name="my-vlm", model_revision="v1",
    image_cache=".cache/rpx-vqa", output_dir="rpx_results/my-vlm",
)
print(report["metrics"]["bbox_accuracy_at_0_5"])
```

Return one JSON object with `label` and `bbox` (XYXY, normalized to 0–1000). The evaluator never shows your model the
answers; parse failures count as misses. The [VQA guide](https://github.com/IRVLUTD/RPX/blob/main/docker/vqa-smoke/README.md)
explains manifest preparation.

## More tools

- **[Mask annotation pipeline](https://irvlutd.github.io/RPX/toolkit-docs/annotation/)**: SAM2 propagation with human correction, as used to label RPX
- **[Hardware profiler](https://irvlutd.github.io/RPX/toolkit-docs/profiling/)**: latency, throughput and memory for any callable
- **[Φ and JEDI calculator](https://irvlutd.github.io/RPX/toolkit-docs/analysis/)**: robustness and quality from raw metric records
- **[Add a metric or task](https://irvlutd.github.io/RPX/toolkit-docs/metrics/)**: register your own evaluation
- **[Dataset](https://huggingface.co/datasets/IRVLUTD/RPX)**: 100 scenes, 70 objects, ~133K frames on Hugging Face

## Testing

From `benchmark/` in a source checkout:

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
  title  = {Same Scene, Different Story: Evaluating Robot
            Perception Across Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran
            and Kamath, Srinanditha and Allu, Sai Haneesh and Rangappa, Govind Tyagi
            and Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

Toolkit code: MIT. Dataset: CC BY 4.0. Pretrained models keep their own licenses.

<div align="center">

# RPX

**A real-world RGB-D benchmark for robot perception.**

[![tests](https://img.shields.io/github/actions/workflow/status/IRVLUTD/RPX/tests.yml?branch=naren%2Fall&label=tests)](https://github.com/IRVLUTD/RPX/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](benchmark/pyproject.toml)
[![dataset](https://img.shields.io/badge/Dataset-IRVLUTD%2FRPX-yellow)](https://huggingface.co/datasets/IRVLUTD/RPX)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

```bash
git clone --branch naren/all https://github.com/IRVLUTD/RPX.git
cd RPX
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e './benchmark[hub,schemas]'
python benchmark/examples/run_depth.py
```

The example checks the installation using a synthetic depth model and local
data. To evaluate actual checkpoints, use the task-specific environments and
commands in [the benchmark README](benchmark/README.md#reference-models).

## Why RPX

RPX evaluates perception across **100 scenes** captured in three states:
clutter, human interaction, and clean. Results distinguish task accuracy,
robustness across scene changes, and compute cost. Relative-pose evaluation
uses clutter and clean; video-depth evaluation operates on whole phase clips.
The VQA benchmark also includes egocentric and two-image questions.

Images, depth, instance masks, poses and question data live in the
[RPX dataset](https://huggingface.co/datasets/IRVLUTD/RPX). Model weights and
large experiment outputs are downloaded or generated separately.

## Get started

| Goal | Start here |
|---|---|
| Run the supplied models | [Reference models and environments](benchmark/README.md#reference-models) |
| Evaluate your own model or API | [Bring your own model](benchmark/README.md#bring-your-own-model) |
| Inspect the dataset and splits | [Dataset metadata](benchmark/data/README.md) |
| Develop or verify the toolkit | [Testing](benchmark/README.md#testing) |

## How it works

**Capture.** RealSense RGB-D and tracking cameras record the scene.
See [capture tools](data/capture/README.md).

**Annotate.** The mask pipeline propagates reviewed object annotations through
each phase. Grounding and VQA tools derive the corresponding question data.
See [annotation tools](data/mask_annotation/README.md).

**Benchmark.** Task runners load a pinned dataset revision or local manifest,
execute a model, compute task metrics, and write reports with sample and
scene/phase details. Reference model adapters and custom callables use the
same underlying metrics. See [benchmark usage](benchmark/README.md).

<details>
<summary><b>Tasks and outputs</b></summary>

The core toolkit supports ten tasks: monocular depth, video depth, instance
segmentation, object detection, open-vocabulary detection, object tracking,
visual grounding, relative camera pose, sparse depth, and keypoint matching.
VQA has a separate question-manifest and bbox-scoring workflow.

Generic task runs write `result.json`, `summary.md`, and `cells.parquet`;
frame tasks also write `per_sample_metrics.parquet`. Production runners add
predictions and protocol-specific reports. Store outputs outside the source
tree or under the ignored `rpx_results/` directory.

</details>

<details>
<summary><b>Repository map</b></summary>

```text
benchmark/                 Installable Python package, examples and tests
  rpx_benchmark/           Loaders, adapters, metrics and public APIs
  scripts/                 Reference model runners and analysis utilities
  data/                    Dataset splits and release metadata
tracking/metadata/         Versioned tracking vocabulary
data/capture/             Sensor capture tools
data/mask_annotation/     Mask, grounding and VQA annotation tools
docker/                    Task-specific reproducible model environments
tools/dataset/             Dataset release preparation and QA
.github/workflows/         Tests, lint, types and package release
```

</details>

## Contributing and license

Run the [checks](benchmark/README.md#testing) before submitting changes.
Keep documentation in README files, dataset metadata in its designated
folders, and generated outputs out of Git. GPU checkpoint acceptance is
separate from the offline test suite; each model environment documents its
gates. Code is MIT; see [LICENSE](LICENSE). Dataset terms are recorded in the
[dataset card](https://huggingface.co/datasets/IRVLUTD/RPX).

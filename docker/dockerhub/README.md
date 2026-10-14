# RPX model environments

Ready-to-run Docker images for every model evaluated in **Same Scene, Different Story: Evaluating Robot Perception Across Scene Phases in the Wild**, a real-world benchmark that scores robot perception on the same scenes before, during and after manipulation.

[![Dataset](https://img.shields.io/badge/dataset-IRVLUTD%2FRPX-ffcc4d?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/IRVLUTD/RPX) [![Docs](https://img.shields.io/badge/docs-toolkit-0d9488?logo=materialformkdocs&logoColor=white)](https://irvlutd.github.io/RPX/toolkit-docs/) [![PyPI](https://img.shields.io/pypi/v/rpx-benchmark?label=pypi&color=4F46E5&logo=pypi&logoColor=white)](https://pypi.org/project/rpx-benchmark/) [![GitHub](https://img.shields.io/badge/source-IRVLUTD%2FRPX-24292f?logo=github)](https://github.com/IRVLUTD/RPX) [![License](https://img.shields.io/badge/license-MIT-3da639)](https://github.com/IRVLUTD/RPX/blob/main/LICENSE)

![RPX overview: six perception tasks scored on the same real scenes, captured before (Clutter), during (Interaction) and after (Clean) manipulation plus an egocentric view](https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-overview.webp)

*The same real scenes before, during and after manipulation, scored on six perception tasks. Each image below holds every model the paper evaluated on one task.*

[Which image to pull](#which-image-to-pull) · [Quick start](#quick-start) · [Reproducibility](#reproducibility) · [Citation](#citation) · [All tags](https://hub.docker.com/r/irvlutd/rpx/tags)

## Which image to pull

Each task's images are built cumulatively, so one tag holds every model the paper evaluated on that task.

| Task | Image tag | Models |
|:--|:--|:--|
| T1 Image depth | `depth-zipdepth-latest` | 10 |
| T2 Video depth | `depth-zipdepth-latest` | 8 |
| | `depth-dvd-latest` | DVD v1.1 |
| T3 Object tracking, first-frame mask | `tracking-dam4sam-rpx-latest` | 8 |
| T3 Object tracking, box prompt | `tracking-sam3.1-bbox-rpx-latest` | SAM 3.1 |
| T3 Object tracking, text prompt | `tracking-sam3.1-text-rpx-latest` | SAM 3.1 |
| | `tracking-grounded-sam2-text-rpx-latest` | Grounded SAM 2 |
| T4 Relative camera pose | `rcpe-monst3r-rpx-latest` | 10 |
| T5 VQA, T6 In-context VQA | `vqa-vllm` | 12 configurations |

Models in each image:

| Image | Models |
|:--|:--|
| `depth-zipdepth-latest` (image depth) | DA-V2 Large, DA3 Metric-L, Depth Pro, FE2E, HyDen, Lotus-2, Metric3D V2, MoGe-2 ViT-L, UniDepth V2, ZipDepth |
| `depth-zipdepth-latest` (video depth) | ChronoDepth, DA3 (video), DepthCrafter, MonST3R, RollingDepth, VGGT, ViGeo, Video-DA |
| `tracking-dam4sam-rpx-latest` | SAM 2, EdgeTAM, Cutie, SAM2Long, SAM 2++, MiTS, XMem, DAM4SAM |
| `rcpe-monst3r-rpx-latest` | VGGT, DA3, CUT3R, DUSt3R, MASt3R, MUSt3R, Reloc3R, Pi3X, Fast3R, MonST3R |
| `vqa-vllm` | 12 configurations of Cosmos Reason2, DeepSeek-VL2, Florence-2, InternVL 3.5, PaliGemma 2 and Qwen3-VL |

Always name a tag. A bare `docker pull irvlutd/rpx` fetches an old, unrelated capture environment, and tags not listed above are earlier build stages kept for provenance. DA3, MonST3R and VGGT appear in both the depth and pose images; use the `rcpe-` image for camera pose. GemDepth (T2) has a [build recipe](https://github.com/IRVLUTD/RPX/blob/main/docker/depth-gemdepth/README.md) but no published image.

## Quick start

You need Linux x86-64, an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and disk space for the image, checkpoints and dataset cache. Weights and data download at run time.

```bash
docker pull irvlutd/rpx:vqa-vllm

# What this image can run
docker run --rm irvlutd/rpx:vqa-vllm list-models

# Environment and GPU preflight
docker run --rm --gpus all irvlutd/rpx:vqa-vllm verify
```

`verify` checks the runtime, not a model. From there, each task's guide walks through a one-model smoke run and then the full benchmark.

| Task | Guide |
|---|---|
| T1, T2 | [Image and video depth](https://github.com/IRVLUTD/RPX/blob/main/docker/depth-smoke/README.md), [DVD](https://github.com/IRVLUTD/RPX/blob/main/docker/depth-dvd/README.md) |
| T3 | [Object tracking](https://github.com/IRVLUTD/RPX/blob/main/docker/tracking-smoke/README.md) |
| T4 | [Relative camera pose](https://github.com/IRVLUTD/RPX/blob/main/docker/rcpe-smoke/README.md) |
| T5, T6 | [VQA and in-context VQA](https://github.com/IRVLUTD/RPX/blob/main/docker/vqa-smoke/README.md) |

Mount caches and results from the host so they outlive the container. For the VQA image:

```bash
mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/rpx-vqa" "$PWD/rpx-results"
docker run --rm --gpus all --ipc=host -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/rpx-vqa:/cache/rpx-vqa" \
  -v "$PWD/rpx-results:/outputs" \
  irvlutd/rpx:vqa-vllm help
```

`HF_TOKEN` is only needed for gated checkpoints.

## Reproducibility

`-latest` tags move. For results you plan to report, pin the image by digest and record it alongside the checkpoint revision, dataset revision and model arguments:

```bash
docker image inspect irvlutd/rpx:tracking-dam4sam-rpx-latest \
  --format '{{index .RepoDigests 0}}'
# irvlutd/rpx@sha256:...  (use this reference for every later run)
```

You don't need Docker to score your own model: `pip install rpx-benchmark` and follow the [toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/getting-started/).

## Citation

```bibtex
@misc{rpx2026,
  title  = {Same Scene, Different Story: Evaluating Robot Perception
            Across Scene Phases in the Wild},
  author = {{Jishnu Jaykumar P} and Kadosh, Itay and Vijayakumar, Narendhiran
            and Kamath, Srinanditha and Allu, Sai Haneesh and Rangappa, Govind Tyagi
            and Maheshwari, Animesh and Wang, Jikai and Xiang, Yu},
  year   = {2026},
  note   = {Dataset: \url{https://huggingface.co/datasets/IRVLUTD/RPX}}
}
```

Environment code is MIT-licensed. Each model keeps its own license.

---

Part of [RPX](https://github.com/IRVLUTD/RPX) · [Toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/) · [Dataset](https://huggingface.co/datasets/IRVLUTD/RPX)

# RPX: reference-model environments

Ready-to-run Docker environments for every model evaluated in **RPX**, a real-world benchmark that scores
perception models on the same scenes before, during and after manipulation.

[Toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/) ·
[GitHub](https://github.com/IRVLUTD/RPX) ·
[Dataset](https://huggingface.co/datasets/IRVLUTD/RPX) ·
[PyPI](https://pypi.org/project/rpx-benchmark/) ·
[All tags](https://hub.docker.com/r/irvlutd/rpx/tags)

![RPX overview: six perception tasks scored on the same real scenes, captured before (Clutter), during (Interaction) and after (Clean) manipulation plus an egocentric view](https://raw.githubusercontent.com/IRVLUTD/RPX/main/benchmark/rpx_benchmark/guides/assets/rpx-overview.webp)

## Pick your image

Each task's images are built cumulatively, so **one tag per task holds every model the paper evaluated on it**.

| Task | `docker pull irvlutd/rpx:…` | Models inside |
|---|---|---|
| **T1** Image depth | `depth-zipdepth-latest` | DA-V2 Large, DA3 Metric-L, Depth Pro, FE2E, HyDen, Lotus-2, Metric3D V2, MoGe-2 ViT-L, UniDepth V2, ZipDepth |
| **T2** Video depth | `depth-zipdepth-latest` | ChronoDepth, DA3 (video), DepthCrafter, MonST3R, RollingDepth, VGGT, ViGeo, Video-DA |
| | `depth-dvd-latest` | DVD v1.1 |
| **T3** Tracking, first-frame mask | `tracking-dam4sam-rpx-latest` | SAM 2, EdgeTAM, Cutie, SAM2Long, SAM 2++, MiTS, XMem, DAM4SAM |
| **T3** Tracking, box prompt | `tracking-sam3.1-bbox-rpx-latest` | SAM 3.1 |
| **T3** Tracking, text prompt | `tracking-sam3.1-text-rpx-latest` | SAM 3.1 |
| | `tracking-grounded-sam2-text-rpx-latest` | Grounded SAM 2 |
| **T4** Relative camera pose | `rcpe-monst3r-rpx-latest` | VGGT, DA3, CUT3R, DUSt3R, MASt3R, MUSt3R, Reloc3R, Pi3X, Fast3R, MonST3R |
| **T5 / T6** VQA grounding, one or two images | `vqa-vllm` | 12 configurations of Cosmos Reason2, DeepSeek-VL2, Florence-2, InternVL 3.5, PaliGemma 2 and Qwen3-VL |

> **Always name a tag.** A bare `docker pull irvlutd/rpx` fetches an unrelated legacy capture environment. Every tag not
> listed above is an earlier build stage kept for provenance.

DA3, MonST3R and VGGT appear in both depth and pose images; use the `rcpe-` image for camera pose. GemDepth (T2) has a
[build recipe](https://github.com/IRVLUTD/RPX/blob/main/docker/depth-gemdepth/README.md) but no published image.

## Quick start

You need Linux x86-64, an NVIDIA GPU with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and disk space for the image, checkpoints and dataset cache. The images hold code and environments only; weights and
data download at run time.

```bash
docker pull irvlutd/rpx:vqa-vllm
docker run --rm irvlutd/rpx:vqa-vllm list-models          # what this image can run
docker run --rm --gpus all irvlutd/rpx:vqa-vllm verify    # environment and GPU preflight
```

`verify` checks the runtime, not a model. Next, follow your task's guide: a one-model smoke run, then the full
benchmark.

| Task | Guide with smoke gates and benchmark commands |
|---|---|
| T1, T2 | [Depth](https://github.com/IRVLUTD/RPX/blob/main/docker/depth-smoke/README.md) · [DVD](https://github.com/IRVLUTD/RPX/blob/main/docker/depth-dvd/README.md) |
| T3 | [Tracking](https://github.com/IRVLUTD/RPX/blob/main/docker/tracking-smoke/README.md) |
| T4 | [Relative camera pose](https://github.com/IRVLUTD/RPX/blob/main/docker/rcpe-smoke/README.md) |
| T5, T6 | [VQA grounding](https://github.com/IRVLUTD/RPX/blob/main/docker/vqa-smoke/README.md) |

Keep caches and results on the host so they survive the container. For the VQA image:

```bash
mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/rpx-vqa" "$PWD/rpx-results"
docker run --rm --gpus all --ipc=host -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/rpx-vqa:/cache/rpx-vqa" \
  -v "$PWD/rpx-results:/outputs" \
  irvlutd/rpx:vqa-vllm help
```

`HF_TOKEN` is only needed for gated checkpoints.

## Reproducing a run

`…-latest` tags can move. For a result you will report, pin the image by digest and record it with the checkpoint
revision, dataset revision and model arguments:

```bash
docker image inspect irvlutd/rpx:tracking-dam4sam-rpx-latest --format '{{index .RepoDigests 0}}'
# → irvlutd/rpx@sha256:…   use this reference for every later run
```

To score your own model instead, you do not need Docker: `pip install rpx-benchmark` and follow the
[toolkit docs](https://irvlutd.github.io/RPX/toolkit-docs/getting-started/).

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

Environment code is MIT. Each model keeps its own license.

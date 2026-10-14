# RPX model environments

Docker environments for every model evaluated in the RPX paper, published as tags of
[`irvlutd/rpx`](https://hub.docker.com/r/irvlutd/rpx/tags). The images are `linux/amd64` and contain code and
environments only. Data, checkpoints, predictions and credentials are not included. GPU runs need the NVIDIA
Container Toolkit on the host.

You don't need Docker to score your own model; see [`benchmark/`](../benchmark/README.md).

## Which image to pull

Each task's images are built cumulatively, so one tag holds every model the paper evaluated on that task.

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
| **T5 / T6** VQA and in-context VQA | `vqa-vllm` | 12 configurations of Cosmos Reason2, DeepSeek-VL2, Florence-2, InternVL 3.5, PaliGemma 2 and Qwen3-VL |

Always name a tag. Bare `irvlutd/rpx` (`latest`) is an old, unrelated capture environment, and tags not listed above
are earlier build stages kept for provenance. DA3, MonST3R and VGGT appear in both depth and pose images; use
the `rcpe-` image for camera pose. GemDepth has a [build recipe](depth-gemdepth/README.md) but no published image.

## Pull and verify

```bash
export RPX_CACHE="$HOME/.cache/rpx" RPX_OUTPUT="$PWD/rpx_results"
mkdir -p "$RPX_CACHE" "$RPX_OUTPUT"

docker pull irvlutd/rpx:vqa-vllm
docker run --rm irvlutd/rpx:vqa-vllm list-models
docker run --rm --gpus all irvlutd/rpx:vqa-vllm verify
```

`verify` confirms that the runtime imports its model stack and can see the GPU. Next run the model's smoke command,
which loads real weights, and only then the full dataset. Set `HF_TOKEN` for gated checkpoints. For results you
plan to report, pin the image by digest (`docker image inspect <tag> --format '{{index .RepoDigests 0}}'`).

## Task guides

Each guide covers model selection, checkpoint revisions, gates and full benchmark commands.

| Task | Guide |
|---|---|
| T1, T2 | [Image and video depth](depth-smoke/README.md), with the [FE2E](depth-fe2e/README.md), [ZipDepth](depth-zipdepth/README.md), [DVD](depth-dvd/README.md) and [GemDepth](depth-gemdepth/README.md) overlays |
| T3 | [Object tracking](tracking-smoke/README.md) |
| T4 | [Relative camera pose](rcpe-smoke/README.md) |
| T5, T6 | [VQA and in-context VQA](vqa-smoke/README.md) |

## Building images

Build scripts live in each task directory. By default they push to the maintainers' staging repositories
(`<user>/rpx-<task>-smoke:<tag>`), which are then published as `irvlutd/rpx:<task>-<tag>`. Set `RPX_DEPTH_IMAGE`,
`RPX_RCPE_IMAGE`, `RPX_TRACKING_IMAGE` or `RPX_VQA_IMAGE` to push elsewhere.

The Docker Hub description for `irvlutd/rpx` is kept in [`dockerhub/README.md`](dockerhub/README.md). After editing
it, paste it into the repository overview on Docker Hub.

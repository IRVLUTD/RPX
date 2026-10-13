# RPX model environments

Docker environments for every reference model in the RPX paper, published as tags of
**[`irvlutd/rpx`](https://hub.docker.com/r/irvlutd/rpx/tags)**. Images are `linux/amd64` and hold code and
environments only: no dataset shards, checkpoints, predictions or credentials. GPU runs need the NVIDIA Container
Toolkit on the host.

To score your own model you do not need Docker; see [`benchmark/`](../benchmark/README.md).

## Which image to pull

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

Always name a tag: bare `irvlutd/rpx` (`latest`) is an unrelated legacy capture environment, and every tag not listed
above is an earlier build stage kept for provenance. DA3, MonST3R and VGGT appear in both depth and pose images; use
the `rcpe-` image for camera pose. GemDepth has a [build recipe](depth-gemdepth/README.md) but no published image.

## Pull and verify

```bash
export RPX_CACHE="$HOME/.cache/rpx" RPX_OUTPUT="$PWD/rpx_results"
mkdir -p "$RPX_CACHE" "$RPX_OUTPUT"

docker pull irvlutd/rpx:vqa-vllm
docker run --rm irvlutd/rpx:vqa-vllm list-models
docker run --rm --gpus all irvlutd/rpx:vqa-vllm verify
```

`verify` (or a task's gate) proves the runtime imports its model stack and reaches the GPU. Then run the
model-specific smoke command, which loads real weights, and only then the full dataset sweep. Set `HF_TOKEN` for gated
checkpoints. For a result you will report, pin the image by digest
(`docker image inspect <tag> --format '{{index .RepoDigests 0}}'`).

## Task guides

Model selection, checkpoint revisions, gate commands and full benchmark commands:

| Task | Guide |
|---|---|
| T1, T2 | [Image and video depth](depth-smoke/README.md), with the [FE2E](depth-fe2e/README.md), [ZipDepth](depth-zipdepth/README.md), [DVD](depth-dvd/README.md) and [GemDepth](depth-gemdepth/README.md) overlays |
| T3 | [Object tracking](tracking-smoke/README.md) |
| T4 | [Relative camera pose](rcpe-smoke/README.md) |
| T5, T6 | [VQA grounding](vqa-smoke/README.md) |

## Building images

Each task directory has build scripts. They push to the maintainers' staging repositories by default
(`<user>/rpx-<task>-smoke:<tag>`, published as `irvlutd/rpx:<task>-<tag>`); override the target with
`RPX_DEPTH_IMAGE`, `RPX_RCPE_IMAGE`, `RPX_TRACKING_IMAGE` or `RPX_VQA_IMAGE`.

The Docker Hub overview for `irvlutd/rpx` is kept in [`dockerhub/README.md`](dockerhub/README.md); paste it into the repository's
overview on Docker Hub after editing.

# RPX model environments

The task-specific images reproduce the reference-model runtimes without
placing dataset shards, checkpoints, predictions or credentials in an image.
Published images are `linux/amd64`; NVIDIA runs require the NVIDIA Container
Toolkit on the host.

## Which image to pull

All published images are tags of **[`irvlutd/rpx`](https://hub.docker.com/r/irvlutd/rpx/tags)**.
Each task's images are built cumulatively, so pull the **final image for your
task**: it already contains every earlier model of that task.

| Task | Pull | Models inside |
|---|---|---|
| T1 image depth, T2 video depth | `irvlutd/rpx:depth-zipdepth-latest` | All ten T1 models; T2 ChronoDepth, DA3 (video), DepthCrafter, MonST3R, RollingDepth, VGGT, ViGeo, Video-DA |
| T2 video depth: DVD | `irvlutd/rpx:depth-dvd-latest` | DVD v1.1 (plus the shared depth runtime) |
| T3 tracking, mask-initialized | `irvlutd/rpx:tracking-dam4sam-rpx-latest` | SAM 2, EdgeTAM, Cutie, SAM2Long, SAM 2++, MiTS, XMem, DAM4SAM |
| T3 tracking, SAM 3.1 box prompt | `irvlutd/rpx:tracking-sam3.1-bbox-rpx-latest` | SAM 3.1 (bbox) |
| T3 tracking, SAM 3.1 text prompt | `irvlutd/rpx:tracking-sam3.1-text-rpx-latest` | SAM 3.1 (text) |
| T3 tracking, Grounded SAM 2 text prompt | `irvlutd/rpx:tracking-grounded-sam2-text-rpx-latest` | GSAM2 (text) |
| T4 relative camera pose | `irvlutd/rpx:rcpe-monst3r-rpx-latest` | All ten: VGGT, DA3, CUT3R, DUSt3R, MASt3R, MUSt3R, Reloc3R, Pi3X, Fast3R, MonST3R |
| T5 / T6 VQA and grounding | `irvlutd/rpx:vqa-vllm` | All twelve VLM configurations, one-image and two-image |

GemDepth (T2) has a build recipe in [`depth-gemdepth/`](depth-gemdepth/README.md)
but no published image. Every other tag in `irvlutd/rpx` is an earlier
stage of one of the images above and is kept only for provenance; `latest`
is an unrelated older environment, so always name a tag.

The exact model selection, checkpoint revisions, gate commands and full
benchmark commands are in each task README:

- [Image and video depth](depth-smoke/README.md), plus the dedicated
  [FE2E](depth-fe2e/README.md), [ZipDepth](depth-zipdepth/README.md),
  [DVD](depth-dvd/README.md) and [GemDepth](depth-gemdepth/README.md) overlays
- [Relative camera pose](rcpe-smoke/README.md)
- [Object tracking](tracking-smoke/README.md)
- [VQA and bbox grounding](vqa-smoke/README.md)

## Pull and verify

Set the cache and output directories on the host so runs are reproducible and
survive container removal. Keep `HF_TOKEN` in the environment.

```bash
export HF_TOKEN=hf_...
export RPX_CACHE="$HOME/.cache/rpx"
export RPX_OUTPUT="$PWD/rpx_results"
mkdir -p "$RPX_CACHE" "$RPX_OUTPUT"

docker pull irvlutd/rpx:vqa-vllm
docker run --rm irvlutd/rpx:vqa-vllm list-models
docker run --rm --gpus all irvlutd/rpx:vqa-vllm verify
```

The small `verify` or task gate proves that the runtime can import its model
stack and reach the GPU. Run the model-specific smoke command next; it loads
real weights and executes inference. Only then start the full dataset sweep.

To build instead of pull, use the scripts in the corresponding task directory.
The build scripts push to the maintainers' staging repositories by default;
override the target with `RPX_DEPTH_IMAGE`, `RPX_RCPE_IMAGE`,
`RPX_TRACKING_IMAGE` or `RPX_VQA_IMAGE`.

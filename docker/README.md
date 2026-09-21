# RPX model environments

The task-specific images reproduce the reference-model runtimes without
placing dataset shards, checkpoints, predictions or credentials in an image.
Published images are `linux/amd64`; NVIDIA runs require the NVIDIA Container
Toolkit on the host.

## Published repositories

| Task | Docker Hub repository | What it contains |
|---|---|---|
| Image and video depth | `narendhiranv04/rpx-depth-smoke` | Shared depth runtime and cumulative reference-model tags |
| Relative camera pose | `narendhiranv04/rpx-rcpe-smoke` | Ten canonical RCPE model stages |
| Object tracking | `narendhiranv04/rpx-tracking-smoke` | Mask-, box- and text-initialized model stages |
| VQA and bbox grounding | `narendhiranv04/rpx-vqa-smoke` | Native and vLLM runtimes for the 20-entry roster |

The exact tag for each model, its checkpoint revision, the gate command and
the full benchmark command are documented in the task README:

- [Image and video depth](depth-smoke/README.md), plus the dedicated
  [FE2E](depth-fe2e/README.md), [ZipDepth](depth-zipdepth/README.md),
  [DVD](depth-dvd/README.md) and [GEMDepth](depth-gemdepth/README.md) overlays
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

docker pull narendhiranv04/rpx-vqa-smoke:vllm
docker run --rm narendhiranv04/rpx-vqa-smoke:vllm list-models
docker run --rm --gpus all narendhiranv04/rpx-vqa-smoke:vllm verify
```

The small `verify` or task gate proves that the runtime can import its model
stack and reach the GPU. Run the model-specific smoke command next; it loads
real weights and executes inference. Only then start the full dataset sweep.

To build instead of pull, use the scripts in the corresponding task directory.
Every registry name can be overridden through `RPX_DEPTH_IMAGE`,
`RPX_RCPE_IMAGE`, `RPX_TRACKING_IMAGE` or `RPX_VQA_IMAGE`.

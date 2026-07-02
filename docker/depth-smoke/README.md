# Four-model GPU depth smoke image

This image packages the exact Python 3.11/CUDA dependency union used by the
four currently accepted RPX image-depth models:

- `da-v2-large`
- `depth-pro`
- `unidepth-v2`
- `moge-2-vit-l`

It embeds RPX at one Git commit plus the pinned UniDepth and MoGe upstream
commits. It does **not** contain RPX data, Hugging Face tokens, checkpoints or
smoke outputs. Those stay in mounted host directories.

## Host prerequisite

Docker must have NVIDIA Container Toolkit configured. Verify it before running
RPX:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Follow NVIDIA's official install guide if Docker reports that no GPU device
driver is available, then run:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Pull and verify

```bash
docker pull irvlutd/rpx-depth-smoke:latest
docker run --rm --gpus all --ipc=host --shm-size=8g \
  irvlutd/rpx-depth-smoke:latest verify
```

## Run all four smoke gates

```bash
mkdir -p "$HOME/.cache/huggingface" "$PWD/rpx-smoke"

docker run --rm --gpus all --ipc=host --shm-size=8g \
  -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$PWD/rpx-smoke:/outputs" \
  irvlutd/rpx-depth-smoke:latest smoke
```

The command runs sequentially and resumes from the matrix JSON for the exact
image commit. To run one model, set `-e RPX_MODELS=depth-pro`. To repeat a
failed dependency/code gate after a fix, append `--retry-failed`. The image
never enables FE2E, D4RT or GemDepth's unverified-weight override.

## Build and publish

The helper refuses a dirty source tree and tags both `latest` and the immutable
`sha-<12 characters>` revision. It automatically reuses the setup helper's
`../depth-smoke-envs/.wheelhouse` when present; set `RPX_TORCH_WHEELHOUSE` to
override it. The wheelhouse is mounted only during the build and is not copied
into the image:

```bash
docker/depth-smoke/build_and_push.sh
docker/depth-smoke/build_and_push.sh --push
```

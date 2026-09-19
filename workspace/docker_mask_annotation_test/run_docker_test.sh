#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="/media/naren/New Volume/rpx/docker_mask_annotation_test"
IMAGE_NAME="${IMAGE_NAME:-irvlutd/rpx-mask-annotation:latest}"

if [[ ! -d "${TEST_ROOT}/sample_scene/rgb" ]]; then
  echo "Missing sample_scene/rgb. Run ${TEST_ROOT}/prepare_sample_scene.sh first." >&2
  exit 1
fi

if [[ ! -f "${TEST_ROOT}/checkpoints/sam2.1_hiera_large.pth" ]]; then
  echo "Warning: ${TEST_ROOT}/checkpoints/sam2.1_hiera_large.pth is missing." >&2
  echo "The container will start, but full SAM2 segmentation will fail until the checkpoint is provided." >&2
fi

docker_args=(
  -it
  --rm
  -v "/media/naren/New Volume/rpx/github_rpx:/workspace/rpx"
  -v "${TEST_ROOT}/sample_scene:/workspace/data/sample_scene"
  -v "${TEST_ROOT}/checkpoints:/workspace/checkpoints"
  -v "${TEST_ROOT}/hf_cache:/home/mambauser/.cache/huggingface"
  -e "HF_HOME=/home/mambauser/.cache/huggingface"
  -e "SAM2_DEVICE=${SAM2_DEVICE:-cpu}"
  -e "SAM2_CHECKPOINT=/workspace/checkpoints/sam2.1_hiera_large.pth"
  -e "SAM2_PIPELINE_LOG=/tmp/sam2_pipeline.log"
  -w /workspace/rpx/data/mask_annotation
)

if [[ -n "${DISPLAY:-}" && -S /tmp/.X11-unix/X${DISPLAY#:} ]]; then
  docker_args+=(
    -e "DISPLAY=${DISPLAY}"
    -v /tmp/.X11-unix:/tmp/.X11-unix
  )
fi

docker run "${docker_args[@]}" "${IMAGE_NAME}" "$@"

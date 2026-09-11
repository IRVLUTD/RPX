#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: start_persistent_engine.sh MODEL GPU}"
gpu="${2:?usage: start_persistent_engine.sh MODEL GPU}"
runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/vqa-runtime}"
image="${RPX_VQA_IMAGE:-vndhiran123/rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
name="rpx-vqa-${model//[^a-zA-Z0-9_.-]/-}"
startup_polls="${RPX_VQA_STARTUP_POLLS:-360}"
startup_poll_seconds="${RPX_VQA_STARTUP_POLL_SECONDS:-2}"
pytorch_cuda_alloc_conf="${RPX_VQA_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ "${startup_polls}" =~ ^[1-9][0-9]*$ ]] || {
  echo "RPX_VQA_STARTUP_POLLS must be a positive integer" >&2
  exit 2
}
[[ "${startup_poll_seconds}" =~ ^[1-9][0-9]*$ ]] || {
  echo "RPX_VQA_STARTUP_POLL_SECONDS must be a positive integer" >&2
  exit 2
}

test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not exported" >&2; exit 2; }
mkdir -p "${runtime}"/{hf-cache,cache,outputs,logs}

if docker ps --format '{{.Names}}' | grep -Fxq "${name}"; then
  running_image="$(docker inspect --format '{{.Config.Image}}' "${name}")"
  if [[ "${running_image}" != "${image}:${tag}" ]]; then
    echo "${name} is running the stale image ${running_image}; expected ${image}:${tag}" >&2
    echo "stop and remove that exact container before starting this revision" >&2
    exit 1
  fi
  echo "${name} is already running the requested image"
  exit 0
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
  docker rm "${name}" >/dev/null
fi

gpu_request="device=${gpu}"
if [[ "${gpu}" == *,* ]]; then
  gpu_request="\"device=${gpu}\""
fi

echo "PyTorch CUDA allocator: ${pytorch_cuda_alloc_conf}"
docker run -d \
  --name "${name}" \
  --gpus "${gpu_request}" \
  --ipc=host \
  --shm-size=16g \
  -e HF_TOKEN \
  -e "PYTORCH_CUDA_ALLOC_CONF=${pytorch_cuda_alloc_conf}" \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${runtime}/outputs:/outputs" \
  "${image}:${tag}" serve "${model}" >/dev/null

echo "loading ${model} in ${name} on GPU ${gpu}"
for _ in $(seq 1 "${startup_polls}"); do
  if ! docker ps --format '{{.Names}}' | grep -Fxq "${name}"; then
    docker logs "${name}" --tail 100
    exit 1
  fi
  if docker exec "${name}" python3 -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)' \
    >/dev/null 2>&1; then
    echo "${name}: READY"
    exit 0
  fi
  sleep "${startup_poll_seconds}"
done
echo "timed out waiting for ${name} after $((startup_polls * startup_poll_seconds)) seconds" >&2
docker logs "${name}" --tail 100
exit 1

#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: start_persistent_engine.sh MODEL GPU}"
gpu="${2:?usage: start_persistent_engine.sh MODEL GPU}"
runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/vqa-runtime}"
image="${RPX_VQA_IMAGE:-vndhiran123/rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
name="rpx-vqa-${model//[^a-zA-Z0-9_.-]/-}"

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

docker run -d \
  --name "${name}" \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=16g \
  -e HF_TOKEN \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${runtime}/outputs:/outputs" \
  "${image}:${tag}" serve "${model}" >/dev/null

echo "loading ${model} in ${name} on GPU ${gpu}"
for _ in $(seq 1 360); do
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
  sleep 2
done
echo "timed out waiting for ${name}" >&2
docker logs "${name}" --tail 100
exit 1

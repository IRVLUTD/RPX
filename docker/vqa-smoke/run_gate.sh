#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash docker/vqa-smoke/run_gate.sh {smoke|acceptance} MODEL [GPU] [runner args...]" >&2
  exit 2
fi

gate="$1"
model="$2"
gpu="${3:-0}"
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi

if [[ "${gate}" != "smoke" && "${gate}" != "acceptance" ]]; then
  echo "gate must be smoke or acceptance" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not exported" >&2
  exit 2
fi

runtime="${RPX_VQA_RUNTIME:-/data/rpx/src/vqa-runtime}"
image="${RPX_VQA_IMAGE:-rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
mkdir -p "${runtime}/hf-cache" "${runtime}/cache" "${runtime}/outputs" "${runtime}/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${runtime}/logs/${gate}-${model}-${timestamp}.log"
compat_args=()
if [[ "${RPX_VQA_CUDA_COMPAT:-0}" == "1" ]]; then
  compat_args+=(
    -e NVIDIA_DISABLE_REQUIRE=true
    -e VLLM_ENABLE_CUDA_COMPATIBILITY=1
  )
fi

echo "backend=auto image=${image}:${tag} gate=${gate} model=${model} gpu=${gpu}"
docker run --rm \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=16g \
  "${compat_args[@]}" \
  -e HF_TOKEN \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${runtime}/outputs:/outputs" \
  "${image}:${tag}" "${gate}" "${model}" "$@" 2>&1 | tee "${log}"

echo "log=${log}"

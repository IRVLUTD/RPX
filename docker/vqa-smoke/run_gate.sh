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

runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/src/vqa-runtime}"
image="${RPX_VQA_IMAGE:-vndhiran123/rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
mkdir -p "${runtime}/hf-cache" "${runtime}/cache" "${runtime}/outputs" "${runtime}/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${runtime}/logs/${gate}-${model}-${timestamp}.log"

echo "backend=vllm image=${image}:${tag} gate=${gate} model=${model} gpu=${gpu}"
docker run --rm \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=16g \
  -e HF_TOKEN \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${runtime}/outputs:/outputs" \
  "${image}:${tag}" "${gate}" "${model}" "$@" 2>&1 | tee "${log}"

echo "log=${log}"

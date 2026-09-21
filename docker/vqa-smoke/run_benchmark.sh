#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: bash docker/vqa-smoke/run_benchmark.sh MODEL GPU MANIFEST SHARD_INDEX SHARD_COUNT [runner args...]" >&2
  exit 2
fi

model="$1"
gpu="$2"
manifest="$3"
shard_index="$4"
shard_count="$5"
shift 5

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not exported" >&2
  exit 2
fi
if ! [[ "${shard_index}" =~ ^[0-9]+$ && "${shard_count}" =~ ^[0-9]+$ && "${shard_index}" -lt "${shard_count}" ]]; then
  echo "SHARD_INDEX must be an integer in [0, SHARD_COUNT)" >&2
  exit 2
fi
if [[ ! -f "${manifest}" ]]; then
  echo "manifest not found: ${manifest}" >&2
  exit 2
fi

runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/src/vqa-runtime}"
image="${RPX_VQA_IMAGE:-narendhiranv04/rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
manifest_dir="$(cd "$(dirname "${manifest}")" && pwd)"
manifest_name="$(basename "${manifest}")"
mkdir -p "${runtime}/hf-cache" "${runtime}/cache" "${runtime}/outputs" "${runtime}/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
repo_sha="$(git rev-parse --short=12 HEAD)"
run_dir_host="${runtime}/outputs/${model}/sha-${repo_sha}/benchmark/shard-${shard_index}-of-${shard_count}"
mkdir -p "${run_dir_host}"
log="${runtime}/logs/benchmark-${model}-shard${shard_index}of${shard_count}-${timestamp}.log"
compat_args=()
if [[ "${RPX_VQA_CUDA_COMPAT:-0}" == "1" ]]; then
  compat_args+=(
    -e NVIDIA_DISABLE_REQUIRE=true
    -e VLLM_ENABLE_CUDA_COMPATIBILITY=1
  )
fi

echo "backend=auto image=${image}:${tag} model=${model} gpu=${gpu} shard=${shard_index}/${shard_count} manifest=${manifest}"
docker run --rm \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=16g \
  "${compat_args[@]}" \
  -e HF_TOKEN \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${runtime}/outputs:/outputs" \
  -v "${manifest_dir}:/manifests:ro" \
  "${image}:${tag}" benchmark "${model}" "/manifests/${manifest_name}" "${shard_index}" "${shard_count}" "$@" 2>&1 | tee "${log}"

echo "log=${log}"
echo "run_dir=${run_dir_host}"

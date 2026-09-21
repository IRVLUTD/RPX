#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: bash docker/vqa-smoke/run_persistent_benchmark.sh MODEL MANIFEST SHARD_INDEX SHARD_COUNT [runner args...]" >&2
  exit 2
fi

model="$1"
manifest="$2"
shard_index="$3"
shard_count="$4"
shift 4

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=persistent_engine_lib.sh
source "${script_dir}/persistent_engine_lib.sh"
runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/vqa-runtime}"
image="${RPX_VQA_IMAGE:-narendhiranv04/rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
name="$(rpx_vqa_engine_name "${model}")"
instance_suffix="$(rpx_vqa_engine_instance_suffix)"
repo_sha="${RPX_GIT_SHA:-}"
if [[ -z "${repo_sha}" ]]; then
  repo_sha="$(git rev-parse HEAD)"
fi
repo_sha="${repo_sha:0:12}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not exported" >&2; exit 2; }
test -f "${manifest}" || { echo "manifest not found: ${manifest}" >&2; exit 2; }
docker ps --format '{{.Names}}' | grep -Fxq "${name}" || {
  echo "resident engine is not running: ${name}" >&2
  exit 1
}

manifest_dir="$(cd "$(dirname "${manifest}")" && pwd)"
manifest_name="$(basename "${manifest}")"
mkdir -p "${runtime}/hf-cache" "${runtime}/cache" "${runtime}/outputs" "${runtime}/logs"
log="${runtime}/logs/benchmark-${model}${instance_suffix}-shard${shard_index}of${shard_count}-${timestamp}.log"
run_dir="${runtime}/outputs/${model}/sha-${repo_sha}/benchmark/shard-${shard_index}-of-${shard_count}"

health="$(docker exec "${name}" python3 -c \
  'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10).read().decode())')"
echo "resident=${name} health=${health}"
echo "client_image=${image}:${tag} model=${model} shard=${shard_index}/${shard_count}"

# The short-lived client shares the resident container's network namespace,
# but not its process/GPU lifecycle.  It downloads images into the same cache,
# writes resumable JSONL after every row, and leaves the model resident.
docker run --rm \
  --network "container:${name}" \
  -e HF_TOKEN \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${runtime}/outputs:/outputs" \
  -v "${manifest_dir}:/manifests:ro" \
  "${image}:${tag}" benchmark-remote "${model}" "/manifests/${manifest_name}" \
  "${shard_index}" "${shard_count}" "$@" 2>&1 | tee "${log}"

echo "engine remains resident: ${name}"
echo "log=${log}"
echo "run_dir=${run_dir}"

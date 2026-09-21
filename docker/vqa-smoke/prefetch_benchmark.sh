#!/usr/bin/env bash
set -euo pipefail

manifest="${1:?usage: bash docker/vqa-smoke/prefetch_benchmark.sh MANIFEST [prefetch args...]}"
shift
runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/vqa-runtime}"
image="${RPX_VQA_IMAGE:-narendhiranv04/rpx-vqa-smoke}"
tag="${RPX_VQA_TAG:-vllm}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not exported" >&2; exit 2; }
test -f "${manifest}" || { echo "manifest not found: ${manifest}" >&2; exit 2; }
manifest_dir="$(cd "$(dirname "${manifest}")" && pwd)"
manifest_name="$(basename "${manifest}")"
mkdir -p "${runtime}/hf-cache" "${runtime}/cache" "${runtime}/logs"
log="${runtime}/logs/benchmark-prefetch-${timestamp}.log"

docker run --rm \
  -e HF_TOKEN \
  -v "${runtime}/hf-cache:/cache/huggingface" \
  -v "${runtime}/cache:/cache/rpx-vqa" \
  -v "${manifest_dir}:/manifests:ro" \
  "${image}:${tag}" benchmark-prefetch "/manifests/${manifest_name}" \
  "$@" \
  2>&1 | tee "${log}"

echo "log=${log}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 {sam3.1|grounded-sam2} {smoke|micro|acceptance} {mos|ego} GPU" >&2
  exit 2
fi
model="$1"
gate="$2"
protocol="$3"
gpu="$4"
case "${model}" in sam3.1|grounded-sam2) ;; *) exit 2 ;; esac
case "${gate}" in smoke|micro|acceptance) ;; *) exit 2 ;; esac
case "${protocol}" in mos|ego) ;; *) exit 2 ;; esac
[[ "${gpu}" =~ ^[0-9]+$ ]] || exit 2

runtime="${RPX_TRACKING_RUNTIME:-/data/narendhiran_rpx/tracking-bbox-runtime}"
hf_cache="${RPX_HF_CACHE:-/data/narendhiran_rpx/docker-smoke/tracking-caches/rpx-shared/huggingface}"
output_root="${RPX_TRACKING_OUTPUT:-${runtime}/outputs}"
registry="${RPX_TRACKING_IMAGE:-narendhiranv04/rpx-tracking-smoke}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short="${revision:0:12}"
image="${registry}:${model}-bbox-rpx-sha-${short}"
safe_model="${model//./_}"
safe_model="${safe_model//-/_}"
python="/opt/rpx-envs/${safe_model}/bin/python"
mkdir -p "${hf_cache}" "${output_root}" "${runtime}/logs"
log="${runtime}/logs/${model}-${protocol}-${gate}-${short}-$(date -u +%Y%m%dT%H%M%SZ).log"

docker image inspect "${image}" >/dev/null
docker run --rm \
  --name "rpx-${safe_model}-bbox-${protocol}-${gate}" \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=32g \
  -e HF_TOKEN \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_CACHE=/cache/huggingface/hub \
  -e PYTHONUNBUFFERED=1 \
  -v "${hf_cache}:/cache/huggingface" \
  -v "${output_root}:/outputs" \
  "${image}" \
  "${python}" scripts/run_tracking_gate.py \
    --model "${model}" \
    --gate "${gate}" \
    --dataset-protocol "${protocol}" \
    --cache-dir /cache/huggingface \
    --output-root /outputs \
  2>&1 | tee "${log}"

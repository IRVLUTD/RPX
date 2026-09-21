#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {sam3.1|grounded-sam2} {smoke|micro|acceptance} {mos|ego} GPU [VOCAB_PARQUET]" >&2
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 2
fi
model="$1"
gate="$2"
protocol="$3"
gpu="$4"
vocab="${5:-${RPX_TRACKING_TEXT_VOCAB:-}}"
case "${model}" in sam3.1|grounded-sam2) ;; *) usage; exit 2 ;; esac
case "${gate}" in smoke|micro|acceptance) ;; *) usage; exit 2 ;; esac
case "${protocol}" in mos|ego) ;; *) usage; exit 2 ;; esac
[[ "${gpu}" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ -f "${vocab}" ]] || { echo "Vocabulary parquet not found: ${vocab}" >&2; exit 1; }

runtime="${RPX_TRACKING_RUNTIME:-/data/rpx/tracking-runtime}"
hf_cache="${RPX_HF_CACHE:-${runtime}/cache/huggingface}"
data_cache="${RPX_TRACKING_DATA_CACHE:-${runtime}/cache/rpx}"
output_root="${RPX_TRACKING_OUTPUT:-${runtime}/outputs}"
registry="${RPX_TRACKING_IMAGE:-rpx-tracking-smoke}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short="${revision:0:12}"
image="${registry}:${model}-text-rpx-sha-${short}"
safe_model="${model//./_}"
safe_model="${safe_model//-/_}"
python="/opt/rpx-envs/${safe_model}/bin/python"
mkdir -p "${hf_cache}" "${data_cache}" "${output_root}" "${runtime}/logs"
log="${runtime}/logs/${model}-${protocol}-${gate}-${short}-$(date -u +%Y%m%dT%H%M%SZ).log"

docker image inspect "${image}" >/dev/null
echo "model=${model} gate=${gate} protocol=${protocol} gpu=${gpu} image=${image}"
docker run --rm \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=24g \
  -e HF_TOKEN \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_CACHE=/cache/huggingface/hub \
  -v "${hf_cache}:/cache/huggingface" \
  -v "${data_cache}:/cache/rpx" \
  -v "${output_root}:/outputs" \
  -v "$(realpath "${vocab}"):/vocab/scene_condition_vocab.parquet:ro" \
  "${image}" \
  "${python}" scripts/run_tracking_gate.py \
    --model "${model}" \
    --gate "${gate}" \
    --dataset-protocol "${protocol}" \
    --cache-dir /cache/rpx \
    --output-root /outputs \
    --text-vocab /vocab/scene_condition_vocab.parquet \
  2>&1 | tee "${log}"

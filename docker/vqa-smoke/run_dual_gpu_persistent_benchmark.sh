#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: bash docker/vqa-smoke/run_dual_gpu_persistent_benchmark.sh MODEL MANIFEST GPU_A GPU_B [runner args...]" >&2
  exit 2
fi

model="$1"
manifest="$2"
gpu_a="$3"
gpu_b="$4"
shift 4

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/vqa-runtime}"

[[ "${gpu_a}" =~ ^[0-9]+$ && "${gpu_b}" =~ ^[0-9]+$ ]] || {
  echo "GPU_A and GPU_B must be numeric physical GPU indices" >&2
  exit 2
}
[[ "${gpu_a}" != "${gpu_b}" ]] || {
  echo "GPU_A and GPU_B must be different" >&2
  exit 2
}
test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not exported" >&2; exit 2; }
test -s "${manifest}" || { echo "manifest not found or empty: ${manifest}" >&2; exit 2; }

available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
grep -Fxq "${gpu_a}" <<<"${available_gpus}" || {
  echo "physical GPU ${gpu_a} is unavailable" >&2
  exit 2
}
grep -Fxq "${gpu_b}" <<<"${available_gpus}" || {
  echo "physical GPU ${gpu_b} is unavailable" >&2
  exit 2
}

instance_a="gpu${gpu_a}"
instance_b="gpu${gpu_b}"

echo "starting ${model} replicas on physical GPUs ${gpu_a} and ${gpu_b}"
RPX_VQA_ENGINE_INSTANCE="${instance_a}" \
  bash "${script_dir}/start_persistent_engine.sh" "${model}" "${gpu_a}"
RPX_VQA_ENGINE_INSTANCE="${instance_b}" \
  bash "${script_dir}/start_persistent_engine.sh" "${model}" "${gpu_b}"

mkdir -p "${runtime}/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
orchestrator_log="${runtime}/logs/benchmark-${model}-dual-gpu${gpu_a}-gpu${gpu_b}-${timestamp}.log"

echo "launching shard 0/2 on GPU ${gpu_a} and shard 1/2 on GPU ${gpu_b}"
RPX_VQA_ENGINE_INSTANCE="${instance_a}" \
  bash "${script_dir}/run_persistent_benchmark.sh" \
    "${model}" "${manifest}" 0 2 "$@" \
    > >(tee -a "${orchestrator_log}") 2>&1 &
pid_a=$!

RPX_VQA_ENGINE_INSTANCE="${instance_b}" \
  bash "${script_dir}/run_persistent_benchmark.sh" \
    "${model}" "${manifest}" 1 2 "$@" \
    > >(tee -a "${orchestrator_log}") 2>&1 &
pid_b=$!

status_a=0
status_b=0
wait "${pid_a}" || status_a=$?
wait "${pid_b}" || status_b=$?

echo "shard 0/2 exit=${status_a}"
echo "shard 1/2 exit=${status_b}"
echo "orchestrator_log=${orchestrator_log}"

if (( status_a != 0 || status_b != 0 )); then
  echo "dual-GPU benchmark failed; both resident engines were left running for diagnosis/resume" >&2
  exit 1
fi

echo "dual-GPU benchmark complete; both resident engines remain available"

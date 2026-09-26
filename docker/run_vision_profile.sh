#!/usr/bin/env bash
set -euo pipefail

if (($# < 5)); then
  echo "Usage: $0 TASK MODEL GPU IMAGE PYTHON [task-runner arguments...]" >&2
  exit 2
fi
task="$1" model="$2" gpu="$3" image="$4" python="$5"
shift 5
case "$task" in image-depth|video-depth|tracking|rcpe) ;; *) exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git_sha="$(git -C "$repo" rev-parse HEAD)"
cache="${RPX_HF_CACHE:?Set RPX_HF_CACHE to the existing host Hugging Face cache}"
base="${RPX_PROFILE_ROOT:?Set RPX_PROFILE_ROOT to a writable NEW profiling output root}"
split="${RPX_PROFILE_SPLIT:-easy}"
run="${RPX_PROFILE_RUN_NAME:-${task}-${model}-${split}-$(date -u +%Y%m%dT%H%M%S)-$$}"
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid run name' >&2; exit 2; }
mkdir -p "$base"
[[ -d "$cache" ]] || { echo "Missing cache: $cache" >&2; exit 1; }
image_id="$(docker image inspect "$image" --format '{{.Id}}')"
# Preserve ALL native task code, configuration classes and registries. The
# entrypoint installs temporary method hooks in this disposable process.
overlay=()
for relative in \
  scripts/profile_vision.py \
  scripts/hardware_runtime_hooks.py \
  rpx_benchmark/hardware_profile.py
do
  [[ -f "$repo/benchmark/$relative" ]] || { echo "Missing $relative" >&2; exit 1; }
  overlay+=(--mount "type=bind,src=$repo/benchmark/$relative,dst=/opt/rpx/benchmark/$relative,readonly")
done
# Some historical caches place dataset repositories directly in HF_HOME.
# Expose that same repository at HF_HUB_CACHE without copying or changing it.
dataset_repo="${RPX_PROFILE_DATASET_REPO:-}"
if [[ -n "$dataset_repo" ]]; then
  [[ -d "$dataset_repo/snapshots" ]] || { echo "Missing dataset snapshots: $dataset_repo" >&2; exit 1; }
  overlay+=(--mount "type=bind,src=$dataset_repo,dst=/cache/huggingface/hub/datasets--IRVLUTD--RPX,readonly")
elif [[ ! -d "$cache/hub/datasets--IRVLUTD--RPX" && -d "$cache/datasets--IRVLUTD--RPX" ]]; then
  overlay+=(--mount "type=bind,src=$cache/datasets--IRVLUTD--RPX,dst=/cache/huggingface/hub/datasets--IRVLUTD--RPX,readonly")
fi
# Expose direct-layout model repositories at the standard hub path without
# changing the host cache, copying weights, or merging different snapshots.
for model_repo in "$cache"/models--*; do
  [[ -d "$model_repo" ]] || continue
  model_name="${model_repo##*/}"
  if [[ ! -d "$cache/hub/$model_name" ]]; then
    overlay+=(--mount "type=bind,src=$model_repo,dst=/cache/huggingface/hub/$model_name,readonly")
  fi
done
device_args=(--gpus "device=${gpu}" --ipc=host)
profile_args=()
if [[ "${RPX_PROFILE_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  device_args=(--network none)
  profile_args=(--preflight-only)
fi
docker run --rm --name "rpx-profile-${run}" "${device_args[@]}" \
  -e HF_TOKEN -e HF_HOME=/cache/huggingface -e HF_HUB_CACHE=/cache/huggingface/hub \
  -e HF_HUB_OFFLINE -e TRANSFORMERS_OFFLINE \
  -e PYTHONUNBUFFERED=1 \
  -e "RPX_GIT_SHA=${git_sha}" \
  -e "RPX_PROFILE_IMAGE_ID=${image_id}" -e "RPX_PROFILE_HOST_GPU=${gpu}" \
  -v "$cache:/cache/huggingface" -v "$base:/profiles" \
  "${overlay[@]}" \
  --workdir /opt/rpx/benchmark --entrypoint "$python" "$image_id" \
  scripts/profile_vision.py --task "$task" --model "$model" --split "$split" \
  --samples "${RPX_PROFILE_SAMPLES:-1000}" --warmup-calls "${RPX_PROFILE_WARMUP_CALLS:-1}" \
  --flop-calls "${RPX_PROFILE_FLOP_CALLS:-1}" --output-dir "/profiles/$run" "${profile_args[@]}" -- "$@" \
  2>&1 | tee "$base/$run.log"

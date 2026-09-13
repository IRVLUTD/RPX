#!/usr/bin/env bash

# Shared naming helpers for resident VQA engines.  RPX_VQA_ENGINE_INSTANCE is
# optional so all existing single-engine commands keep their historical
# container names.  Set it when several replicas of the same model run on
# different GPUs (for example, gpu3 and gpu4).

rpx_vqa_validate_engine_instance() {
  local instance="${1:-}"
  if [[ -n "${instance}" && ! "${instance}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "RPX_VQA_ENGINE_INSTANCE must match [a-zA-Z0-9][a-zA-Z0-9_.-]*" >&2
    return 2
  fi
}

rpx_vqa_engine_instance_suffix() {
  local instance="${RPX_VQA_ENGINE_INSTANCE:-}"
  rpx_vqa_validate_engine_instance "${instance}" || return
  if [[ -n "${instance}" ]]; then
    printf -- '-%s' "${instance}"
  fi
}

rpx_vqa_engine_name() {
  local model="${1:?rpx_vqa_engine_name requires MODEL}"
  local safe_model="${model//[^a-zA-Z0-9_.-]/-}"
  local instance_suffix
  instance_suffix="$(rpx_vqa_engine_instance_suffix)" || return
  printf 'rpx-vqa-%s%s' "${safe_model}" "${instance_suffix}"
}

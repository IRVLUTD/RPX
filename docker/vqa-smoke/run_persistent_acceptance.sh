#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: run_persistent_acceptance.sh MODEL}"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=persistent_engine_lib.sh
source "${script_dir}/persistent_engine_lib.sh"
runtime="${RPX_VQA_RUNTIME:-/data/rpx/vqa-runtime}"
name="$(rpx_vqa_engine_name "${model}")"
instance_suffix="$(rpx_vqa_engine_instance_suffix)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${runtime}/logs/acceptance-${model}${instance_suffix}-${timestamp}.log"

docker ps --format '{{.Names}}' | grep -Fxq "${name}" || {
  echo "resident engine is not running: ${name}" >&2
  exit 1
}

docker exec "${name}" /usr/local/bin/rpx-vqa-smoke \
  acceptance-remote "${model}" "$@" 2>&1 | tee "${log}"
echo "engine remains resident: ${name}"
echo "log=${log}"

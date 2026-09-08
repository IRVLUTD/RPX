#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: run_persistent_acceptance.sh MODEL}"
runtime="${RPX_VQA_RUNTIME:-/data/narendhiran_rpx/vqa-runtime}"
name="rpx-vqa-${model//[^a-zA-Z0-9_.-]/-}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${runtime}/logs/acceptance-${model}-${timestamp}.log"

docker ps --format '{{.Names}}' | grep -Fxq "${name}" || {
  echo "resident engine is not running: ${name}" >&2
  exit 1
}

docker exec "${name}" /usr/local/bin/rpx-vqa-smoke \
  acceptance-remote "${model}" "$@" 2>&1 | tee "${log}"
echo "engine remains resident: ${name}"
echo "log=${log}"

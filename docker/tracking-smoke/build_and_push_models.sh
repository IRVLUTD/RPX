#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
registry="${RPX_TRACKING_IMAGE:-vndhiran123/rpx-tracking-smoke}"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
push=false
host_python="${RPX_HOST_PYTHON:-}"

case "${1:-}" in
  "")
    ;;
  --push)
    push=true
    ;;
  *)
    echo "usage: $0 [--push]" >&2
    exit 2
    ;;
esac

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

if [[ -z "${host_python}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    host_python="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    host_python="$(command -v python)"
  else
    echo "Python 3 is required on the Docker host to read model-matrix.json." >&2
    exit 1
  fi
fi

mapfile -t rows < <(
  "${host_python}" - "${script_dir}/model-matrix.json" <<'PY'
import json
import sys

for model in json.load(open(sys.argv[1]))["models"]:
    print(model["id"], model["target"])
PY
)

if [[ "${#rows[@]}" -ne 10 ]]; then
  echo "Expected 10 tracking model rows, found ${#rows[@]}." >&2
  exit 1
fi

for row in "${rows[@]}"; do
  read -r model target <<<"${row}"
  immutable="${registry}:${model}-sha-${short_revision}"
  moving="${registry}:${model}-latest"

  echo
  echo "Building ${model}: ${target}"
  docker build \
    --file "${script_dir}/Dockerfile.models" \
    --target "${target}" \
    --build-arg "RPX_GIT_SHA=${revision}" \
    --label "org.opencontainers.image.revision=${revision}" \
    --label "org.opencontainers.image.version=${model}-sha-${short_revision}" \
    --tag "${immutable}" \
    --tag "${moving}" \
    "${repo_root}"

  docker inspect "${immutable}" \
    --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}}'

  if [[ "${push}" == true ]]; then
    docker push "${immutable}"
    docker push "${moving}"
  fi
done

final_immutable="${registry}:all-sha-${short_revision}"
final_moving="${registry}:all-latest"
docker tag "${registry}:grounded-sam2-sha-${short_revision}" "${final_immutable}"
docker tag "${registry}:grounded-sam2-sha-${short_revision}" "${final_moving}"

if [[ "${push}" == true ]]; then
  docker push "${final_immutable}"
  docker push "${final_moving}"
fi

echo
echo "Complete: ${registry}:all-sha-${short_revision}"

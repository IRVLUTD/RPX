#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
registry="${RPX_TRACKING_IMAGE:-vndhiran123/rpx-tracking-smoke}"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
push=false
host_python="${RPX_HOST_PYTHON:-}"
selected_model=""
base_image=""

usage() {
  cat <<EOF
usage:
  $0 [--push]
  $0 --model MODEL [--base-image IMAGE] [--push]
  $0 --list

Without --model, build all nine cumulative targets in paper order.
With --model, build only that target. --base-image makes that target inherit
an already-built/published preceding model image instead of rebuilding it.
EOF
}

list_only=false
while (($#)); do
  case "$1" in
    --push)
      push=true
      shift
      ;;
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
      selected_model="$2"
      shift 2
      ;;
    --base-image)
      [[ $# -ge 2 ]] || { echo "--base-image requires a value" >&2; exit 2; }
      base_image="$2"
      shift 2
      ;;
    --list)
      list_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${base_image}" && -z "${selected_model}" ]]; then
  echo "--base-image requires --model." >&2
  exit 2
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
    print(model["id"], model["target"], model.get("base_arg", "-"))
PY
)

if [[ "${#rows[@]}" -ne 9 ]]; then
  echo "Expected 9 tracking model rows, found ${#rows[@]}." >&2
  exit 1
fi

if [[ "${list_only}" == true ]]; then
  printf '%s\n' "${rows[@]}"
  exit 0
fi

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

if [[ -n "${selected_model}" ]]; then
  found=false
  for row in "${rows[@]}"; do
    read -r model _target _base_arg <<<"${row}"
    if [[ "${model}" == "${selected_model}" ]]; then
      found=true
      break
    fi
  done
  if [[ "${found}" != true ]]; then
    echo "Unknown tracking model: ${selected_model}" >&2
    exit 2
  fi
fi

built_grounded_sam2=false
for row in "${rows[@]}"; do
  read -r model target base_arg <<<"${row}"
  if [[ -n "${selected_model}" && "${model}" != "${selected_model}" ]]; then
    continue
  fi
  immutable="${registry}:${model}-sha-${short_revision}"
  moving="${registry}:${model}-latest"

  echo
  echo "Building ${model}: ${target}"
  build_args=(
    --file "${script_dir}/Dockerfile.models" \
    --target "${target}" \
    --build-arg "RPX_GIT_SHA=${revision}" \
    --label "org.opencontainers.image.revision=${revision}" \
    --label "org.opencontainers.image.version=${model}-sha-${short_revision}" \
    --tag "${immutable}" \
    --tag "${moving}" \
    "${repo_root}"
  )
  if [[ -n "${base_image}" ]]; then
    if [[ "${base_arg}" == "-" ]]; then
      echo "${model} is the first model and cannot use --base-image." >&2
      exit 2
    fi
    docker image inspect "${base_image}" >/dev/null 2>&1 || docker pull "${base_image}"
    build_args=(
      --file "${script_dir}/Dockerfile.models"
      --target "${target}"
      --build-arg "RPX_GIT_SHA=${revision}"
      --build-arg "${base_arg}=${base_image}"
      --label "org.opencontainers.image.revision=${revision}"
      --label "org.opencontainers.image.version=${model}-sha-${short_revision}"
      --tag "${immutable}"
      --tag "${moving}"
      "${repo_root}"
    )
  fi
  docker build "${build_args[@]}"

  docker inspect "${immutable}" \
    --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}}'

  if [[ "${push}" == true ]]; then
    docker push "${immutable}"
    docker push "${moving}"
  fi
  if [[ "${model}" == "grounded-sam2" ]]; then
    built_grounded_sam2=true
  fi
done

if [[ "${built_grounded_sam2}" == true ]]; then
  final_immutable="${registry}:all-sha-${short_revision}"
  final_moving="${registry}:all-latest"
  docker tag "${registry}:grounded-sam2-sha-${short_revision}" "${final_immutable}"
  docker tag "${registry}:grounded-sam2-sha-${short_revision}" "${final_moving}"

  if [[ "${push}" == true ]]; then
    docker push "${final_immutable}"
    docker push "${final_moving}"
  fi
  echo
  echo "Complete: ${final_immutable}"
else
  echo
  echo "Complete: ${registry}:${selected_model:-matrix}-sha-${short_revision}"
fi

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
image="${RPX_DEPTH_IMAGE:-vndhiran123/rpx-depth-smoke}"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
tag="${RPX_DEPTH_TAG:-sha-${short_revision}}"
wheelhouse="${RPX_TORCH_WHEELHOUSE:-${repo_root}/../depth-smoke-envs/.wheelhouse}"
empty_wheelhouse=""

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

if [[ ! -d "${wheelhouse}" ]]; then
  empty_wheelhouse="$(mktemp -d)"
  wheelhouse="${empty_wheelhouse}"
  trap 'rm -rf "${empty_wheelhouse}"' EXIT
  echo "No local torch wheelhouse found; the build will use the cu128 index."
else
  echo "Using read-only torch wheelhouse: ${wheelhouse}"
fi

docker build \
  --file "${script_dir}/Dockerfile" \
  --build-context "torch_wheelhouse=${wheelhouse}" \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --label "org.opencontainers.image.revision=${revision}" \
  --tag "${image}:${tag}" \
  --tag "${image}:latest" \
  "${repo_root}"

echo "Built ${image}:${tag} (${revision})"

if [[ "${1:-}" == "--push" ]]; then
  docker push "${image}:${tag}"
  docker push "${image}:latest"
fi

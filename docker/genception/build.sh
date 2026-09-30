#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short="${revision:0:12}"
image="${RPX_GENCEPTION_IMAGE:-rpx-genception}"

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

docker buildx build --load \
  --file "${repo_root}/docker/genception/Dockerfile" \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --tag "${image}:sha-${short}" \
  --tag "${image}:latest" \
  "${repo_root}"

echo "Built ${image}:sha-${short} and ${image}:latest"

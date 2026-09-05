#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
image="${RPX_VQA_IMAGE:-vndhiran123/rpx-vqa-smoke}"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
tag="${RPX_VQA_TAG:-paligemma-sha-${short_revision}}"

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

docker buildx build --load \
  --file "${script_dir}/Dockerfile" \
  --target vqa_paligemma \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --label "org.opencontainers.image.revision=${revision}" \
  --tag "${image}:${tag}" \
  --tag "${image}:paligemma" \
  "${repo_root}"

echo "Built ${image}:${tag} (${revision})"

if [[ "${1:-}" == "--push" ]]; then
  docker push "${image}:${tag}"
  docker push "${image}:paligemma"
fi

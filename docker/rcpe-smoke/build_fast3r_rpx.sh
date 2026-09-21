#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
registry="${RPX_RCPE_IMAGE:-rpx-rcpe-smoke}"
base_image="${RPX_RCPE_BASE_IMAGE:-rpx-rcpe-smoke:pi3x-rpx-sha-ae8a893234b9}"
push=false

usage() { echo "usage: $0 [--base-image IMAGE] [--push]"; }
while (($#)); do
  case "$1" in
    --base-image) base_image="$2"; shift 2 ;;
    --push) push=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

docker pull "${base_image}"
if [[ "${base_image}" == *@sha256:* ]]; then
  pinned_base="${base_image}"
else
  mapfile -t digests < <(docker image inspect "${base_image}" --format '{{range .RepoDigests}}{{println .}}{{end}}')
  pinned_base="${digests[0]:-}"
fi
[[ "${pinned_base}" == *@sha256:* ]] || {
  echo "Could not resolve ${base_image} to an immutable digest." >&2
  exit 1
}

revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
immutable="${registry}:fast3r-rpx-sha-${short_revision}"
moving="${registry}:fast3r-rpx-latest"

docker build \
  --file "${script_dir}/Dockerfile.fast3r-rpx" \
  --build-arg "BASE_IMAGE=${pinned_base}" \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --tag "${immutable}" \
  --tag "${moving}" \
  "${repo_root}"

if [[ "${push}" == true ]]; then
  docker push "${immutable}"
  docker push "${moving}"
  docker buildx imagetools inspect "${immutable}"
fi

echo "base=${pinned_base}"
echo "immutable=${immutable}"
echo "moving=${moving}"

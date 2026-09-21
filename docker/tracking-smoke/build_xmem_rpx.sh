#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
registry="${RPX_TRACKING_IMAGE:-narendhiranv04/rpx-tracking-smoke}"
base_image="${RPX_XMEM_BASE_IMAGE:-${registry}:mits-rpx-latest}"
push=false

usage() {
  echo "usage: $0 [--base-image MITS_IMAGE] [--push]"
}

while (($#)); do
  case "$1" in
    --base-image)
      [[ $# -ge 2 ]] || { echo "--base-image requires a value" >&2; exit 2; }
      base_image="$2"
      shift 2
      ;;
    --push)
      push=true
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

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

docker pull "${base_image}"
if [[ "${base_image}" == *@sha256:* ]]; then
  pinned_base="${base_image}"
else
  mapfile -t repo_digests < <(
    docker image inspect "${base_image}" \
      --format '{{range .RepoDigests}}{{println .}}{{end}}'
  )
  pinned_base="${repo_digests[0]:-}"
fi
if [[ "${pinned_base}" != *@sha256:* ]]; then
  echo "Could not resolve ${base_image} to an immutable repository digest." >&2
  exit 1
fi

revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
immutable="${registry}:xmem-rpx-sha-${short_revision}"
moving="${registry}:xmem-rpx-latest"

echo "Building XMem on ${pinned_base}"
docker build \
  --file "${script_dir}/Dockerfile.xmem-cumulative" \
  --build-arg "BASE_IMAGE=${pinned_base}" \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --label "org.opencontainers.image.version=xmem-rpx-sha-${short_revision}" \
  --tag "${immutable}" \
  --tag "${moving}" \
  "${repo_root}"

docker image inspect "${immutable}" \
  --format 'built {{.RepoTags}} id={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'

if [[ "${push}" == true ]]; then
  docker push "${immutable}"
  docker push "${moving}"
  docker buildx imagetools inspect "${immutable}"
fi

echo "base=${pinned_base}"
echo "immutable=${immutable}"
echo "moving=${moving}"

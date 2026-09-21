#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
registry="${RPX_TRACKING_IMAGE:-narendhiranv04/rpx-tracking-smoke}"
base_image="${RPX_SAM2_BASE_IMAGE:-narendhiranv04/rpx-tracking-smoke:sam2-sha-cc347e5a3b2b@sha256:b3e0d935b6898049848a046e24d9b3cc4a451cb536081ca9a02a22ff841b98aa}"
push=false

usage() {
  echo "usage: $0 [--base-image DIGEST_PINNED_IMAGE] [--push]"
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

if [[ "${base_image}" != *@sha256:* ]]; then
  echo "Refusing a mutable base; --base-image must include @sha256:<digest>." >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi

revision="$(git -C "${repo_root}" rev-parse HEAD)"
short_revision="${revision:0:12}"
immutable="${registry}:sam2-rpx-sha-${short_revision}"
moving="${registry}:sam2-rpx-latest"

docker pull "${base_image}"
docker build \
  --file "${script_dir}/Dockerfile.sam2-rpx" \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --label "org.opencontainers.image.version=sam2-rpx-sha-${short_revision}" \
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

echo "immutable=${immutable}"
echo "moving=${moving}"

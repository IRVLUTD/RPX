#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {sam3.1|grounded-sam2} [--base-image IMAGE] [--push]" >&2
  exit 2
fi
model="$1"
shift
case "${model}" in
  sam3.1) default_base="${RPX_TRACKING_IMAGE:-narendhiranv04/rpx-tracking-smoke}:sam3.1-latest" ;;
  grounded-sam2) default_base="${RPX_TRACKING_IMAGE:-narendhiranv04/rpx-tracking-smoke}:grounded-sam2-latest" ;;
  *) echo "model must be sam3.1 or grounded-sam2" >&2; exit 2 ;;
esac
base_image="${default_base}"
push=false
while (($#)); do
  case "$1" in
    --base-image) base_image="$2"; shift 2 ;;
    --push) push=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "Refusing to build an uncommitted RPX tree." >&2
  exit 1
fi
docker pull "${base_image}"
mapfile -t digests < <(docker image inspect "${base_image}" --format '{{range .RepoDigests}}{{println .}}{{end}}')
pinned_base="${digests[0]:-}"
if [[ "${pinned_base}" != *@sha256:* ]]; then
  echo "Could not resolve ${base_image} to an immutable digest." >&2
  exit 1
fi
revision="$(git -C "${repo_root}" rev-parse HEAD)"
short="${revision:0:12}"
registry="${RPX_TRACKING_IMAGE:-narendhiranv04/rpx-tracking-smoke}"
safe_model="${model//./_}"
safe_model="${safe_model//-/_}"
immutable="${registry}:${model}-text-rpx-sha-${short}"
moving="${registry}:${model}-text-rpx-latest"
docker build \
  --file "${script_dir}/Dockerfile.text-rpx" \
  --build-arg "BASE_IMAGE=${pinned_base}" \
  --build-arg "RPX_GIT_SHA=${revision}" \
  --build-arg "MODEL=${model}" \
  --label "org.opencontainers.image.version=${model}-text-rpx-sha-${short}" \
  --tag "${immutable}" \
  --tag "${moving}" \
  "${repo_root}"
if [[ "${push}" == true ]]; then
  docker push "${immutable}"
  docker push "${moving}"
fi
echo "immutable=${immutable}"
echo "moving=${moving}"
echo "runtime_python=/opt/rpx-envs/${safe_model}/bin/python"

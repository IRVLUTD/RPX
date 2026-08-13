#!/usr/bin/env bash
set -euo pipefail

model=""
revision=""
output_root="${TRACK_OUTPUT:-/data/narendhiran_rpx/docker-smoke/tracking-paper-outputs}"

usage() {
  echo "usage: $0 --model MODEL --revision FULL_RPX_SHA [--output-root DIR]"
}

while (($#)); do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
      model="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || { echo "--revision requires a value" >&2; exit 2; }
      revision="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; exit 2; }
      output_root="$2"
      shift 2
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

if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ || -z "${model}" ]]; then
  usage >&2
  exit 2
fi

acceptance_dir="${output_root}/${revision}/${model}/acceptance"
frames_dir="${acceptance_dir}/prediction_frames"
if [[ ! -f "${frames_dir}/manifest.json" ]]; then
  echo "Missing rendered acceptance frames: ${frames_dir}" >&2
  exit 1
fi

short_revision="${revision:0:12}"
archive="${output_root}/${model}-acceptance-${short_revision}-prediction-frames.tar.gz"
tar -C "${acceptance_dir}" -czf "${archive}" prediction_frames
(
  cd "${output_root}"
  sha256sum "$(basename "${archive}")" > "$(basename "${archive}").sha256"
)

cat "${archive}.sha256"
ls -lh "${archive}" "${archive}.sha256"

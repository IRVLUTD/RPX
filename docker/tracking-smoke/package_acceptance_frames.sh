#!/usr/bin/env bash
set -euo pipefail

model=""
revision=""
output_root="${TRACK_OUTPUT:-/data/narendhiran_rpx/docker-smoke/tracking-paper-outputs}"
dataset_protocol="mos"

usage() {
  echo "usage: $0 --model MODEL --revision FULL_RPX_SHA [--dataset-protocol mos|ego] [--output-root DIR]"
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
    --dataset-protocol)
      [[ $# -ge 2 ]] || { echo "--dataset-protocol requires a value" >&2; exit 2; }
      dataset_protocol="$2"
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

if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ || -z "${model}" ||
      ! "${dataset_protocol}" =~ ^(mos|ego)$ ]]; then
  usage >&2
  exit 2
fi

acceptance_dir="${output_root}/${revision}/${model}/acceptance"
archive_protocol=""
if [[ "${dataset_protocol}" == "ego" ]]; then
  acceptance_dir="${output_root}/${revision}/${model}/ego/acceptance"
  archive_protocol="-ego"
fi
frames_dir="${acceptance_dir}/prediction_frames"
if [[ ! -f "${frames_dir}/manifest.json" ]]; then
  echo "Missing rendered acceptance frames: ${frames_dir}" >&2
  exit 1
fi

short_revision="${revision:0:12}"
archive="${output_root}/${model}${archive_protocol}-acceptance-${short_revision}-prediction-frames.tar.gz"
archive_members=(prediction_frames)
if [[ -d "${acceptance_dir}/open_vocabulary_predictions" ]]; then
  archive_members+=(open_vocabulary_predictions)
fi
tar -C "${acceptance_dir}" -czf "${archive}" "${archive_members[@]}"
(
  cd "${output_root}"
  sha256sum "$(basename "${archive}")" > "$(basename "${archive}").sha256"
)

cat "${archive}.sha256"
ls -lh "${archive}" "${archive}.sha256"

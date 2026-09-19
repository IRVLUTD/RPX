#!/usr/bin/env bash
set -euo pipefail

model=""
gate="acceptance"
output_root=""
protocol=""
while (($#)); do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --gate) gate="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --protocol) protocol="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${model}" && -n "${output_root}" ]] || {
  echo "usage: $0 --model MODEL [--gate acceptance|sos] [--protocol NAME] --output-root DIR" >&2
  exit 2
}

display="${model}"
[[ "${model}" == "depthsplat" ]] && display="DepthSplat"
if [[ "${gate}" == "sos" ]]; then
  source_dir="${output_root}/${display}/sos"
  archive_suffix="sos"
  if [[ -n "${protocol}" ]]; then
    source_dir="${source_dir}/${protocol}"
    archive_suffix="sos-${protocol}"
  fi
else
  source_dir="${output_root}/${model}/${gate}/${display}/easy"
  archive_suffix="${gate}"
fi
[[ -d "${source_dir}" ]] || { echo "missing results: ${source_dir}" >&2; exit 1; }

archive="${output_root}/${model}-${archive_suffix}-results.tar.gz"
tar -C "$(dirname "${source_dir}")" -czf "${archive}" "$(basename "${source_dir}")"
(cd "$(dirname "${archive}")" && sha256sum "$(basename "${archive}")") \
  | tee "${archive}.sha256"
ls -lh "${archive}" "${archive}.sha256"

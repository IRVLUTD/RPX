#!/usr/bin/env bash
set -euo pipefail

model=""
gate="acceptance"
output_root=""
while (($#)); do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --gate) gate="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${model}" && -n "${output_root}" ]] || {
  echo "usage: $0 --model MODEL [--gate acceptance|sos] --output-root DIR" >&2
  exit 2
}

display="${model}"
[[ "${model}" == "depthsplat" ]] && display="DepthSplat"
if [[ "${gate}" == "sos" ]]; then
  source_dir="${output_root}/${display}/sos"
else
  source_dir="${output_root}/${model}/${gate}/${display}/easy"
fi
[[ -d "${source_dir}" ]] || { echo "missing results: ${source_dir}" >&2; exit 1; }

archive="${output_root}/${model}-${gate}-results.tar.gz"
tar -C "$(dirname "${source_dir}")" -czf "${archive}" "$(basename "${source_dir}")"
sha256sum "${archive}" | tee "${archive}.sha256"
ls -lh "${archive}" "${archive}.sha256"

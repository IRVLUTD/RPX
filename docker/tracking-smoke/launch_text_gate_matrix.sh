#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(smoke|micro|acceptance)$ ]]; then
  echo "usage: $0 {smoke|micro|acceptance}" >&2
  exit 2
fi
gate="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
vocab="${RPX_TRACKING_TEXT_VOCAB:-${repo_root}/tracking/metadata/text_initialization_v1/scene_condition_vocab.parquet}"
session="rpx-text-${gate}"
[[ -f "${vocab}" ]] || { echo "Vocabulary parquet not found: ${vocab}" >&2; exit 1; }
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  echo "inspect it with: tmux attach -t ${session}" >&2
  exit 1
fi

# A long-lived tmux server does not automatically inherit arbitrary variables
# from the invoking shell. Copy only the runtime variables these panes need;
# this also avoids placing the HF token in pane command lines.
for name in \
  HF_TOKEN RPX_TRACKING_IMAGE RPX_TRACKING_RUNTIME RPX_HF_CACHE \
  RPX_TRACKING_DATA_CACHE RPX_TRACKING_OUTPUT RPX_TRACKING_TEXT_VOCAB; do
  if [[ -v "${name}" ]]; then
    tmux set-environment -g "${name}" "${!name}"
  fi
done

jobs=(
  "grounded-mos grounded-sam2 mos 0"
  "grounded-ego grounded-sam2 ego 1"
  "sam31-mos sam3.1 mos 2"
  "sam31-ego sam3.1 ego 3"
)
first=true
for specification in "${jobs[@]}"; do
  read -r window model protocol gpu <<<"${specification}"
  printf -v command \
    'sleep 1; cd %q && exec %q %q %q %q %q %q' \
    "${repo_root}" "${script_dir}/run_text_gate.sh" \
    "${model}" "${gate}" "${protocol}" "${gpu}" "${vocab}"
  if [[ "${first}" == true ]]; then
    tmux new-session -d -s "${session}" -n "${window}" "${command}"
    tmux set-option -t "${session}" remain-on-exit on >/dev/null
    first=false
  else
    tmux new-window -d -t "${session}" -n "${window}" "${command}"
  fi
done

echo "Launched ${gate} on four GPUs in tmux session ${session}."
echo "Attach: tmux attach -t ${session}"
echo "Status: tmux list-panes -s -t ${session} -F '#{window_name} dead=#{pane_dead} exit=#{pane_dead_status}'"

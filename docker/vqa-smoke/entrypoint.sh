#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help)
    cat <<'EOF'
RPX VQA cumulative smoke image — PaliGemma stage

Commands:
  verify                         Check imports and CUDA visibility.
  smoke paligemma2-3b [args]     Run all 36 rows and score the result.
  smoke paligemma2-10b [args]    Same dependency image; use both A5000s.
  shell                          Open Bash.
EOF
    ;;
  verify)
    exec python -c "import json,torch,transformers; assert torch.cuda.is_available(), 'CUDA unavailable'; print(json.dumps({'rpx_git_sha':'${RPX_GIT_SHA}','torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'transformers':transformers.__version__},indent=2))"
    ;;
  smoke)
    model="${1:-paligemma2-3b}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}"
    mkdir -p "${run_dir}"
    python scripts/run_paligemma2_smoke.py \
      --model "${model}" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --predictions "${run_dir}/predictions.jsonl" \
      "$@"
    exec python scripts/run_vqa_smoke_gate.py \
      --manifest data/vqa_smoke/v1/manifest.jsonl \
      --model "${model}" \
      --predictions "${run_dir}/predictions.jsonl" \
      --report "${run_dir}/report.json"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac

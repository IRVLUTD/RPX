#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help)
    cat <<'EOF'
RPX VQA cumulative smoke image

Commands:
  verify                         Check imports and CUDA visibility.
  smoke gemma4-12b [args]        Run the 14-row bbox smoke gate.
  acceptance gemma4-12b [args]   Run 700 bbox rows covering every current cell.
  shell                          Open Bash.
EOF
    ;;
  verify)
    exec python -c "import json,torch,transformers; assert torch.cuda.is_available(), 'CUDA unavailable'; print(json.dumps({'rpx_git_sha':'${RPX_GIT_SHA}','torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'transformers':transformers.__version__},indent=2))"
    ;;
  smoke|acceptance)
    gate="${command_name}"
    model="${1:-gemma4-12b}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}/${gate}"
    mkdir -p "${run_dir}"
    manifest="${run_dir}/manifest.jsonl"
    if [[ "${model}" == gemma4-* ]]; then
      python scripts/fetch_vqa_smoke_parquets.py --out "${RPX_VQA_CACHE}/parquets"
      if [[ "${gate}" == "smoke" ]]; then
        python scripts/build_vqa_smoke_sample.py \
          --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
      else
        python scripts/build_vqa_acceptance_sample.py \
          --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
      fi
      python scripts/run_gemma4_smoke.py \
        --model "${model}" --manifest "${manifest}" \
        --image-cache "${RPX_VQA_CACHE}/images" \
        --predictions "${run_dir}/predictions.jsonl" "$@"
    else
      manifest="data/vqa_smoke/v1/manifest.jsonl"
      python scripts/run_paligemma2_smoke.py \
        --model "${model}" --image-cache "${RPX_VQA_CACHE}/images" \
        --predictions "${run_dir}/predictions.jsonl" "$@"
    fi
    exec python scripts/run_vqa_smoke_gate.py \
      --manifest "${manifest}" \
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

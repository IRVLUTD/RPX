#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help)
    cat <<'EOF'
RPX VQA vLLM-only smoke and acceptance image

Commands:
  verify                         Check vLLM, CUDA and single-GPU visibility.
  list-models                    Print the frozen ten-model vLLM roster.
  smoke MODEL [args]             Run the 14-row bbox smoke gate.
  acceptance MODEL [args]        Run the 700-row current-data acceptance gate.
  shell                          Open Bash.
EOF
    ;;
  verify)
    exec python3 -c "import json,vllm,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() == 1, 'expose exactly one GPU'; print(json.dumps({'backend':'vllm','vllm':vllm.__version__,'rpx_git_sha':'${RPX_GIT_SHA}','torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},indent=2))"
    ;;
  list-models)
    exec env PYTHONPATH=scripts python3 -c "from vqa_models.vllm_backend import CHECKPOINTS; print('\n'.join(f'{key}\t{cfg.repo_id}@{cfg.revision}' for key,cfg in CHECKPOINTS.items()))"
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
    python3 scripts/fetch_vqa_smoke_parquets.py --out "${RPX_VQA_CACHE}/parquets"
    if [[ "${gate}" == "smoke" ]]; then
      python3 scripts/build_vqa_smoke_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    else
      python3 scripts/build_vqa_acceptance_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    fi
    python3 scripts/run_vllm_vqa.py \
      --model "${model}" --manifest "${manifest}" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --predictions "${run_dir}/predictions.jsonl" \
      --resume "$@"
    exec python3 scripts/run_vqa_smoke_gate.py \
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

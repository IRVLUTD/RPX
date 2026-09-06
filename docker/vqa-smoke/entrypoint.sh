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
  smoke paligemma2-3b [args]     Run all 36 rows and score the result.
  smoke paligemma2-10b [args]    Same dependency image; use both A5000s.
  smoke gemma4-12b [args]        Bbox-only RPX smoke on one 48 GB GPU.
  shell                          Open Bash.
EOF
    ;;
  verify)
    exec python -c "import json,torch,transformers; assert torch.cuda.is_available(), 'CUDA unavailable'; print(json.dumps({'rpx_git_sha':'${RPX_GIT_SHA}','torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'transformers':transformers.__version__},indent=2))"
    ;;
  smoke)
    model="${1:-gemma4-12b}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}"
    mkdir -p "${run_dir}"
    manifest="${run_dir}/manifest.jsonl"
    if [[ "${model}" == gemma4-* ]]; then
      python scripts/fetch_vqa_smoke_parquets.py --out "${RPX_VQA_CACHE}/parquets"
      python scripts/build_vqa_smoke_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
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

#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help)
    cat <<'EOF'
RPX VQA vLLM-only smoke, acceptance and benchmark image. Normal (one-image)
and in-context (two-image) tasks are both supported; see model-matrix.json.

Commands:
  verify                                   Check vLLM, CUDA and single-GPU visibility.
  list-models                              Print the frozen twelve-model vLLM roster.
  smoke MODEL [args]                       Run the 14-row bbox smoke gate.
  smoke-remote MODEL [args]                Run smoke through the resident engine.
  acceptance MODEL [args]                  Run the 104-row normal+in-context acceptance gate.
  serve MODEL [args]                       Keep one model resident on localhost:8000.
  acceptance-remote MODEL [args]           Run acceptance through the resident engine.
  diagnostic-remote MODEL [args]           Run semantic/oracle/end-to-end diagnostics.
  benchmark MODEL MANIFEST SHARD_INDEX SHARD_COUNT [args]
                                            Run one shard of the full benchmark plan
                                            (see build_vqa_benchmark_plan.py), batched.
  benchmark-remote MODEL MANIFEST SHARD_INDEX SHARD_COUNT [args]
                                            Run through a resident engine on localhost.
  benchmark-prefetch MANIFEST                Fetch and verify all benchmark images once.
  shell                                    Open Bash.
EOF
    ;;
  verify)
    exec python3 -c "import json,vllm,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() == 1, 'expose exactly one GPU'; print(json.dumps({'backend':'vllm','vllm':vllm.__version__,'rpx_git_sha':'${RPX_GIT_SHA}','torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},indent=2))"
    ;;
  list-models)
    exec env PYTHONPATH=scripts python3 -c "from vqa_models.vllm_backend import CHECKPOINTS; print('\n'.join(f'{key}\t{cfg.repo_id}@{cfg.revision}' for key,cfg in CHECKPOINTS.items()))"
    ;;
  serve)
    model="${1:?serve requires MODEL}"
    shift
    exec python3 scripts/serve_vllm_vqa.py \
      --model "${model}" --image-cache "${RPX_VQA_CACHE}/images" "$@"
    ;;
  smoke|smoke-remote|acceptance|acceptance-remote)
    gate="${command_name}"
    if [[ "${gate}" == *-remote ]]; then
      gate="${gate%-remote}"
      remote_args=(--server-url http://127.0.0.1:8000)
    else
      remote_args=()
    fi
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
      --resume "${remote_args[@]}" "$@"
    exec python3 scripts/run_vqa_smoke_gate.py \
      --manifest "${manifest}" \
      --model "${model}" \
      --predictions "${run_dir}/predictions.jsonl" \
      --report "${run_dir}/report.json"
    ;;
  diagnostic-remote)
    model="${1:?diagnostic-remote requires MODEL}"
    shift
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}/diagnostic"
    mkdir -p "${run_dir}"
    manifest="${run_dir}/manifest.jsonl"
    python3 scripts/fetch_vqa_smoke_parquets.py --out "${RPX_VQA_CACHE}/parquets"
    python3 scripts/build_vqa_acceptance_sample.py \
      --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    exec python3 scripts/run_vqa_diagnostic.py \
      --model "${model}" --manifest "${manifest}" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --out "${run_dir}/diagnostic_predictions.jsonl" \
      --server-url http://127.0.0.1:8000 "$@"
    ;;
  benchmark|benchmark-remote)
    mode="${command_name}"
    model="${1:?${mode} requires MODEL}"
    manifest="${2:?${mode} requires MANIFEST}"
    shard_index="${3:?${mode} requires SHARD_INDEX}"
    shard_count="${4:?${mode} requires SHARD_COUNT}"
    shift 4
    if [[ "${mode}" == "benchmark-remote" ]]; then
      remote_args=(--server-url http://127.0.0.1:8000 --batch-size 1)
    else
      remote_args=()
    fi
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}/benchmark/shard-${shard_index}-of-${shard_count}"
    mkdir -p "${run_dir}"
    exec python3 scripts/run_vqa_benchmark.py \
      --model "${model}" --manifest "${manifest}" \
      --shard-index "${shard_index}" --shard-count "${shard_count}" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --predictions "${run_dir}/predictions.jsonl" \
      --failures "${run_dir}/failures.jsonl" \
      --report "${run_dir}/report.json" \
      --run-config-out "${run_dir}/run_config.json" \
      --resume "${remote_args[@]}" "$@"
    ;;
  benchmark-prefetch)
    manifest="${1:?benchmark-prefetch requires MANIFEST}"
    exec python3 scripts/prefetch_vqa_benchmark.py \
      --manifest "${manifest}" --image-cache "${RPX_VQA_CACHE}/images"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac

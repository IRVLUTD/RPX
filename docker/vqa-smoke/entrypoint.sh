#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

python_for_model() {
  if [[ "$1" == internvl3.5-* ]]; then
    printf '%s\n' /opt/rpx-envs/internvl/bin/python
  elif [[ "$1" == florence2-* || "$1" == paligemma2-* ]]; then
    printf '%s\n' /opt/rpx-envs/florence2/bin/python
  else
    printf '%s\n' python3
  fi
}

case "${command_name}" in
  help)
    cat <<'EOF'
RPX VQA smoke, acceptance and benchmark image. Normal (one-image)
and in-context (two-image) tasks are both supported; see model-matrix.json.

Commands:
  verify                                   Check vLLM, CUDA and single-GPU visibility.
  list-models                              Print the frozen model/backend roster.
  smoke MODEL [args]                       Run the 14-row bbox smoke gate.
  smoke-remote MODEL [args]                Run smoke through the resident engine.
  acceptance MODEL [args]                  Run the 104-row normal+in-context acceptance gate.
  serve MODEL [args]                       Keep one model resident on localhost:8000.
  acceptance-remote MODEL [args]           Run acceptance through the resident engine.
  diagnostic-smoke-remote MODEL [args]     Run 14-row unscored native diagnostics and gallery.
  diagnostic-acceptance-remote MODEL [args]
                                            Run 104-row unscored native diagnostics and gallery.
  diagnostic-remote MODEL [args]           Alias for diagnostic-acceptance-remote.
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
    florence_transformers="$([ -x /opt/rpx-envs/florence2/bin/python ] && /opt/rpx-envs/florence2/bin/python -c 'import transformers; print(transformers.__version__)')"
    internvl_transformers="$([ -x /opt/rpx-envs/internvl/bin/python ] && /opt/rpx-envs/internvl/bin/python -c 'import transformers; print(transformers.__version__)')"
    exec python3 -c "import json,torch,transformers,vllm; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() in (1,2), 'expose one GPU, or two only for a TP2 model'; print(json.dumps({'backends':['vllm','transformers-florence2','transformers-paligemma2','transformers-internvl'],'vllm':vllm.__version__,'transformers':transformers.__version__,'native_transformers':'${florence_transformers}','internvl_transformers':'${internvl_transformers}','rpx_git_sha':'${RPX_GIT_SHA}','torch':torch.__version__,'cuda':torch.version.cuda,'gpu_count':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0)},indent=2))"
    ;;
  list-models)
    exec env PYTHONPATH=scripts python3 -c "from vqa_models.backend_registry import CHECKPOINTS,backend_name; print('\n'.join(f'{key}\t{backend_name(key)}\t{cfg.repo_id}@{cfg.revision}' for key,cfg in CHECKPOINTS.items()))"
    ;;
  serve)
    model="${1:?serve requires MODEL}"
    shift
    python_bin="$(python_for_model "${model}")"
    exec "${python_bin}" scripts/serve_vllm_vqa.py \
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
    python_bin="$(python_for_model "${model}")"
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}/${gate}"
    mkdir -p "${run_dir}"
    manifest="${run_dir}/manifest.jsonl"
    "${python_bin}" scripts/fetch_vqa_smoke_parquets.py --out "${RPX_VQA_CACHE}/parquets"
    if [[ "${gate}" == "smoke" ]]; then
      "${python_bin}" scripts/build_vqa_smoke_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    else
      "${python_bin}" scripts/build_vqa_acceptance_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    fi
    "${python_bin}" scripts/run_vllm_vqa.py \
      --model "${model}" --manifest "${manifest}" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --predictions "${run_dir}/predictions.jsonl" \
      --resume "${remote_args[@]}" "$@"
    "${python_bin}" scripts/run_vqa_smoke_gate.py \
      --manifest "${manifest}" \
      --model "${model}" \
      --predictions "${run_dir}/predictions.jsonl" \
      --report "${run_dir}/report.json"
    if [[ "${gate}" == "acceptance" ]]; then
      "${python_bin}" scripts/build_vqa_acceptance_gallery.py \
        --manifest "${manifest}" \
        --predictions "${run_dir}/predictions.jsonl" \
        --model "${model}" \
        --image-cache "${RPX_VQA_CACHE}/images" \
        --out "${run_dir}/gallery.html"
    fi
    ;;
  diagnostic-smoke-remote|diagnostic-acceptance-remote|diagnostic-remote)
    diagnostic_gate="acceptance"
    if [[ "${command_name}" == "diagnostic-smoke-remote" ]]; then
      diagnostic_gate="smoke"
    fi
    model="${1:?${command_name} requires MODEL}"
    shift
    python_bin="$(python_for_model "${model}")"
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}/diagnostic-${diagnostic_gate}"
    mkdir -p "${run_dir}"
    manifest="${run_dir}/manifest.jsonl"
    "${python_bin}" scripts/fetch_vqa_smoke_parquets.py --out "${RPX_VQA_CACHE}/parquets"
    if [[ "${diagnostic_gate}" == "smoke" ]]; then
      "${python_bin}" scripts/build_vqa_smoke_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    else
      "${python_bin}" scripts/build_vqa_acceptance_sample.py \
        --parquet-dir "${RPX_VQA_CACHE}/parquets" --out "${manifest}"
    fi
    "${python_bin}" scripts/run_vqa_diagnostic.py \
      --model "${model}" --manifest "${manifest}" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --out "${run_dir}/diagnostic_predictions.jsonl" \
      --server-url http://127.0.0.1:8000 "$@"
    exec "${python_bin}" scripts/build_vqa_diagnostic_gallery.py \
      --manifest "${manifest}" \
      --predictions "${run_dir}/diagnostic_predictions.jsonl" \
      --image-cache "${RPX_VQA_CACHE}/images" \
      --model "${model}" \
      --out "${run_dir}/gallery.html"
    ;;
  benchmark|benchmark-remote)
    mode="${command_name}"
    model="${1:?${mode} requires MODEL}"
    manifest="${2:?${mode} requires MANIFEST}"
    shard_index="${3:?${mode} requires SHARD_INDEX}"
    shard_count="${4:?${mode} requires SHARD_COUNT}"
    shift 4
    python_bin="$(python_for_model "${model}")"
    if [[ "${mode}" == "benchmark-remote" ]]; then
      remote_args=(--server-url http://127.0.0.1:8000 --batch-size 1)
    else
      remote_args=()
    fi
    run_dir="${RPX_VQA_OUTPUTS}/${model}/sha-${RPX_GIT_SHA:0:12}/benchmark/shard-${shard_index}-of-${shard_count}"
    mkdir -p "${run_dir}"
    exec "${python_bin}" scripts/run_vqa_benchmark.py \
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
    shift
    exec python3 scripts/prefetch_vqa_benchmark.py \
      --manifest "${manifest}" --image-cache "${RPX_VQA_CACHE}/images" "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help)
    cat <<'EOF'
RPX GenCeption runtime

Commands:
  verify                         Verify CUDA, JAX and pinned source.
  capabilities                   Print executable RPX task coverage.
  download {1.3b|14b|all}        Download and size-check official assets.
  smoke [arguments]              Run run_genception_smoke.py.
  shell                          Open Bash.
  COMMAND [arguments]            Execute any command in /opt/rpx/benchmark.
EOF
    ;;
  verify)
    exec python -c "import json,jax,torch,transformers; assert torch.cuda.is_available(), 'CUDA unavailable'; assert jax.devices('gpu'), 'JAX GPU unavailable'; print(json.dumps({'rpx_git_sha':'${RPX_GIT_SHA}','jax':jax.__version__,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0)},indent=2))"
    ;;
  capabilities)
    exec python scripts/genception_capabilities.py --json
    ;;
  download)
    variant="${1:-all}"
    exec python scripts/download_genception.py \
      --variant "${variant}" --output "${RPX_GENCEPTION_ROOT}"
    ;;
  smoke)
    exec python scripts/run_genception_smoke.py "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac

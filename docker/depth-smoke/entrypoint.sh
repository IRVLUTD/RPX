#!/usr/bin/env bash
set -euo pipefail

readonly MODELS="da-v2-large,depth-pro,unidepth-v2,moge-2-vit-l"
readonly ENV_ROOT="${RPX_ENV_ROOT:-/opt/rpx-envs}"
readonly CACHE_DIR="${HF_HOME:-/cache/huggingface}"
readonly OUTPUT_ROOT="${RPX_SMOKE_ROOT:-/outputs}"

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help)
    cat <<'EOF'
RPX four-model depth smoke image

Commands:
  verify             Check imports and CUDA visibility without downloading weights.
  list               Show all 20 canonical rows and their safety-rail status.
  smoke [extra args] Run micro then acceptance for the four accepted image models.
  shell              Open an interactive Bash shell.
  <command...>       Execute an arbitrary command inside the image.

Runtime mounts:
  /cache/huggingface  Persistent Hugging Face data and checkpoint cache.
  /outputs            Smoke result and matrix state directory.
EOF
    ;;
  verify)
    exec python -c "import json, torch, transformers, moge; from unidepth.models import UniDepthV2; assert torch.cuda.is_available(), 'CUDA is unavailable; install/configure NVIDIA Container Toolkit and use --gpus all'; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'transformers': transformers.__version__, 'models': '${MODELS}'.split(',')}, indent=2))"
    ;;
  list)
    exec python scripts/run_depth_smoke_matrix.py --list-models
    ;;
  smoke)
    exec python scripts/run_depth_smoke_matrix.py \
      --task image \
      --models "${RPX_MODELS:-${MODELS}}" \
      --gates "${RPX_GATES:-micro,acceptance}" \
      --env-root "${ENV_ROOT}" \
      --cache-dir "${CACHE_DIR}" \
      --output-root "${OUTPUT_ROOT}" \
      --gpu-index "${RPX_GPU_INDEX:-0}" \
      "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac

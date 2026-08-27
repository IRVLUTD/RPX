#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_nvs_gates.sh --model MODEL --image IMAGE [--python PATH] [--gate smoke|micro|acceptance|all]

Required environment:
  NVS_GPU, HF_CACHE, TORCH_CACHE, NVS_OUTPUT, RPX_REVISION
Optional:
  HF_TOKEN
EOF
  exit 2
}

model=""
image=""
python_path="python"
gate="all"
while (($#)); do
  case "$1" in
    --model) model="${2:-}"; shift 2 ;;
    --image) image="${2:-}"; shift 2 ;;
    --python) python_path="${2:-}"; shift 2 ;;
    --gate) gate="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "$model" && -n "$image" ]] || usage
[[ "$gate" =~ ^(smoke|micro|acceptance|all)$ ]] || usage
: "${NVS_GPU:?set NVS_GPU}"
: "${HF_CACHE:?set HF_CACHE}"
: "${TORCH_CACHE:?set TORCH_CACHE}"
: "${NVS_OUTPUT:?set NVS_OUTPUT}"
: "${RPX_REVISION:?set RPX_REVISION}"

snapshot="/cache/huggingface/datasets--IRVLUTD--RPX/snapshots/${RPX_REVISION}"
mkdir -p "$HF_CACHE" "$TORCH_CACHE" "$NVS_OUTPUT/logs"

docker pull "$image"

gates=(smoke micro acceptance)
[[ "$gate" == all ]] || gates=("$gate")
for current_gate in "${gates[@]}"; do
  echo "Running ${model} ${current_gate} gate on GPU ${NVS_GPU}"
  docker run --rm \
    --gpus "device=${NVS_GPU}" \
    --shm-size=32g \
    -e HF_TOKEN \
    -e HF_HOME=/cache/huggingface \
    -e TORCH_HOME=/cache/torch \
    -v "$HF_CACHE:/cache/huggingface" \
    -v "$TORCH_CACHE:/cache/torch" \
    -v "$NVS_OUTPUT:/outputs" \
    "$image" \
    "$python_path" /opt/rpx/benchmark/scripts/run_nvs_gate.py \
      --model "$model" \
      --gate "$current_gate" \
      --split easy \
      --device cuda \
      --extracted-root "$snapshot/extracted" \
      --parquet-path "$snapshot/manifest/frames_v1.parquet" \
      --output-root /outputs \
    2>&1 | tee "$NVS_OUTPUT/logs/${model}-${current_gate}.log"
done

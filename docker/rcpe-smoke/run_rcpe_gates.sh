#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_rcpe_gates.sh --model MODEL --image IMAGE [--python PATH] [--gate smoke|micro|acceptance|all]

Required environment: RCPE_GPU, HF_CACHE, RCPE_OUTPUT, RPX_REVISION
Optional environment: HF_TOKEN
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
: "${RCPE_GPU:?set RCPE_GPU}"
: "${HF_CACHE:?set HF_CACHE}"
: "${RCPE_OUTPUT:?set RCPE_OUTPUT}"
: "${RPX_REVISION:?set RPX_REVISION}"

mkdir -p "$HF_CACHE" "$RCPE_OUTPUT/logs"
docker pull "$image"

gates=(smoke micro acceptance)
[[ "$gate" == all ]] || gates=("$gate")
for current_gate in "${gates[@]}"; do
  echo "Running ${model} ${current_gate} on GPU ${RCPE_GPU}"
  docker run --rm \
    --gpus "device=${RCPE_GPU}" \
    --shm-size=32g \
    -e HF_TOKEN \
    -e HF_HOME=/cache/huggingface \
    -v "$HF_CACHE:/cache/huggingface" \
    -v "$RCPE_OUTPUT:/outputs" \
    "$image" \
    "$python_path" /opt/rpx/benchmark/scripts/run_relative_pose_gate.py \
      --model "$model" \
      --gate "$current_gate" \
      --split easy \
      --repo IRVLUTD/RPX \
      --revision "$RPX_REVISION" \
      --device cuda \
      --output-root /outputs \
    2>&1 | tee "$RCPE_OUTPUT/logs/${model}-${current_gate}.log"
done

#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="/media/naren/New Volume/rpx/docker_mask_annotation_test"
SOURCE_SCENE="${SOURCE_SCENE:-/media/naren/New Volume/rpx/ego_example/2/2}"
SAMPLE_DIR="${TEST_ROOT}/sample_scene"
FRAME_START="${FRAME_START:-0}"
FRAME_COUNT="${FRAME_COUNT:-10}"

if [[ ! -d "${SOURCE_SCENE}/rgb" ]]; then
  echo "Source scene is missing rgb/: ${SOURCE_SCENE}" >&2
  exit 1
fi

case "${SAMPLE_DIR}" in
  "${TEST_ROOT}/sample_scene") ;;
  *)
    echo "Refusing to prepare unexpected sample directory: ${SAMPLE_DIR}" >&2
    exit 1
    ;;
esac

rm -rf "${SAMPLE_DIR}" "${TEST_ROOT}/hf_cache"
mkdir -p \
  "${SAMPLE_DIR}/rgb" \
  "${SAMPLE_DIR}/depth" \
  "${SAMPLE_DIR}/cam_pose" \
  "${SAMPLE_DIR}/fisheye/left" \
  "${SAMPLE_DIR}/fisheye/right" \
  "${TEST_ROOT}/checkpoints" \
  "${TEST_ROOT}/hf_cache/hub" \
  "${TEST_ROOT}/outputs"

last=$((FRAME_START + FRAME_COUNT - 1))
for idx in $(seq "${FRAME_START}" "${last}"); do
  stem="$(printf "%05d" "${idx}")"
  for rel in \
    "rgb/${stem}.png" \
    "depth/${stem}.png" \
    "cam_pose/${stem}.npz" \
    "fisheye/left/${stem}.png" \
    "fisheye/right/${stem}.png"; do
    if [[ -f "${SOURCE_SCENE}/${rel}" ]]; then
      cp -f "${SOURCE_SCENE}/${rel}" "${SAMPLE_DIR}/${rel}"
    else
      echo "Missing expected file: ${SOURCE_SCENE}/${rel}" >&2
      exit 1
    fi
  done
done

checkpoint="$(find "/media/naren/New Volume/rpx" \( -name "sam2.1_hiera_large.pth" -o -name "*.pth" \) 2>/dev/null | head -1 || true)"
if [[ -n "${checkpoint}" ]]; then
  ln -sfn "${checkpoint}" "${TEST_ROOT}/checkpoints/sam2.1_hiera_large.pth"
  echo "Linked checkpoint: ${checkpoint}"
else
  echo "No .pth checkpoint found under /media/naren/New Volume/rpx"
  echo "Place sam2.1_hiera_large.pth in ${TEST_ROOT}/checkpoints/ before full segmentation."
fi

chmod -R a+rwX "${SAMPLE_DIR}" "${TEST_ROOT}/checkpoints" "${TEST_ROOT}/hf_cache" "${TEST_ROOT}/outputs" 2>/dev/null || true

echo "Prepared sample scene: ${SAMPLE_DIR}"
find "${SAMPLE_DIR}" -maxdepth 2 -type f | sort | sed -n "1,80p"

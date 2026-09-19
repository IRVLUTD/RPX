export RPX_LOCAL_ROOT="/media/naren/New Volume/rpx"
export RPX_SOURCE_REPO="$RPX_LOCAL_ROOT/github_rpx"
export RPX_TRACKING_REPO="$RPX_LOCAL_ROOT/github_rpx_tracking"

export HOST_GPU=0
export RPX_GPU_INDEX=0
export RPX_RUN_HOST=naren-popos-gpu0

export RPX_TRACKING_IMAGE=vndhiran123/rpx-tracking-smoke
export RPX_TRACKING_CACHE="$RPX_LOCAL_ROOT/docker-tracking-smoke-cache"
export RPX_TRACKING_OUTPUT="$RPX_LOCAL_ROOT/smoke-results-tracking"

mkdir -p \
  "$RPX_TRACKING_CACHE/ultralytics/models" \
  "$RPX_TRACKING_CACHE/ultralytics/config" \
  "$RPX_TRACKING_CACHE/huggingface" \
  "$RPX_TRACKING_CACHE/torch" \
  "$RPX_TRACKING_CACHE/xdg" \
  "$RPX_TRACKING_OUTPUT"

cd "$RPX_TRACKING_REPO"

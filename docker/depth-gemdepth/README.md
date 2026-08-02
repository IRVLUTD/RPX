# GemDepth RPX overlay

This overlay adds the official GemDepth source and its isolated pinned
environment to an existing cumulative RPX video-depth image.

The adapter verifies the pinned Hugging Face checkpoint SHA-256 before strict
state-dict loading. GemDepth emits inverse depth; RPX follows the official
evaluation convention by fitting one affine transform in disparity space over
the complete scene-phase clip and then converting to metric depth.

```bash
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  -t "rpx-video-gemdepth:$(git rev-parse --short=12 HEAD)" \
  -f docker/depth-gemdepth/Dockerfile .
```

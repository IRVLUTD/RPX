# GemDepth overlay

Adds the official GemDepth source, in its own pinned environment, to the cumulative video-depth image. There is no
published image; build it yourself.

The adapter checks the pinned checkpoint's SHA-256 before loading the state dict strictly. GemDepth predicts inverse
depth, so, following the official evaluation, RPX fits one affine transform in disparity space over each full
scene-phase clip and converts the result to metric depth.

```bash
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  -t "rpx-video-gemdepth:$(git rev-parse --short=12 HEAD)" \
  -f docker/depth-gemdepth/Dockerfile .
```

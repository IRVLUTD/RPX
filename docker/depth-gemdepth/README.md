# GemDepth overlay

Adds the official GemDepth source, in its own pinned environment, to the cumulative video-depth image. There is no
published image; build it yourself.

## How RPX runs GemDepth

The adapter checks the pinned checkpoint's SHA-256 before loading the state dict strictly. GemDepth predicts inverse
depth, so, following the official evaluation, RPX fits one affine transform in disparity space over each full
scene-phase clip and converts the result to metric depth.

## Build

```bash
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  -t "rpx-video-gemdepth:$(git rev-parse --short=12 HEAD)" \
  -f docker/depth-gemdepth/Dockerfile .
```

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="../README.md">Model environments</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a></sub>

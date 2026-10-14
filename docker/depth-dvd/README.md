# DVD v1.1 overlay

Adds the official EnVision-Research/DVD source, at a pinned revision and in its own environment, to the cumulative
video-depth image. Published as `irvlutd/rpx:depth-dvd-latest`.

The adapter pins and verifies the DVD v1.1 checkpoint and pins the Wan2.1 base-model revision. DVD predicts relative
inverse depth, so RPX fits one affine transform in disparity space over each full scene-phase clip. Inference uses the
authors' 81-frame windows with 21-frame overlap and their Global Affine Coherence implementation.

```bash
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  --tag "rpx-video-dvd:$(git rev-parse --short=12 HEAD)" \
  --file docker/depth-dvd/Dockerfile .
```

The DVD checkpoint (about 4.5 GB) and the Wan2.1 weights are not in the image. They download into the mounted Hugging
Face cache on the first smoke run and are reused afterwards.

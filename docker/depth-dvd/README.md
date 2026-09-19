# DVD v1.1 RPX overlay

This thin overlay adds the official EnVision-Research/DVD source at a pinned
Git revision and an isolated DVD environment to an existing cumulative RPX
video-depth image.

The adapter pins and verifies the official DVD v1.1 checkpoint. It also pins
the Wan2.1 base-model revision. DVD emits relative inverse depth, so RPX fits
one affine transform in disparity space over each complete scene-phase clip.
The full run uses the authors' 81-frame windows, 21-frame overlap, and Global
Affine Coherence implementation.

```bash
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  --tag "rpx-video-dvd:$(git rev-parse --short=12 HEAD)" \
  --file docker/depth-dvd/Dockerfile .
```

The image does not bake the roughly 4.5 GB DVD v1.1 checkpoint or Wan2.1 base
weights. They are downloaded once into the mounted Hugging Face cache during
the first smoke test and reused by the acceptance and production runs.

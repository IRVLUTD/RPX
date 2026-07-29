# RPX tracking Docker overlay

Build the cumulative SAM 2 target from the repository root:

```bash
SHA="$(git rev-parse --short=12 HEAD)"
docker build \
  --file docker/tracking-smoke/Dockerfile \
  --target tracking_all_sam2 \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  --tag "rpx-tracking-sam2:${SHA}" \
  .
```

The target pins:

- SAM 2 source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- `facebook/sam2-hiera-large` model revision
  `e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`
- TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`

See `benchmark/docs/tracking_runbook.md` for the dataset, smoke, production
and resume protocol.

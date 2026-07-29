# RPX tracking Docker images

There are two deliberately separate Docker paths:

- `Dockerfile` is the already-tested YOLOE engineering smoke plus the RPX SAM 2
  adapter image.
- `Dockerfile.models` is the weight-free, cumulative environment matrix for all
  ten paper D3 tracking models.

The paper matrix is source-pinned and dependency-locked. It does **not** claim
that an environment has passed RPX inference merely because it builds: the
checkpoint and GPU smoke gate is a separate, later acceptance step.

## Build and publish the paper model matrix

From a clean repository checkout:

```bash
docker login
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_and_push_models.sh --push
```

The script builds in paper order:

1. SAM 3.1
2. SAM 2
3. SAM 2++
4. SAMURAI
5. EdgeTAM
6. DeAOT
7. Cutie
8. MOTIP
9. MASA
10. Grounded-SAM2

Every target inherits the preceding target, so Docker Hub stores common layers
once. Each model still has its own Python environment under `/opt/rpx-envs` to
prevent packages with conflicting `sam2` module names from shadowing each other.
The large MASA Torch 2.1/OpenMMLab compatibility layer is isolated. No
checkpoint is baked into any layer; mount `/cache/huggingface` and
`/cache/torch` during later smoke and production runs.

Tags use both immutable and moving forms:

```text
vndhiran123/rpx-tracking-smoke:sam2-sha-<12-char-RPX-SHA>
vndhiran123/rpx-tracking-smoke:sam2-latest
vndhiran123/rpx-tracking-smoke:all-sha-<12-char-RPX-SHA>
vndhiran123/rpx-tracking-smoke:all-latest
```

`model-matrix.json` is the authoritative model/source/checkpoint provenance
record. Each built environment also contains `rpx-environment.json` and
`rpx-pip-freeze.txt`.

## Existing SAM 2 RPX smoke image

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

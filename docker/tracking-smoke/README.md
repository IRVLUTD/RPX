# RPX tracking Docker images

There are two deliberately separate Docker paths:

- `Dockerfile` is the already-tested YOLOE engineering smoke plus the RPX SAM 2
  adapter image.
- `Dockerfile.models` is the weight-free, cumulative environment matrix for all
  ten paper D3 tracking models.

The paper matrix is source-pinned and dependency-locked. It does **not** claim
that an environment has passed RPX inference merely because it builds: the
checkpoint and GPU smoke gate is a separate, later acceptance step.

`Dockerfile.rpx-adapter` is the thin code overlay used after a model-specific
RPX adapter is implemented. It updates the benchmark package without rebuilding
or duplicating the cumulative upstream environments and does not add weights.

EdgeTAM uses `Dockerfile.edgetam-rpx`, which additionally applies the
versioned non-contiguous-tensor compatibility patch required by current
PyTorch. The patch replaces an invalid `expand(...).view(...)` with the
equivalent materializing `reshape(...)`; model weights and mathematics are
otherwise unchanged.

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

To build and push one model at a time on top of the preceding published image:

```bash
docker/tracking-smoke/build_and_push_models.sh \
  --model samurai \
  --base-image vndhiran123/rpx-tracking-smoke:sam2-plus-sha-<previous-sha> \
  --push
```

Use the newly emitted immutable tag as `--base-image` for the next row. This
mode is the preferred recovery path after a late-stage failure: it neither
rebuilds nor repushes the already-successful prefix.

The DeAOT overlay applies the repository-versioned
`patches/pytorch-correlation-torch27-openmp.patch` to its pinned correlation
extension. It removes two CPU-only OpenMP pragmas that do not compile inside
PyTorch 2.7's dispatch macro. The CUDA correlation implementation used for GPU
benchmarking is unchanged.

The MOTIP overlay similarly carries
`patches/motip-cuda-toolkit-build.patch`. The upstream setup script requires a
live GPU during extension compilation; Docker builds normally expose the CUDA
toolkit but no GPU device. The patch gates compilation on `CUDA_HOME` instead.
It also updates the two deprecated tensor-type dispatch calls for PyTorch 2.7.
It still builds the official CUDA sources and does not introduce a CPU fallback.

MASA's isolated OpenMMLab environment installs
`requirements-masa-runtime.lock`. These are the pinned MMDetection runtime
packages absent from the shared environment; source packages remain installed
with `--no-deps` so the resolver cannot silently replace the locked Torch/MMCV
ABI. MASA's required `scalabel-evalAPI` source branch and the TETA package that
it imports are also commit-pinned, installed without dependency resolution,
and import-checked during the build.

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

## SAM2 RPX adapter overlay (current first-model milestone)

Build the tested adapter on the immutable published cumulative SAM2 base. The
helper refuses dirty checkouts and mutable base references, regenerates the
SAM2 environment manifest and pip freeze, and emits both commit-specific and
moving tags:

```bash
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_sam2_rpx.sh --push
```

The default base is
`sam2-sha-cc347e5a3b2b@sha256:b3e0d935b6898049848a046e24d9b3cc4a451cb536081ca9a02a22ff841b98aa`.
Override it only with `--base-image` and another digest-pinned cumulative SAM2
image. See `benchmark/docs/tracking_runbook.md` for the sequential smoke,
micro, and acceptance commands.

The next cumulative milestone adds EdgeTAM directly on that SAM2 RPX image:

```bash
docker/tracking-smoke/build_edgetam_rpx.sh --push
```

The helper resolves the SAM2 tag to its repository digest before building.
The EdgeTAM image therefore retains SAM2 while recording an immutable parent.
Acceptance gates automatically render prediction overlays; use
`package_acceptance_frames.sh` to archive and checksum them for SCP.

Cutie is the third cumulative milestone and inherits the accepted EdgeTAM
image (which already retains SAM2):

```bash
docker/tracking-smoke/build_cutie_rpx.sh --push
```

The image contains pinned Cutie source and dependencies but no weights. The
official `cutie-base-mega.pth` v1.0 release is downloaded into the mounted
runtime cache, verified against the publisher's MD5, and recorded with its
SHA-256 in RPX result metadata.

SAM2Long is the fourth cumulative milestone and inherits the accepted Cutie
image, retaining Cutie, EdgeTAM and SAM2:

```bash
docker/tracking-smoke/build_sam2long_rpx.sh --push
```

The isolated environment pins the official SAM2Long source and SAM 2.1 Hiera
Large checkpoint revision. The checkpoint stays in the mounted Hugging Face
cache. RPX uses the paper implementation's three-pathway defaults.

SAM 2++ is the fifth cumulative milestone and inherits the accepted SAM2Long
image, retaining SAM2Long, Cutie, EdgeTAM and SAM2:

```bash
docker/tracking-smoke/build_sam2_plus_rpx.sh --push
```

The isolated environment pins the official SAM2-Plus source, unified
mask-prompt predictor and `checkpoint_phase123.pt` revision. Its Hydra config
is loaded directly from the pinned source tree because the upstream wheel does
not package the `sam2_plus/configs` directory. Weights remain in the mounted
Hugging Face cache rather than an image layer.

## Legacy full SAM2 rebuild

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

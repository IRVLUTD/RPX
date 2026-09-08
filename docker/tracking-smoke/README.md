# RPX tracking Docker images

There are two deliberately separate Docker paths:

- `Dockerfile` is the already-tested YOLOE engineering smoke plus the RPX SAM 2
  adapter image.
- `Dockerfile.models` is the weight-free, cumulative environment matrix for the
  nine multi-object D3 tracking models; single-object SAMURAI is excluded.

The paper matrix is source-pinned and dependency-locked. It does **not** claim
that an environment has passed RPX inference merely because it builds: the
checkpoint and GPU smoke gate is a separate, later acceptance step.

## Text-initialized tracking: Grounded-SAM2 and SAM 3.1

These two adapters use the released RPX scene-condition vocabulary
(`primary_color + canonical object name`) and never receive a ground-truth mask
or box. Grounded-SAM2 detects every vocabulary prompt in frame zero with pinned
GroundingDINO and initializes pinned SAM 2.1 from the predicted boxes. SAM 3.1
uses each prompt through its native semantic-video API. Because neither method
receives GT spatial initialization, both are scored from frame zero.

Build the thin RPX overlays on the previously published model environments:

```bash
docker/tracking-smoke/build_text_rpx.sh grounded-sam2 --push
docker/tracking-smoke/build_text_rpx.sh sam3.1 --push
```

Then run `smoke`, `micro`, and `acceptance` independently for MOS and Ego. The
launcher requires the SHA-pinned vocabulary parquet and keeps model/data caches
and outputs under `RPX_TRACKING_RUNTIME`. Existing caches and output roots can
be selected with `RPX_HF_CACHE`, `RPX_TRACKING_DATA_CACHE`, and
`RPX_TRACKING_OUTPUT`:

```bash
docker/tracking-smoke/run_text_gate.sh grounded-sam2 smoke mos 0 "$VOCAB"
docker/tracking-smoke/run_text_gate.sh sam3.1 smoke ego 1 "$VOCAB"
```

The SAM 3.1 checkpoint repository is gated; `HF_TOKEN` must belong to an account
that has accepted its license. An Ego vocabulary row may be absent for an
upstream identity-map gap documented in the released vocabulary metadata; RPX
does not invent an object prompt for such an occurrence.

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
4. EdgeTAM
5. DeAOT
6. Cutie
7. MOTIP
8. MASA
9. Grounded-SAM2

To build and push one model at a time on top of the preceding published image:

```bash
docker/tracking-smoke/build_and_push_models.sh \
  --model motip \
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

The isolated environment pins the official SAM2-Plus source, unified predictor
and `checkpoint_phase123.pt` revision. RPX derives a tight XYXY bounding box for
each object in the first-frame instance annotation, initializes the official
box-granularity task with those boxes, and evaluates the masks predicted by the
model—including its first-frame prediction. The segmentation annotation itself
is never passed to SAM 2++. Its Hydra config is loaded directly from the pinned
source tree because the upstream wheel does not package the
`sam2_plus/configs` directory or all of the bundled `training.dataset_plus`
modules. The image prepends the pinned source tree and verifies the box utility
import during the build. Weights remain in the mounted Hugging Face cache rather
than an image layer.

SAMURAI is intentionally excluded from the RPX cumulative chain because its
official benchmark implementation is single-object tracking. MOTIP is the sixth
milestone and inherits the accepted SAM 2++ image directly:

```bash
docker/tracking-smoke/build_motip_rpx.sh --push
```

MOTIP runs its official detector and joint ID-prediction tracker with no RPX
first-frame prompt. Its tracked XYWH boxes are rasterized into instance-ID masks
for the common D3 metric pipeline. The pinned v0.1 DanceTrack checkpoint is
downloaded into the runtime cache, never baked into the image. Because that
checkpoint detects people, poor or empty detections on RPX household objects are
a valid domain-transfer result rather than a reason to use ground-truth prompts.

MASA is the seventh milestone and inherits the accepted MOTIP image directly:

```bash
docker/tracking-smoke/build_masa_rpx.sh --push
```

The adapter runs the official unified MASA-Detic model with the Detic-SwinB
open-vocabulary detector and fixed LVIS vocabulary. It receives no RPX mask,
box, object-name or text prompt. MASA's official video post-processing is
applied before score-filtered XYXY tracks are rasterized for the shared metric
pipeline. Unlike prompt-initialized trackers, detector-driven MOTIP and MASA
are scored on frame 0 because they predict it without annotation input.

## Replacement milestone: MITS

The active RPX registry replaces SAMURAI, MOTIP and MASA rather than carrying
those excluded adapters forward. MITS is built directly on the accepted SAM 2++
image:

```bash
docker/tracking-smoke/build_mits_rpx.sh --push
```

The adapter uses the official full MITS checkpoint. For every annotated object,
RPX converts the first-frame instance mask to its tight bounding rectangle and
initializes all objects together through MITS's box-to-mask transformer. The
original instance IDs are restored in every saved prediction. The initialization
frame is excluded from scored metrics under the common `box` prompt protocol.

TRACT is not represented by an adapter or image at this revision. The authors'
repository now contains a MASA fork and TraCLIP research scripts, but it provides
no released TRACT checkpoint, no release assets, and no complete end-to-end
inference command (the README's Usage section is empty). Those omissions prevent
an immutable, reproducible RPX smoke gate. An open-vocabulary detector plus an
unrelated association method must not be reported as TRACT.

## Replacement milestone: XMem

XMem follows MITS in the active cumulative chain:

```bash
docker/tracking-smoke/build_xmem_rpx.sh --push
```

The adapter pins the official XMem v1.0 `XMem.pth` release and verifies its
size and SHA-256 before loading it. RPX supplies the full multi-object instance
mask on frame zero, maps arbitrary RPX IDs into XMem's contiguous label space,
and restores the original IDs in saved predictions. The image retains the
official 480-pixel short-side resize and long-term memory defaults.

## Open-vocabulary milestone: OVTR

OVTR follows the accepted XMem image in the active cumulative chain:

```bash
docker/tracking-smoke/build_ovtr_rpx.sh --push
```

The adapter pins the official full five-frame OVTR checkpoint and both published
CLIP embedding assets. The official OpenAI CLIP package is source-pinned at
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`; it is installed without dependency
resolution so it cannot replace the cumulative image's Torch/CUDA ABI. OVTR runs
detector-driven open-vocabulary MOT against the
official 1,203-class LVIS/DetPro vocabulary, receives no RPX first-frame prompt,
and is scored from frame zero. Tracked boxes are rasterized for the shared D3
metrics, while class IDs, class names, scores, track IDs and XYXY boxes are saved
under `open_vocabulary_predictions/` and included in acceptance archives.

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

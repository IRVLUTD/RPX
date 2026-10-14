# Object tracking images (T3)

Most users only need the published images. See the [image table](../README.md#which-image-to-pull) for the full list.

| Prompt | Image | Models |
|:--|:--|:--|
| First-frame mask | `irvlutd/rpx:tracking-dam4sam-rpx-latest` | 8, [listed below](#mask-initialized-trackers) |
| Box | `irvlutd/rpx:tracking-sam3.1-bbox-rpx-latest` | SAM 3.1 |
| Text | `irvlutd/rpx:tracking-sam3.1-text-rpx-latest` | SAM 3.1 |
| Text | `irvlutd/rpx:tracking-grounded-sam2-text-rpx-latest` | Grounded SAM 2 |

The rest of this page is for building the images and running the acceptance gates. Builds are source-pinned and
dependency-locked, and no checkpoint is baked into any layer: mount `/cache/huggingface` and `/cache/torch` at run time.
An image that builds has not necessarily passed RPX inference. The checkpoint and GPU smoke gate is a separate step.

<p align="center">
  <a href="#mask-initialized-trackers">Mask-initialized trackers</a> ·
  <a href="#box--and-text-prompted-trackers">Box- and text-prompted trackers</a> ·
  <a href="#other-build-files">Other build files</a>
</p>

## Mask-initialized trackers

These trackers form one cumulative chain. Each image extends the previous one and adds its model in its own Python
environment under `/opt/rpx-envs`, so packages that ship conflicting `sam2` modules cannot shadow each other. Docker
Hub stores the shared layers once.

| Order | Model | Build script | Notes |
|:--|:--|:--|:--|
| 1 | SAM 2 | `build_sam2_rpx.sh` | base pinned by digest |
| 2 | EdgeTAM | `build_edgetam_rpx.sh` | PyTorch patch, see notes |
| 3 | Cutie | `build_cutie_rpx.sh` | official v1.0 weights |
| 4 | SAM2Long | `build_sam2long_rpx.sh` | paper defaults |
| 5 | SAM 2++ | `build_sam2_plus_rpx.sh` | box-initialized |
| 6 | MiTS | `build_mits_rpx.sh` | box-initialized |
| 7 | XMem | `build_xmem_rpx.sh` | official v1.0 weights |
| 8 | OVTR | `build_ovtr_rpx.sh` | open vocabulary |
| 9 | DAM4SAM | `build_dam4sam_rpx.sh` | published final image |

Build and push one stage with:

```bash
export RPX_TRACKING_IMAGE="narendhiranv04/rpx-tracking-smoke"
docker/tracking-smoke/build_sam2_rpx.sh --push
```

The scripts refuse dirty checkouts and mutable base references. Each one resolves its parent tag to a repository digest,
regenerates the environment manifest and pip freeze, and pushes an immutable `<model>-rpx-sha-<12-char-sha>` tag
along with a moving `<model>-rpx-latest` tag. Override the parent with `--base-image` (or the script's
`RPX_<MODEL>_BASE_IMAGE` variable). The SAM 2 base defaults to this tag and digest:

| | |
|:--|:--|
| Tag | `narendhiranv04/rpx-tracking-smoke:sam2-sha-cc347e5a3b2b` |
| Digest | `sha256:b3e0d935b6898049848a046e24d9b3cc4a451cb536081ca9a02a22ff841b98aa` |

Acceptance gates render prediction overlays automatically. `package_acceptance_frames.sh` archives and checksums them
for transfer.

### Model notes

EdgeTAM builds from `Dockerfile.edgetam-rpx`, which patches an invalid `expand(...).view(...)` to the equivalent
`reshape(...)` for current PyTorch. Weights and math are unchanged.

Cutie uses the official `cutie-base-mega.pth` v1.0, checked against the publisher's MD5; its SHA-256 is recorded in the
result metadata. SAM2Long uses a pinned source and SAM 2.1 Hiera Large revision with the paper's three-pathway
defaults.

For SAM 2++, RPX takes a tight XYXY box around each object in the first-frame instance mask and initializes the
official box-granularity task with it. Every predicted mask is scored, including the first frame. SAM 2++ never sees the
segmentation itself. The upstream wheel omits `sam2_plus/configs` and parts of `training.dataset_plus`, so the image
prepends the pinned source tree and checks the box utility import at build time.

MiTS uses the full official checkpoint. All objects are initialized together through MiTS's box-to-mask transformer, using each first-frame mask's
bounding rectangle. The original instance IDs are restored in every saved prediction. The initialization frame is
excluded from scoring, as under the common `box` prompt protocol.

XMem uses the official v1.0 `XMem.pth`, size- and SHA-256-verified before loading. RPX passes the full multi-object instance mask on frame zero. It maps RPX IDs to XMem's contiguous labels and
back, and keeps the official 480-pixel short-side resize and long-term memory defaults.

OVTR is an open-vocabulary detector. The image pins the full five-frame OVTR checkpoint, both published CLIP embedding assets, and OpenAI CLIP at
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. CLIP is installed without dependency resolution so it cannot replace the
image's Torch/CUDA build. OVTR runs open-vocabulary MOT over the official 1,203-class LVIS/DetPro vocabulary. It gets
no RPX prompt and is scored from frame zero. Tracked boxes are rasterized for the shared D3 metrics. Class IDs, names,
scores, track IDs and boxes are saved under `open_vocabulary_predictions/` and included in acceptance archives.

DAM4SAM's source is pinned at `9c954504b39ebca4c412f207be0787c26bfac85a`; its image is the published final stage.

### Excluded models

- SAMURAI is excluded because its official benchmark is single-object.
- MOTIP and MASA have build scripts (`build_motip_rpx.sh`, `build_masa_rpx.sh`) but are not in the active
  chain; MiTS replaces them.
  - MOTIP's pinned DanceTrack checkpoint detects people, so weak results on household objects are a domain-transfer
    result, not a reason to add ground-truth prompts.
  - MASA runs MASA-Detic with the Detic-SwinB detector and a fixed LVIS vocabulary. Its runtime dependencies are
    pinned in `requirements-masa-runtime.lock`.
  - Both are detector-driven and scored from frame 0.
- TRACT has no adapter. The authors' repository has no released checkpoint, no release assets and no end-to-end
  inference command, so no reproducible gate is possible. Do not report an open-vocabulary detector combined with an
  unrelated association method as TRACT.

## Box- and text-prompted trackers

### Box prompts

These runs are comparable to the paper. SAM 3.1 and Grounded SAM 2 start from tight boxes derived from the released
frame-zero GT instance mask, and frame zero is excluded from scoring.

- SAM 3.1 encodes each normalized XYXY box as its native corner points (labels 2/3). It keeps the GT object ID, adds
  every object to one multiplex session, and propagates once.
- For Grounded SAM 2, GroundingDINO is bypassed and the pinned SAM 2.1 video backend receives pixel XYXY boxes. Every
  prediction's metadata records this.

```bash
docker/tracking-smoke/build_bbox_rpx.sh grounded-sam2 --push
docker/tracking-smoke/build_bbox_rpx.sh sam3.1 --push
docker/tracking-smoke/run_bbox_gate.sh sam3.1 smoke mos 0
docker/tracking-smoke/run_bbox_gate.sh sam3.1 smoke ego 1
```

Use a new output root for box runs. Resume markers already include the code revision, but a separate root also keeps
these runs from being compared with the text-prompted ones by accident.

### Text prompts

Both models get the released scene-condition vocabulary (`primary_color + canonical object name`)
and no ground-truth mask or box, so they are scored from frame zero.

- Grounded SAM 2 detects each prompt in frame zero with pinned GroundingDINO and starts SAM 2.1 from those boxes.
- SAM 3.1 uses its native semantic-video API.

```bash
docker/tracking-smoke/build_text_rpx.sh grounded-sam2 --push
docker/tracking-smoke/build_text_rpx.sh sam3.1 --push

docker/tracking-smoke/run_text_gate.sh grounded-sam2 smoke mos 0 "$VOCAB"
docker/tracking-smoke/run_text_gate.sh sam3.1 smoke ego 1 "$VOCAB"
```

Run `smoke`, `micro` and `acceptance` separately for MOS and Ego. The launcher requires the SHA-pinned vocabulary
parquet. It keeps caches and outputs under `RPX_TRACKING_RUNTIME`; point it at existing ones with `RPX_HF_CACHE`,
`RPX_TRACKING_DATA_CACHE` and `RPX_TRACKING_OUTPUT`. On a four-GPU host,
`launch_text_gate_matrix.sh {smoke,micro,acceptance}` runs both models on both protocols at once. Wait for all four
panes to exit with status zero before moving to the next level.

The SAM 3.1 checkpoint is gated, so `HF_TOKEN` must belong to an account that has accepted its license. A few Ego
objects have no vocabulary row because of an upstream identity-map gap documented in the vocabulary metadata; RPX does
not invent prompts for them.

## Other build files

<details>
<summary>Legacy images, provenance files and patches</summary>

- `Dockerfile` is the original YOLOE engineering smoke image plus the SAM 2 adapter.
- `Dockerfile.rpx-adapter` is a thin code overlay that updates the benchmark package without rebuilding the upstream
  environments.
- `Dockerfile.models` with `build_and_push_models.sh` is the earlier nine-model matrix: SAM 3.1, SAM 2, SAM 2++,
  EdgeTAM, DeAOT, Cutie, MOTIP, MASA and Grounded SAM 2.
  - Build one stage at a time with `--model <name> --base-image <previous immutable tag> --push`. This lets you recover
    from a late failure without rebuilding the successful prefix.
  - Tags take the form `<model>-sha-<12-char-sha>` and `<model>-latest`, plus `all-sha-…` and `all-latest`.
- `model-matrix.json` is the authoritative model, source and checkpoint provenance record. Each environment also
  contains `rpx-environment.json` and `rpx-pip-freeze.txt`.

Patches in `patches/`:

- `pytorch-correlation-torch27-openmp.patch` (DeAOT) removes two CPU-only OpenMP pragmas that do not compile under
  PyTorch 2.7. The CUDA path is unchanged.
- `motip-cuda-toolkit-build.patch` gates MOTIP's extension build on `CUDA_HOME` instead of a live GPU, and updates two
  deprecated dispatch calls for PyTorch 2.7. It still builds the official CUDA sources and adds no CPU fallback.

To rebuild the original SAM 2 target from scratch:

```bash
SHA="$(git rev-parse --short=12 HEAD)"
docker build \
  --file docker/tracking-smoke/Dockerfile \
  --target tracking_all_sam2 \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  --tag "rpx-tracking-sam2:${SHA}" \
  .
```

It pins SAM 2 at `2b90b9f5ceec907a1c18123530e92e794ad901a4`, `facebook/sam2-hiera-large` at
`e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251` and TrackEval at `12c8791b303e0a0b50f753af204249e622d0281a`.

</details>

---

<sub>Part of <a href="../../README.md">RPX</a> ·
<a href="../README.md">Model environments</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a></sub>

# RPX D3 tracking runbook

## Protocol fixed by the paper

- The evaluation unit is a complete `(scene, phase)` clip (250 frames).
- Promptable trackers are initialized from the released frame-0 annotation
  using their declared mask or box prompt. Their frame 0 is preserved but
  excluded from metrics. Detector-driven trackers receive no RPX prompt,
  predict frame 0 themselves, and are scored on every frame.
- Positive mask values are persistent instance IDs within the clip.
- Predictions are evaluated as tight boxes derived from the predicted and GT
  masks using the official TrackEval implementation:
  - HOTA (mean over thresholds 0.05 through 0.95)
  - DetA (mean over thresholds 0.05 through 0.95)
  - AssA (mean over thresholds 0.05 through 0.95)
  - IDF1
  - MOTA
  - ID switches
- CLEAR and Identity association use the MOTChallenge IoU threshold 0.5.
- No depth F-score is part of D3.
- Each cell also records median propagation latency, derived throughput,
  peak CUDA allocated/reserved memory, clip wall time and parameter count.
  Result metadata identifies the GPU, CUDA runtime and PyTorch build.

YOLOE is in the paper's D2 detection roster. The historical
`run_yoloe_tracking_smoke.py` is retained only as an engineering check for
YOLOE plus ByteTrack; its synthetic bus video and text prompts are not an RPX
D3 result.

## Released ground truth

The pinned `IRVLUTD/RPX` release contains RGB, temporally consistent instance
masks and `sam2/mask_to_object.json`. It does not contain the historical
`tracklets/v1.json` referenced by older manifests. The D3 runner therefore
uses the masks as authoritative GT and downloads `sam2_meta` for ID/name
provenance.

Expected production inventory:

| Split | Frames | Scene-phase clips |
|---|---:|---:|
| Easy | 24,750 | 99 |
| Medium | 24,750 | 99 |
| Hard | 25,500 | 102 |
| Total | 75,000 | 300 |

## Resume contract

Predictions are written atomically to:

`predictions/<scene>/<phase>/<frame>.npz`

Each file contains one non-negative integer `mask` of shape `480×640`. A
scene-phase `_complete.json` marker is written only after all 250 predictions
are valid. The marker includes the full RPX adapter commit; outputs from a
different commit are never resumed. A completed clip is skipped on rerun with
zero model propagation.
If interruption occurs inside a clip, that clip is recomputed from frame 0:
SAM 2's temporal memory state cannot be reconstructed from isolated output
masks, so pretending to resume at an arbitrary frame would be invalid.

## Docker layout

`docker/tracking-smoke/Dockerfile` is cumulative:

1. `tracking_base`
2. `tracking_all_yoloe` (historical engineering smoke)
3. `tracking_all_sam2` (first paper-valid D3 model)

The SAM 2 target retains preceding environments, while Docker shares their
immutable layers. Do not run `docker system prune -a` while benchmark
containers or required images exist.

## Model acceptance status

An environment image proving that an upstream package imports is not an RPX
benchmark result. A model enters the production roster only after it has:

1. an official, source-pinned adapter;
2. a micro run on one real Easy RPX clip truncated to 8 frames;
3. an acceptance run on one real Easy RPX clip truncated to 25 frames;
4. validated atomic predictions and a second identical run with zero model
   propagation frames.

The production adapters currently exposed by `run_tracking.py` are:

| Model | Prompt supplied by RPX | Pinned checkpoint |
|---|---|---|
| SAM 2 | released GT instance mask on frame 0 | `facebook/sam2-hiera-large` |
| EdgeTAM | released GT instance mask on frame 0 | `facebook/EdgeTAM/edgetam.pt` |
| Cutie | released GT instance mask on frame 0 | `hkchengrex/Cutie/cutie-base-mega.pth` |
| SAM2Long | released GT instance mask on frame 0 | `facebook/sam2.1-hiera-large` |
| SAM 2++ | tight boxes derived from released frame-0 instances | `MCG-NJU/SAM2-Plus/checkpoint_phase123.pt` |
| MOTIP | none; native detector | `MCG-NJU/MOTIP/r50_deformable_detr_motip_dancetrack.pth` |
| MASA-Detic | none; unified open-vocabulary Detic detector | `dereksiyuanli/masa/detic_masa.pth` |

## SAM2 cumulative image and real-RPX gates

SAM2 is the first model brought through the complete gate sequence. Its thin
RPX overlay inherits the immutable published cumulative SAM2 image by digest;
it does not rebuild the earlier environment or bake model weights into a
layer. Build from a clean, committed checkout:

```bash
docker login
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_sam2_rpx.sh --push
```

The script emits an immutable `sam2-rpx-sha-<12-char-RPX-SHA>` tag and the
moving `sam2-rpx-latest` convenience tag. Use the immutable tag for every
recorded test and production run.

The gate runner uses one real Easy scene-phase clip and always performs a
fresh inference pass followed by an identical resume pass. The budgets are:

| Gate | Real RPX frames | Purpose |
|---|---:|---|
| smoke | 2 | checkpoint load and one propagation |
| micro | 8 | bounded integration run |
| acceptance | 25 | acceptance plus persisted-output validation |

Run each gate with the SAM2 virtual environment and persistent cache/output
mounts:

```bash
/opt/rpx-envs/sam2/bin/python scripts/run_tracking_gate.py \
  --model sam2 \
  --gate acceptance \
  --cache-dir /cache/huggingface \
  --output-root /outputs
```

The runner validates CUDA availability, embedded source/checkpoint provenance,
the RPX adapter commit, the checkpoint SHA-256, mask artefacts, and the
zero-forward resume contract. Acceptance additionally renders all 25 persisted
instance masks over their exact manifest RGB frames under
`prediction_frames/<scene>/<phase>/`.

## EdgeTAM cumulative image and real-RPX gates

EdgeTAM is the second accepted environment and inherits the immutable SAM2 RPX
image. Its source, isolated Python environment, compatibility patch and RPX
adapter are added without rebuilding or replacing SAM2:

```bash
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_edgetam_rpx.sh --push
```

Run the same three gates using `/opt/rpx-envs/edgetam/bin/python` and
`--model edgetam`. After acceptance, package the automatically rendered frames:

```bash
docker/tracking-smoke/package_acceptance_frames.sh \
  --model edgetam \
  --revision "$(git rev-parse HEAD)"
```

The resulting checksum file contains only the archive basename, so it remains
valid after both files are copied to another machine with SCP.

## Cutie cumulative image and real-RPX gates

Cutie is the third accepted environment. It inherits EdgeTAM and SAM2, then
adds the official pinned Cutie source and an isolated `/opt/rpx-envs/cutie`.
The adapter follows the upstream `InferenceCore.step` mask-initialized API and
uses the official v1.0 `cutie-base-mega.pth` checkpoint:

```bash
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_cutie_rpx.sh --push
```

The checkpoint is not baked into the image. On first smoke it is downloaded
to the persistent cache, verified against the official release MD5, and its
SHA-256 is included in result metadata. Run smoke, micro and acceptance with
`/opt/rpx-envs/cutie/bin/python` and `--model cutie`. Acceptance renders the
same portable `prediction_frames` layout and is packaged with the shared
`package_acceptance_frames.sh --model cutie` command.

## SAM2Long cumulative image and real-RPX gates

SAM2Long is the fourth accepted environment. It inherits Cutie, EdgeTAM and
SAM2, then installs the official source at commit
`7193b77fa0c8827e0520ab281acd2cf394ab898e` in an isolated
`/opt/rpx-envs/sam2long` environment:

```bash
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_sam2long_rpx.sh --push
```

The adapter uses the official training-free memory tree with three pathways,
IoU threshold 0.1 and uncertainty threshold 2. It uses the pinned official
SAM 2.1 Hiera Large checkpoint, preserves RPX IDs from the first-frame mask,
and records the downloaded checkpoint SHA-256. Run the standard smoke, micro
and acceptance gates using `/opt/rpx-envs/sam2long/bin/python` and
`--model sam2long`; acceptance rendering and packaging remain unchanged.

EdgeTAM uses the official SAM-style video predictor API and its own isolated
`/opt/rpx-envs/edgetam` environment. It does not import or execute the SAM 2
checkpoint. EdgeTAM's wheel omits its nested Hydra YAML, so the adapter loads
`sam2/configs/edgetam.yaml` from the commit-pinned source checkout retained at
`/opt/rpx-models/edgetam`. It fails closed if that source config is absent.

## MASA-Detic cumulative image and real-RPX gates

MASA is the seventh cumulative milestone and inherits the accepted MOTIP
image. It uses the authors' unified MASA-Detic checkpoint and the pinned
open-vocabulary Detic-SwinB configuration:

```bash
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
docker/tracking-smoke/build_masa_rpx.sh --push
```

MASA receives no RPX first-frame boxes, masks, object names, or text prompts.
Its detector discovers instances independently on every frame; MASA associates
those detections into persistent tracks. The official demo post-processing is
applied, detections above the fixed 0.2 score threshold are rasterized as
instance-ID rectangles, and all frames—including frame 0—are evaluated. The
unified checkpoint is downloaded once into the mounted cache and validated by
its pinned byte count and SHA-256; it is not baked into the image.

## EdgeTAM real-RPX smoke gates

Build a thin adapter overlay over the already-published pinned EdgeTAM
environment:

```bash
export RPX_TRACKING_IMAGE="vndhiran123/rpx-tracking-smoke"
export RPX_SHA="$(git rev-parse --short=12 HEAD)"
export EDGETAM_IMAGE="${RPX_TRACKING_IMAGE}:edgetam-rpx-${RPX_SHA}"

docker pull "${RPX_TRACKING_IMAGE}:edgetam-latest"
docker build \
  --file docker/tracking-smoke/Dockerfile.edgetam-rpx \
  --build-arg BASE_IMAGE="${RPX_TRACKING_IMAGE}:edgetam-latest" \
  --build-arg RPX_GIT_SHA="$(git rev-parse HEAD)" \
  --tag "$EDGETAM_IMAGE" \
  .
```

Use the same persistent Hugging Face cache for both gates:

```bash
export HF_HOME="/data/narendhiran_rpx/hf-cache"
export TRACK_OUTPUT="/data/narendhiran_rpx/docker-smoke/tracking-paper-outputs"
export DATASET_REVISION="2e2a387f7f93e98c177b2e039c141eacda94e5fc"
```

Micro gate:

```bash
docker run --rm \
  --name rpx-edgetam-micro \
  --gpus '"device=0"' \
  --ipc=host \
  --shm-size=16g \
  -e HF_TOKEN \
  -e HF_HOME=/cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$TRACK_OUTPUT:/outputs" \
  "$EDGETAM_IMAGE" \
  /opt/rpx-envs/edgetam/bin/python scripts/run_tracking.py \
    --model edgetam \
    --split easy \
    --revision "$DATASET_REVISION" \
    --device cuda \
    --cache-dir /cache/huggingface \
    --output-dir /outputs/edgetam-smoke/micro \
    --save-predictions \
    --resume-predictions \
    --max-clips 1 \
    --max-frames 8
```

Acceptance gate is the same command with container name
`rpx-edgetam-acceptance`, output `edgetam-smoke/acceptance`, and
`--max-frames 25`.

Run each command a second time unchanged, then validate the saved output and
the zero-forward resume contract:

```bash
docker run --rm \
  -v "$TRACK_OUTPUT:/outputs:ro" \
  "$EDGETAM_IMAGE" \
  /opt/rpx-envs/edgetam/bin/python scripts/validate_tracking_smoke.py \
    --output-dir /outputs/edgetam-smoke/acceptance \
    --model edgetam \
    --expected-clips 1 \
    --expected-frames 25 \
    --require-resume-hit
```

## Outputs

Each split writes `cells.csv`, `cells.parquet`, `result.json`,
`run_metadata.json`, and validated prediction masks. After all splits:

- `combined_cells.parquet` (exactly 300 cells)
- `paper_analysis.json`
- `paper_analysis.md`
- `paper_table.csv`

The analysis reports repeated-measures Φ/Wilks statistics, paired
Hotelling/Holm transitions, per-tier effects, phase×difficulty, and phase
means. JEDI emits `bounds_missing` until an approved versioned D3 bounds file
is supplied; it never guesses bounds.

# RPX D3 tracking runbook

## Protocol fixed by the paper

- The evaluation unit is a complete `(scene, phase)` clip (250 frames).
- Trackers are initialized with the released ground-truth instance masks on
  frame 0. Text prompts are not used for mask-initialized D3 models. Frame 0
  is preserved in the outputs but excluded from metrics because it is supplied
  to the model rather than predicted.
- Positive mask values are persistent instance IDs within the clip.
- Predictions are evaluated as tight boxes derived from the predicted and GT
  masks using the official TrackEval implementation:
  - MOTA
  - IDF1
  - HOTA (mean over thresholds 0.05 through 0.95)
  - ID switches
- CLEAR and Identity association use the MOTChallenge IoU threshold 0.5.
- No depth F-score is part of D3.

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
zero-forward resume contract.

EdgeTAM uses the official SAM-style video predictor API and its own isolated
`/opt/rpx-envs/edgetam` environment. It does not import or execute the SAM 2
checkpoint. EdgeTAM's wheel omits its nested Hydra YAML, so the adapter loads
`sam2/configs/edgetam.yaml` from the commit-pinned source checkout retained at
`/opt/rpx-models/edgetam`. It fails closed if that source config is absent.

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

# RPX GPU depth smoke image

This directory builds the reproducible, GPU-only smoke environment for the
canonical RPX image-depth and video-depth adapters. The image contains code,
pinned Python environments, and pinned upstream source checkouts. It does not
contain RPX data, model checkpoints, Hugging Face credentials, or test results;
those remain in mounted host directories.

The Dockerfile is cumulative: every named stage extends the preceding stage.
The current final stage, `depth_all_hyden`, contains runtimes for 16 models.
"Present in the image" means that the dependencies and upstream imports build;
it does not mean that the model has passed the data-backed acceptance gate.

## 1. What the tests mean

Run the gates in this order:

1. `verify`: container and CUDA preflight only.
2. `micro`: the smallest real-data, real-checkpoint inference test.
3. `acceptance`: a larger real-data inference test.
4. `easy`: the full Easy pilot, only after acceptance has passed and the
   implementation has been reviewed.

An import check or successful image build is not a micro test. A micro pass is
not an acceptance pass.

All smoke gates are CUDA-only and use:

- dataset `IRVLUTD/RPX` at revision
  `2e2a387f7f93e98c177b2e039c141eacda94e5fc`;
- the Easy split;
- batch size 1 for image models;
- the official configured checkpoint;
- standard Hugging Face HTTP transport by default;
- one selected, idle NVIDIA GPU; and
- local output only. The gate never uploads to Box or Hugging Face.

### Image-model gates

| Gate | Work performed | Required validation |
| --- | --- | --- |
| `micro` | One Easy image, batch size 1, official adapter path, saved raw prediction, FLOP profiling skipped | `num_samples == 1`, one cell, finite aggregated metrics, valid GPU and precision SystemCard fields, exactly one finite and non-degenerate `.npz` depth prediction |
| `acceptance` | 25 Easy images with the same settings | `num_samples == 25`, one cell, finite aggregated metrics, valid SystemCard fields, exactly 25 finite saved predictions |
| `easy` | All 24,750 Easy images, saved predictions, comprehensive metrics | 24,750 samples and predictions, finite aggregate metrics, one cell, and a complete `comprehensive_metrics.json` with 24,750 per-sample rows |

### Video-model gates

| Gate | Work performed | Required validation |
| --- | --- | --- |
| `micro` | One Easy clip sampled to 8 frames with stride sampling | `num_samples == 1`, one cell, valid SystemCard fields, and finite `absrel`, `rmse`, `delta1`, `delta2`, and `delta3` |
| `acceptance` | One Easy clip sampled to 25 frames with stride sampling | Same contract as micro, using the 25-frame clip |
| `easy` | All 99 Easy clips using every frame (`sampling=all`) | 99 samples and 99 cells with valid core depth metrics and SystemCard fields |

Video temporal metrics `tae`, `opw`, `tgm`, and `tcc` may be non-finite when a
short clip cannot support the metric. The five core video depth metrics above
must always be present and finite.

Every gate requires these files:

```text
result.json
cells.parquet
summary.md
run_metadata.json
run.log
pip-freeze.txt
```

The image gates also save predictions. `run_metadata.json` records the exact
RPX code identity, dataset revision, Python and PyTorch versions, GPU, upstream
SHAs, executed command, validation result, and failure classification.

## 2. Model inventory

The canonical roster contains 20 rows: 10 image and 10 video. The current final
Docker stage bakes 16 of them. Three are intentionally blocked because official
weights have not been verified, and DepthLM still needs a Docker environment.

| Model | Task | Environment | Current Docker status |
| --- | --- | --- | --- |
| `da-v2-large` | image | `transformers-image` | Baked, matrix-ready, acceptance passed |
| `da3-metric-l` | image | `da3` | Baked and matrix-ready |
| `depth-pro` | image | `transformers-image` | Baked, matrix-ready, acceptance passed |
| `depthlm` | image | `depthlm` | Runnable in the host setup workflow; not yet baked into this image |
| `fe2e` | image | — | Blocked: official weights unverified |
| `hyden` | image | `metadepth` (`hyden` alias) | Baked and matrix-ready; checkpoint access may require approval |
| `lotus-2` | image | `lotus2` | Baked and matrix-ready |
| `metric3d-v2` | image | `metric3d` | Baked and matrix-ready |
| `moge-2-vit-l` | image | `moge2` | Baked, matrix-ready, acceptance passed |
| `unidepth-v2` | image | `unidepth2` | Baked, matrix-ready, acceptance passed |
| `chrono-depth` | video | `chrono` | Baked and import-checked; matrix readiness marker still needs to be added |
| `d4rt` | video | — | Blocked: official weights unverified |
| `da3-video` | video | `da3` | Baked and matrix-ready |
| `depth-crafter` | video | `depthcrafter` | Baked and import-checked; matrix readiness marker still needs to be added |
| `gem-depth` | video | — | Blocked: official weights unverified |
| `monst3r` | video | `monst3r` | Baked and matrix-ready |
| `rolling-depth` | video | `rolling` (`rollingdepth` backing env) | Baked and matrix-ready |
| `vggt-omega` | video | `vggt` (`geometry-video` backing env) | Baked and matrix-ready |
| `video-da` | video | `video_da` (`video-depth-anything` backing env) | Baked and matrix-ready |
| `vigeo` | video | `vigeo` (`geometry-video` backing env) | Baked and matrix-ready |

The four acceptance-passed rows are the previously validated image models.
The later Docker stages prove dependency installation and imports only; run
micro and acceptance before reporting those rows as accepted.

The cumulative stages are:

| Stage | Models added |
| --- | --- |
| Published four-model base | `da-v2-large`, `depth-pro`, `unidepth-v2`, `moge-2-vit-l` |
| `depth_all_da3` | `da3-metric-l`, `da3-video` |
| `depth_all_lotus2` | `lotus-2` |
| `depth_all_metric3d` | `metric3d-v2` |
| `depth_all_geometry_video` | `vggt-omega`, `vigeo` |
| `depth_all_diffusion_video` | `chrono-depth`, `depth-crafter` |
| `depth_all_rolling_vda_video` | `rolling-depth`, `video-da` |
| `depth_all_monst3r` | `monst3r` |
| `depth_all_hyden` | `hyden` |

Confirm the code-level roster at any time with:

```bash
python3 benchmark/scripts/run_depth_smoke_matrix.py --list-models
```

## 3. Host prerequisites

Use a Linux x86-64 host with:

- Docker with BuildKit/buildx;
- an NVIDIA GPU and a driver compatible with CUDA 12.8;
- NVIDIA Container Toolkit configured for Docker;
- enough free disk for the image layers, model weights, RPX cache, and outputs;
  the complete working set can exceed 100 GB; and
- a Hugging Face account/token for gated checkpoints.

Verify GPU passthrough first:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If Docker cannot expose the GPU, configure the NVIDIA runtime and restart
Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Prepare persistent host directories. These mounts prevent repeated downloads
and preserve outputs after the container exits:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export RPX_SMOKE_ROOT="$PWD/rpx-smoke"
mkdir -p "$HF_HOME" "$RPX_SMOKE_ROOT"
```

Set `HF_TOKEN` in the host environment; do not bake it into the image:

```bash
export HF_TOKEN="hf_..."
```

## 4. Build the current branch

Clone the extended Docker branch and record the exact source revision:

```bash
git clone --branch docker/depth-all-hyden --single-branch \
  https://github.com/IRVLUTD/RPX.git rpx-depth-smoke
cd rpx-depth-smoke

export RPX_GIT_SHA="$(git rev-parse HEAD)"
export METADEPTH_SHA="$(tr -d '[:space:]' < docker/depth-smoke/metadepth.sha)"
export IMAGE="rpx-depth-smoke:sha-$(git rev-parse --short=12 HEAD)"
```

The final HyDen stage requires the pinned MetaDepth SHA explicitly. Build and
load the final cumulative image into the local Docker engine:

```bash
mkdir -p ../depth-smoke-envs/.wheelhouse

docker buildx build --load \
  --file docker/depth-smoke/Dockerfile \
  --target depth_all_hyden \
  --build-context \
    "torch_wheelhouse=../depth-smoke-envs/.wheelhouse" \
  --build-arg "RPX_GIT_SHA=$RPX_GIT_SHA" \
  --build-arg "METADEPTH_SHA=$METADEPTH_SHA" \
  --label "org.opencontainers.image.revision=$RPX_GIT_SHA" \
  --tag "$IMAGE" \
  .
```

The first extended stage starts from the immutable published four-model digest.
Each later stage creates a separate environment under `/opt/rpx-envs`, checks
its pinned upstream imports, and passes its environment forward. Build a named
intermediate stage while developing one family, for example:

```bash
docker buildx build --load \
  --file docker/depth-smoke/Dockerfile \
  --target depth_all_rolling_vda_video \
  --build-arg "RPX_GIT_SHA=$RPX_GIT_SHA" \
  --tag rpx-depth-smoke:rolling-vda-dev \
  .
```

`build_and_push.sh` is the original clean-tree four-model publishing helper.
The extended final stage additionally requires `METADEPTH_SHA`; use the explicit
build command above until the helper forwards that build argument. The helper
also refuses a dirty Git tree by design.

## 5. Verify the built image

The inherited `verify` entrypoint checks CUDA plus the original four-model
runtime without downloading weights:

```bash
docker run --rm --gpus all --ipc=host --shm-size=8g \
  "$IMAGE" verify
```

Inspect the full canonical roster:

```bash
docker run --rm "$IMAGE" list
```

Check the immutable source label:

```bash
docker image inspect "$IMAGE" \
  --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
```

The Docker build itself performs import checks for every added environment.
`verify` remains a preflight, not a smoke pass, because it does not download
weights or run RPX samples.

## 6. Run micro and acceptance

Define a small shell helper for the common runtime options:

```bash
rpx_docker_run() {
  docker run --rm --gpus all --ipc=host --shm-size=8g "$@"
}
```

### Original four image models

The `smoke` entrypoint defaults to the four previously accepted image models
and runs micro followed by acceptance, sequentially:

```bash
rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" smoke
```

Run one of them by setting `RPX_MODELS`:

```bash
rpx_docker_run \
  -e HF_TOKEN \
  -e RPX_MODELS=depth-pro \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" smoke
```

The `smoke` entrypoint is currently image-only. Use the matrix command below
for the extended image roster or video models.

### One image model, explicit gates

Select the Python executable for the model family. This example uses DA3:

```bash
rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" \
  /opt/rpx-envs/da3/bin/python \
  scripts/run_depth_smoke_gate.py \
  --task image \
  --model da3-metric-l \
  --gate micro \
  --cache-dir /cache/huggingface \
  --output-root /outputs \
  --upstream-dir /opt/rpx-envs/sources/da3

rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" \
  /opt/rpx-envs/da3/bin/python \
  scripts/run_depth_smoke_gate.py \
  --task image \
  --model da3-metric-l \
  --gate acceptance \
  --cache-dir /cache/huggingface \
  --output-root /outputs \
  --upstream-dir /opt/rpx-envs/sources/da3
```

### One video model, explicit gates

This example uses Video Depth Anything:

```bash
rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" \
  /opt/rpx-envs/video_da/bin/python \
  scripts/run_depth_smoke_gate.py \
  --task video \
  --model video-da \
  --gate micro \
  --cache-dir /cache/huggingface \
  --output-root /outputs \
  --upstream-dir /opt/rpx-envs/sources/video-depth-anything

rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" \
  /opt/rpx-envs/video_da/bin/python \
  scripts/run_depth_smoke_gate.py \
  --task video \
  --model video-da \
  --gate acceptance \
  --cache-dir /cache/huggingface \
  --output-root /outputs \
  --upstream-dir /opt/rpx-envs/sources/video-depth-anything
```

### Resumable matrix

The matrix is the preferred multi-model path. It runs one model at a time,
enforces micro before acceptance, skips gates already passed for the exact same
code identity, and writes a consolidated state file.

Run the eight matrix-ready image environments baked into the final stage:

```bash
rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" \
  python scripts/run_depth_smoke_matrix.py \
  --task image \
  --models da-v2-large,da3-metric-l,depth-pro,hyden,lotus-2,metric3d-v2,moge-2-vit-l,unidepth-v2 \
  --gates micro,acceptance \
  --env-root /opt/rpx-envs \
  --cache-dir /cache/huggingface \
  --output-root /outputs
```

Run the six currently matrix-ready video environments:

```bash
rpx_docker_run \
  -e HF_TOKEN \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$RPX_SMOKE_ROOT:/outputs" \
  "$IMAGE" \
  python scripts/run_depth_smoke_matrix.py \
  --task video \
  --models da3-video,monst3r,rolling-depth,vggt-omega,video-da,vigeo \
  --gates micro,acceptance \
  --env-root /opt/rpx-envs \
  --cache-dir /cache/huggingface \
  --output-root /outputs
```

ChronoDepth and DepthCrafter can be invoked with the explicit single-gate form
using `/opt/rpx-envs/chrono/bin/python` and
`/opt/rpx-envs/depthcrafter/bin/python`. They are not included in the matrix
command until their Docker stages write `rpx-environment.json`, which is the
matrix's readiness contract.

Use a different GPU with `--gpu-index N`. The selected GPU must be idle; do not
use `--allow-busy-gpu` for acceptance or publishable runs.

### Resume and retry rules

Matrix state is keyed by RPX code identity. Re-running the same command resumes
passed work. A changed commit or dirty diff gets a different identity and does
not inherit old passes.

- Passed gates are skipped. Use `--rerun-passed` only for a deliberate repeat.
- After correcting a dependency/API/code problem, use `--retry-failed`.
- Code and dependency failures have a two-attempt ceiling.
- Access, checkpoint-download, and CUDA OOM failures have a one-attempt ceiling.
- A CUDA OOM is a hardware classification; rerun the unchanged command on a
  higher-memory GPU rather than changing model semantics.
- The launcher never bypasses the FE2E, D4RT, or GemDepth weight safety rails.

## 7. Inspect and preserve results

Outputs are already exported to the host through the `/outputs` bind mount; no
`docker cp` is needed. The layout is:

```text
$RPX_SMOKE_ROOT/
├── matrix/<code-id>/<host>/matrix.json
└── <code-id>/<host>/<model>/<gate>/<UTC timestamp>/
    ├── result.json
    ├── cells.parquet
    ├── summary.md
    ├── run_metadata.json
    ├── run.log
    ├── pip-freeze.txt
    └── predictions/                 # image gates
```

Check pass/fail state without guessing from console output:

```bash
find "$RPX_SMOKE_ROOT" -name run_metadata.json -print
find "$RPX_SMOKE_ROOT" -name matrix.json -print
```

Keep failed runs. `run.log` and `run_metadata.json` distinguish dependency,
access, weight-download, upstream-API, shape/dtype, data/manifest, CUDA OOM, and
unknown code failures.

### Export the image to another machine

Use `docker save`, not `docker export`. `docker save` preserves image layers,
labels, the entrypoint, and the immutable tag:

```bash
docker save "$IMAGE" | gzip > "rpx-depth-smoke-$(git rev-parse --short=12 HEAD).tar.gz"
```

Transfer the archive and load it on another NVIDIA Docker host:

```bash
gunzip -c rpx-depth-smoke-<sha>.tar.gz | docker load
docker run --rm --gpus all <loaded-image-tag> verify
```

Export the host-mounted results separately if needed:

```bash
tar -C "$RPX_SMOKE_ROOT" -czf rpx-smoke-results.tar.gz .
```

### Publish to a registry

Tag both an immutable revision and, only after validation, a moving release tag:

```bash
export REGISTRY_IMAGE="<registry>/<namespace>/rpx-depth-smoke"
export IMMUTABLE_TAG="sha-$(git rev-parse --short=12 HEAD)"

docker tag "$IMAGE" "$REGISTRY_IMAGE:$IMMUTABLE_TAG"
docker push "$REGISTRY_IMAGE:$IMMUTABLE_TAG"

# Update latest only after the immutable image has passed the required gates.
docker tag "$IMAGE" "$REGISTRY_IMAGE:latest"
docker push "$REGISTRY_IMAGE:latest"
```

Consumers should use the immutable `sha-...` tag for reproducible smoke runs.

## 8. Add another model

Add models in the following order so that adapter, local setup, Docker runtime,
and smoke orchestration stay consistent.

### Step 1: verify the official release

Identify the paper authors' official source repository, weight repository,
checkpoint name, license, and load procedure. Pin the upstream repository to a
full Git SHA. Do not remove a safety rail based on an unverified community
upload.

### Step 2: implement and register the adapter

For an image model:

1. Add `benchmark/scripts/depth_models/<model_name>.py`.
2. Return a float32 depth array for an RGB `uint8` image.
3. Declare the correct `native_alignment` (`none` for metric depth or the
   appropriate alignment for relative depth).
4. Expose the underlying Torch module when possible for profiling.
5. Register the builder and canonical name in
   `benchmark/scripts/depth_models/__init__.py`.

For a video model:

1. Add `benchmark/scripts/video_depth_models/<model_name>.py` with a
   `build(device=...)` factory.
2. Accept a whole clip and return depth shaped `(T, H, W)`.
3. Declare whether output is metric or relative and restore output to the input
   spatial and temporal dimensions.
4. Do not silently fall back to per-frame inference unless the row is explicitly
   a frame-as-video baseline.

Add the canonical name to `IMAGE_MODELS` or `VIDEO_MODELS` in
`benchmark/scripts/run_depth_smoke_gate.py`. If weights are not verified, add
the model to `BLOCKED_MODELS` instead of adding a bypass.

### Step 3: define the isolated family

Update `benchmark/scripts/setup_depth_smoke_env.py`:

- `MODEL_FAMILY`: canonical model to environment family;
- `UPSTREAMS`: official repository and full pinned SHA when source is needed;
- `FAMILY_PACKAGES`: family-specific dependencies; and
- `FAMILY_IMPORTS`: imports that exercise the adapter's actual load path.

Run the host setup first. It catches dependency conflicts faster than a full
Docker rebuild:

```bash
python3.11 benchmark/scripts/setup_depth_smoke_env.py \
  --model <canonical-name> \
  --env-root /data/rpx-envs
```

### Step 4: lock the Docker dependencies

Create `requirements-<family>.in` with the intentional direct versions and a
reviewed `requirements-<family>.lock` containing the exact resolved runtime
set. Keep Torch, torchvision, CUDA, and other base-owned packages out of a
family lock unless the family intentionally replaces them. Prefer
`opencv-python-headless` in the container.

Review the lock for accidental upgrades to NumPy, Transformers, Diffusers,
OpenCV, xFormers, or Hugging Face Hub. These are the dependencies most likely to
break an upstream model API or ABI.

### Step 5: add a cumulative Docker stage

Extend the current final stage and, in that new stage:

1. declare the upstream SHA as a build argument;
2. clone/fetch exactly that SHA;
3. create `/opt/rpx-envs/<family>` with `--system-site-packages` when sharing the
   base CUDA/PyTorch stack;
4. install the reviewed lock and the pinned upstream source;
5. add source-path aliases if `MODEL_FAMILY` and the checkout directory differ;
6. write `/opt/rpx-envs/<family>/rpx-environment.json`;
7. ensure `/opt/rpx-envs/<family>/bin/python` exists;
8. run imports through that exact Python executable; and
9. leave the new cumulative stage last, or select it explicitly with
   `--target`.

The readiness marker is mandatory for the matrix. A successful import without
`rpx-environment.json` is only a Docker build check.

If the new family changes benchmark code, `COPY benchmark /opt/rpx/benchmark`
and reinstall the RPX package in the stage so the image does not retain an old
adapter from the published base.

### Step 6: update the runtime defaults

After the model is genuinely matrix-ready, update the appropriate defaults in:

- `docker/depth-smoke/entrypoint.sh`;
- `docker/depth-smoke/compose.yaml`; and
- the inventory and examples in this README.

Keep image and video selections separate while `entrypoint.sh smoke` invokes
the image task only.

### Step 7: test in increasing scope

Run fast offline tests first:

```bash
pytest -q benchmark/tests/test_depth_smoke_tools.py
```

Then run:

1. the new named Docker stage build;
2. its build-time import check;
3. container CUDA `verify` or an equivalent family import check;
4. the one-sample/8-frame micro gate;
5. acceptance only after micro passes; and
6. the full Easy pilot only after acceptance and review.

Do not classify a model as complete from installation, import, CPU inference,
an OOM, a diagnostic quantized configuration, or a micro pass alone.

## 9. Common failures

| Symptom | Classification and action |
| --- | --- |
| `torch.cuda.is_available()` is false | NVIDIA Container Toolkit/driver problem; fix GPU passthrough, never fall back to CPU |
| GPU already has compute processes | Select an idle GPU with `--gpu-index`; do not overlap acceptance runs |
| HTTP 401/403 or gated repository | Request access with the same Hugging Face account; do not substitute checkpoints |
| CUDA OOM | Preserve the command and move it to a higher-memory GPU |
| `No module named ...` | Add the missing dependency/source path to the isolated family and rebuild |
| `has no attribute` or `from_pretrained` failure | Reconcile the adapter with the pinned upstream API/checkpoint |
| Wrong shape, dimension, or dtype | Normalize the adapter output to the image `(H, W)` or video `(T, H, W)` contract and float32 |
| Matrix says `missing_environment` | Confirm both `<family>/bin/python` and `<family>/rpx-environment.json` exist under `/opt/rpx-envs` |
| A changed image reruns passed gates | Expected: matrix state is isolated by exact code identity |

For the host-venv workflow and the full Easy handoff, see
[`benchmark/docs/depth_smoke_runbook.md`](../../benchmark/docs/depth_smoke_runbook.md).

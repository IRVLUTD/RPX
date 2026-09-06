# VQA smoke images

The RPX roster is in `model-matrix.json`. The Gemma entries use Google's
official Gemma 4 instruction checkpoints: E4B and 12B Unified. Build
environments cumulatively by dependency family.

The current bbox-only Gemma 4 stage runs the largest 12B model first. Its
14-row smoke manifest covers General bbox in clutter/interaction/clean/ego and
Spatial bbox in the three currently available MOS phases. In-context and Ego
Spatial are added only when their parquets are published.

The acceptance gate contains 700 questions: one General bbox question for
every scene in all four current conditions (400), plus one Spatial bbox
question for every scene in all three current MOS phases (300).

```bash
export RPX_VQA_FAMILY=gemma4
export RPX_VQA_IMAGE=vndhiran123/rpx-vqa-smoke
bash docker/vqa-smoke/build_and_push.sh

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN \
  -v /data/narendhiran_rpx/hf-cache:/cache/huggingface \
  -v /data/narendhiran_rpx/vqa-cache:/cache/rpx-vqa \
  -v /data/narendhiran_rpx/vqa-results:/outputs \
  "$RPX_VQA_IMAGE:gemma4" smoke gemma4-12b
```

Each image must implement the same JSONL request/response contract documented
in `benchmark/data/vqa_smoke/v1/README.md`. Lockfiles, model revision pins, and
GPU-tested Dockerfiles are added one family at a time; the shared benchmark
contract must not acquire model-specific scoring behavior.

## Current stage: PaliGemma 2

`vqa_paligemma` is the first cumulative stage and serves both the 3B and 10B
mix-448 checkpoints. Validate 3B before running 10B. Model weights are fetched
at runtime into a mounted Hugging Face cache and are never baked into the
image. The checkpoints are pinned to immutable Hub revisions.

PaliGemma is license-gated. Accept the terms for both model repositories in a
browser and pass `HF_TOKEN` only at container runtime.

```bash
export RPX_VQA_IMAGE=vndhiran123/rpx-vqa-smoke
bash docker/vqa-smoke/build_and_push.sh

docker run --rm --gpus '"device=0"' \
  "$RPX_VQA_IMAGE:paligemma" verify

docker run --rm --gpus '"device=0"' --ipc=host --shm-size=8g \
  -e HF_TOKEN \
  -v /home/rpx/.cache/huggingface:/cache/huggingface \
  -v /home/rpx/.cache/rpx-vqa:/cache/rpx-vqa \
  -v /home/rpx/RPX_vqa_outputs:/outputs \
  "$RPX_VQA_IMAGE:paligemma" smoke paligemma2-3b
```

Only after the data-backed smoke succeeds, publish both the immutable SHA tag
and the family convenience tag:

```bash
revision="$(git rev-parse HEAD)"
docker push "$RPX_VQA_IMAGE:paligemma-sha-${revision:0:12}"
docker push "$RPX_VQA_IMAGE:paligemma"
```

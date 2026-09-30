# GenCeption on RPX

This image runs both official GenCeption variants through the RPX image-depth,
video-depth, text-initialized tracking, regular grounding VQA and in-context
grounding VQA adapters. It pins the RPX commit at build time and the official
GenCeption source commit at `a47e55120cc2b00027c19c9f1937831e77541056`.

From a clean checkout of `naren/genception-integration`:

```bash
export RPX_GENCEPTION_IMAGE=rpx-genception
bash docker/genception/build.sh

export RPX_GENCEPTION_RUNTIME="$HOME/rpx-genception-runtime"
mkdir -p "$RPX_GENCEPTION_RUNTIME"/{hf-cache,data-cache,models,outputs,vocab,vqa}

read -rsp "Hugging Face token: " HF_TOKEN; echo
export HF_TOKEN

docker run --rm \
  -v "$RPX_GENCEPTION_RUNTIME/models:/models/genception" \
  "$RPX_GENCEPTION_IMAGE:latest" download all

docker run --rm --gpus 'device=0' --ipc=host --shm-size=32g \
  "$RPX_GENCEPTION_IMAGE:latest" verify
```

Every run should mount the four persistent directories and pass the token:

```bash
genception_run() {
  docker run --rm --gpus "device=${RPX_GPU:-0}" --ipc=host --shm-size=32g \
    -e HF_TOKEN \
    -v "$RPX_GENCEPTION_RUNTIME/hf-cache:/cache/huggingface" \
    -v "$RPX_GENCEPTION_RUNTIME/data-cache:/cache/rpx" \
    -v "$RPX_GENCEPTION_RUNTIME/models:/models/genception" \
    -v "$RPX_GENCEPTION_RUNTIME/outputs:/outputs" \
    "$RPX_GENCEPTION_IMAGE:latest" "$@"
}
```

The commands below are bounded real-weight smoke tests:

```bash
genception_run python scripts/run_depth.py \
  --model genception-1.3b --split easy --device cuda --batch-size 1 \
  --cache-dir /cache/huggingface --max-samples 1 --skip-flops \
  --output-dir /outputs/genception-1.3b/image-depth/easy-smoke

genception_run python scripts/run_video_depth.py \
  --model genception-1.3b --split easy --device cuda \
  --max-samples 1 --frame-budget 8 --sampling stride \
  --output-dir /outputs/genception-1.3b/video-depth/easy-smoke
```

Tracking additionally needs the immutable text vocabulary from the dataset
repository. Download it on the host, then mount it into the run:

```bash
hf download IRVLUTD/RPX \
  tracking/metadata/text_initialization_v1/scene_condition_vocab.parquet \
  --repo-type dataset \
  --revision 91921dcb5d328da3e0750c6b5e3f7109cb8b2bb1 \
  --local-dir "$RPX_GENCEPTION_RUNTIME/vocab"

docker run --rm --gpus 'device=0' --ipc=host --shm-size=32g \
  -e HF_TOKEN \
  -v "$RPX_GENCEPTION_RUNTIME/hf-cache:/cache/huggingface" \
  -v "$RPX_GENCEPTION_RUNTIME/data-cache:/cache/rpx" \
  -v "$RPX_GENCEPTION_RUNTIME/models:/models/genception" \
  -v "$RPX_GENCEPTION_RUNTIME/outputs:/outputs" \
  -v "$RPX_GENCEPTION_RUNTIME/vocab/tracking/metadata/text_initialization_v1/scene_condition_vocab.parquet:/vocab.parquet:ro" \
  "$RPX_GENCEPTION_IMAGE:latest" python scripts/run_tracking.py \
    --model genception-1.3b --split easy --dataset-protocol mos \
    --cache-dir /cache/rpx --output-dir /outputs/genception-1.3b/tracking/mos-easy-smoke \
    --text-vocab /vocab.parquet --max-clips 1 --max-frames 8
```

Use the VQA acceptance set to cover both regular and in-context inputs. The
manifest builder uses five Parquets pinned at dataset revision
`5a1702652bc16fc0a6b2699a78cc85a4582a3b93`:

```bash
genception_run python scripts/fetch_vqa_smoke_parquets.py --out /cache/rpx/vqa/parquets
genception_run python scripts/build_vqa_acceptance_sample.py \
  --parquet-dir /cache/rpx/vqa/parquets --out /cache/rpx/vqa/acceptance.jsonl
genception_run python scripts/run_vqa_benchmark.py \
  --model genception-1.3b --manifest /cache/rpx/vqa/acceptance.jsonl \
  --shard-index 0 --shard-count 1 --batch-size 1 \
  --image-cache /cache/rpx/vqa/images \
  --predictions /outputs/genception-1.3b/vqa/acceptance/predictions.jsonl \
  --failures /outputs/genception-1.3b/vqa/acceptance/failures.jsonl \
  --report /outputs/genception-1.3b/vqa/acceptance/report.json \
  --run-config-out /outputs/genception-1.3b/vqa/acceptance/run_config.json --resume
```

After 1.3B passes, replace `genception-1.3b` with `genception-14b`. Run one
model process per GPU. The official pipeline uses the first visible JAX device
and does not shard 14B across multiple GPUs.

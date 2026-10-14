# VQA image (T5 and T6)

One pinned runtime serves both VQA (one image) and in-context VQA (a reference crop plus a target image). The published
image is `irvlutd/rpx:vqa-vllm`; see the [image table](../README.md#which-image-to-pull).

The paper reports 12 configurations, two sizes each of Cosmos Reason2, DeepSeek-VL2, Florence-2, InternVL 3.5,
PaliGemma 2 and Qwen3-VL:

```text
cosmos-reason2-2b   cosmos-reason2-8b
deepseek-vl2-tiny   deepseek-vl2
florence2-base      florence2-large
internvl3.5-1b      internvl3.5-14b
paligemma2-3b       paligemma2-10b
qwen3-vl-2b         qwen3-vl-8b
```

The image can run more models than these. `docker run --rm irvlutd/rpx:vqa-vllm list-models` prints the full roster
from `model-matrix.json`.

## Protocol

The score is based on the box alone. Each model gets one generation with the original question and every required image,
and returns one box as XYXY on a 0–1000 grid relative to the target image. In-context rows send the reference first
(Image 1) and the target second (Image 2), and the box is always in Image 2's coordinates. Generated labels are kept
for analysis but never affect the score.

The parser rejects missing, multiple, non-finite, reversed and out-of-range boxes. These count as model failures, not
infrastructure errors. Generic JSON models also accept the common 0–1 fractional convention and a redundant singleton
list, and the convention used is recorded per row. No decoder guesses at XYWH boxes, invents labels or replaces a
refusal.

Each prediction records its backend and version, Hub repository and immutable model revision. Chat models run on
vLLM, which is pinned as the `vllm/vllm-openai` 0.28.0 CUDA 12.9 image digest. Every engine is built with
`limit_mm_per_prompt={"image": 2}`.

### Per-model adapters

**Florence-2** (`-ft` checkpoints) and **PaliGemma 2** run through their native Transformers interfaces.

- Florence-2 uses `<CAPTION_TO_PHRASE_GROUNDING>` and PaliGemma 2 uses `detect …`.
- Both take a referring expression that is rewritten deterministically from the question, without the answer or
  ground truth.
- Exactly one region must land in the target panel. Zero or several count as a parse failure, and ground truth is
  never used to pick between candidates.
- Neither has a native multi-image interface. In-context rows therefore use one labelled side-by-side composite, and
  the box is mapped back to Image 2. Row metadata records this (`florence2_composite_phrase_grounding`,
  `labelled_side_by_side_composite`).
- Florence-2 runs in an isolated environment on Transformers 4.49.0. The main environment stays on the newer version
  that Qwen3-VL and vLLM require.

**DeepSeek-VL2** uses `<|ref|>expression<|/ref|>` for single-image rows. In-context rows use `<|grounding|>`, with the
hash-verified reference crop followed by the target. Its native 0–999 boxes are converted to the 0–1000 contract
without changing their geometry. The full model runs on two GPUs (`tensor_parallel_size=2`); every other engine uses
one.

**Cosmos Reason2** uses NVIDIA's example system message (`You are a helpful assistant.`) and its 2D-grounding prompt.
NVIDIA does not define a complete output schema, so RPX makes it explicit: exactly one `bbox_2d` box, the same format
Qwen3-VL uses. Reasoning mode is off because the benchmark scores the localization, not a reasoning trace.

**InternVL 3.5** uses the official dynamic 448-pixel tiling and `model.chat`. It returns its native
`label[[x0,y0,x1,y1]]` box for a GT-free referring-expression rewrite of the question. In-context rows mark the images
`Image-1` and `Image-2`.

- The 1B model runs in BF16 on one GPU.
- The 14B model (15.1B parameters) runs unquantized across two GPUs with `device_map="auto"` and must not share
  them with the 1B engine.

## Question sets

| Set | Questions | Contents |
|---|---:|---|
| Smoke | 14 | Single-image only. Two General questions for each of CLU, INT, CLN and EGO; two Spatial questions for each of CLU, INT and CLN |
| Acceptance | 104 | Seed 20260908. VQA: General 5 types × 4 conditions × 2 = 40, Spatial 3 × 3 × 2 = 18. In-context: General 40, Spatial 1 × 3 × 2 = 6 |
| Benchmark | 31,500 runnable | 27,000 MOS + 4,500 EGO, from `scripts/build_vqa_benchmark_plan.py` |

"Conditions" are the three MOS phases plus EGO. Every acceptance answer box is centred in the middle 50% of the image
on both axes, and no two rows reduce to the same question after alias normalization (`rpx_benchmark/vqa/canonical.py`).

The benchmark plan has 36,000 slots. The other 4,500 are EGO slots that do not exist yet: EGO has no depth map, so it
has no single-image Spatial questions, and in-context Spatial ground truth for EGO has not been generated. These slots
are kept as plain descriptors, never as fabricated rows.

## Build

From a clean, committed tree:

```bash
export RPX_VQA_IMAGE=narendhiranv04/rpx-vqa-smoke
bash docker/vqa-smoke/build_and_push.sh --push
```

This pushes an immutable `vllm-sha-<git-sha>` tag and a moving `vllm` tag. Weights download at run time into the
mounted Hugging Face cache.

## Run the gates

```bash
export RPX_VQA_RUNTIME=/path/to/vqa-runtime
export RPX_VQA_IMAGE=narendhiranv04/rpx-vqa-smoke
read -rsp "HF token: " HF_TOKEN; echo
export HF_TOKEN

docker run --rm --gpus 'device=0' "$RPX_VQA_IMAGE:vllm" verify
docker run --rm "$RPX_VQA_IMAGE:vllm" list-models
```

On datacenter GPUs with an older compatible driver, also set `RPX_VQA_CUDA_COMPAT=1`. This turns on vLLM's bundled CUDA
forward-compatibility libraries.

Start one resident engine per GPU, then run smoke and acceptance against it without reloading the model:

```bash
bash docker/vqa-smoke/start_persistent_engine.sh gemma4-12b 0
bash docker/vqa-smoke/run_persistent_smoke.sh gemma4-12b
bash docker/vqa-smoke/run_persistent_acceptance.sh gemma4-12b

docker ps --filter name=rpx-vqa-gemma4-12b
docker logs --tail 30 rpx-vqa-gemma4-12b
```

The engine stays up after the report is written. Stop it when you need the GPU:

```bash
docker stop rpx-vqa-gemma4-12b
docker rm rpx-vqa-gemma4-12b
```

Reports are written to `$RPX_VQA_RUNTIME/outputs/<model>/sha-<rpx-git-sha>/<gate>/report.json`:

```bash
report="$(find "$RPX_VQA_RUNTIME/outputs/gemma4-12b" -path '*/smoke/report.json' -print | sort | tail -1)"
jq 'del(.samples)' "$report"
jq '.samples[] | {question,raw_output,predicted_bbox,ground_truth_bbox,iou,latency_ms,valid,parse_error}' "$report"
```

The gates check coverage, not accuracy: every question must get a request and a prediction. Parse failures, refusals,
invalid boxes and low accuracy stay in the report as model failures. `--min-parse-rate` adds an optional threshold; no
threshold is applied by default.

Gate latency is per request at batch size 1, after one untimed warm-up, with CUDA synchronized before and after each
question.

To review an acceptance run visually, build a self-contained HTML gallery. It shows the images, question, raw output,
parsed box, ground truth, overlay and IoU for each row:

```bash
python3 scripts/build_vqa_acceptance_gallery.py \
  --manifest "$RPX_VQA_RUNTIME/outputs/gemma4-12b/sha-<sha>/acceptance/manifest.jsonl" \
  --predictions "$RPX_VQA_RUNTIME/outputs/gemma4-12b/sha-<sha>/acceptance/predictions.jsonl" \
  --model gemma4-12b \
  --image-cache "$RPX_VQA_CACHE/images" \
  --out gallery.html
```

## Run the benchmark

Build the plan once, then fill the shared image cache and run each model on one or more shards:

```bash
python3 scripts/build_vqa_benchmark_plan.py \
  --parquet-dir "$RPX_VQA_CACHE/parquets" --out-dir benchmark_plan/

bash docker/vqa-smoke/prefetch_benchmark.sh \
  benchmark_plan/benchmark_available_31500.jsonl

bash docker/vqa-smoke/run_benchmark.sh gemma4-12b 0 \
  benchmark_plan/benchmark_available_31500.jsonl 0 4
```

The last two arguments are `SHARD_INDEX` and `SHARD_COUNT`. Shards are deterministic: rows are sorted by `sample_id`
and assigned round-robin, so rerunning a shard resumes it. Each shard writes `predictions.jsonl`, `failures.jsonl`,
`report.json` and `run_config.json` to
`$RPX_VQA_RUNTIME/outputs/<model>/sha-<rpx-git-sha>/benchmark/shard-<i>-of-<n>/`.

`run_benchmark.sh` batches requests (8 by default) and records per-row time as `amortized_latency_ms`, separate from
the gates' `latency_ms`. On resume it rejects duplicate or out-of-shard IDs. A failing batch is retried row by row, and
a row that still fails goes to `failures.jsonl`, so no row is silently dropped.

To reuse a resident engine:

```bash
bash docker/vqa-smoke/run_persistent_benchmark.sh gemma4-12b \
  benchmark_plan/benchmark_available_31500.jsonl 0 1
```

Resident engines use `max_num_seqs=1` and send requests one at a time. To use several GPUs, run a different model on
each one. Each row gets three attempts before it becomes a `failures.jsonl` entry. `--retry-failures` moves those
entries to `failures.jsonl.history` and requeues them.

To run two replicas of one model, give each engine an instance name. The name only affects container and log names,
and both shards write to the same output directory:

```bash
RPX_VQA_ENGINE_INSTANCE=gpu3 \
  bash docker/vqa-smoke/start_persistent_engine.sh internvl3.5-1b 3
RPX_VQA_ENGINE_INSTANCE=gpu4 \
  bash docker/vqa-smoke/start_persistent_engine.sh internvl3.5-1b 4

RPX_VQA_ENGINE_INSTANCE=gpu3 \
  bash docker/vqa-smoke/run_persistent_benchmark.sh \
    internvl3.5-1b benchmark_plan/benchmark_available_31500.jsonl 0 2 &
RPX_VQA_ENGINE_INSTANCE=gpu4 \
  bash docker/vqa-smoke/run_persistent_benchmark.sh \
    internvl3.5-1b benchmark_plan/benchmark_available_31500.jsonl 1 2 &
wait
```

`run_dual_gpu_persistent_benchmark.sh` does the same with GPU and manifest preflight checks, and exits nonzero if
either shard fails:

```bash
bash docker/vqa-smoke/run_dual_gpu_persistent_benchmark.sh \
  internvl3.5-1b benchmark_plan/benchmark_available_31500.jsonl 3 4
```

## Metrics

Reports include:

- box validity;
- Acc@0.25/0.50/0.75 (IoU ≥ threshold);
- mean IoU;
- `bbox_mean_giou`;
- center-in-GT accuracy;
- `bbox_mean_accuracy_50_95`.

`bbox_mean_giou` averages over every row and gives a malformed, missing or out-of-image box the worst value, −1.0. Use
it for comparisons. `bbox_mean_giou_valid`, over valid boxes only, is a diagnostic. Center-in-GT tests the predicted
center against the inclusive GT box, and invalid boxes count as misses.

`bbox_mean_accuracy_50_95` averages single-box success over IoU 0.50:0.05:0.95. It is deliberately not called mAP:
each question gets one box with no confidence score, so ranked detection AP is undefined. Label exact match, token F1 and joint label+IoU
are informational only.

To rescore finished runs after a metric change without rerunning inference, run the extractor. For each model it finds
the newest complete shard set and checks manifest identity when `run_config.json` is present. It counts terminal
failures as invalid predictions and writes JSON, CSV and Markdown:

```bash
PYTHONPATH=benchmark python3 benchmark/scripts/extract_vqa_benchmark_metrics.py \
  --manifest /path/to/benchmark_runnable_verified.jsonl \
  --outputs-root "$RPX_VQA_RUNTIME/outputs" \
  --model florence2-base \
  --model florence2-large \
  --out-dir "$RPX_VQA_RUNTIME/metrics/server-name"
```

## Failure diagnostic

After acceptance, a diagnostic separates choosing the wrong object from localizing the right one badly. It asks each
resident model three things:

1. the answer phrase alone;
2. a box for that phrase;
3. a box for the ground-truth label.

Passing the acceptance report reuses its end-to-end outputs:

```bash
bash docker/vqa-smoke/run_persistent_diagnostic.sh MODEL \
  --acceptance-report /outputs/MODEL/sha-ACCEPTANCE_SHA/acceptance/report.json
```

`diagnostic_report.json` judges object choice by whether the model's own answer phrase grounds to the GT box. Literal
label match is not used, so aliases don't count as reasoning errors. The diagnostic also tries alternative coordinate
conventions, using GT to spot convention mistakes. Florence-2 adds its native `<VQA>` call and PaliGemma 2 adds
`answer en`.

These calls are diagnostics, and some of them use GT. They are never merged into scored results, and oracle or
best-convention numbers must not be reported as benchmark performance.

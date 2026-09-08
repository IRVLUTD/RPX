# RPX VQA: vLLM-only smoke, acceptance and benchmark gates

One pinned runtime image serves the complete ten-model VQA roster, covering
both one-image ("normal") and two-image ("in-context") tasks. There is no
Transformers inference fallback. Each prediction records the backend, vLLM
version, Hub repository, and immutable model revision; the gate rejects
results whose backend is not `vllm`.

The current smoke manifest has 14 bbox questions over four RGB frames
(normal tasks only):

- two General bbox questions for each of CLU, INT, CLN, and EGO;
- two Spatial bbox questions for each of CLU, INT, and CLN.

The current acceptance manifest has 104 questions, deterministically seeded
(20260908), covering normal AND in-context tasks:

- normal general (5 types x 4 conditions x 2 = 40) and normal spatial
  (3 types x 3 MOS phases x 2 = 18);
- in-context general (5 types x 4 conditions x 2 = 40) and in-context
  spatial (1 type x 3 MOS phases x 2 = 6).

"conditions" = MOS phase 0/1/2 and EGO. Every selected row's answer bbox
centroid lies in the middle 50% of the image in both axes; no two selected
rows collapse to the same question after function-alias normalization (see
`rpx_benchmark/vqa/canonical.py`).

The full benchmark plan (`scripts/build_vqa_benchmark_plan.py`) covers
36,000 planned questions: 31,500 currently runnable (27,000 MOS + 4,500 EGO)
and 4,500 reserved-but-not-yet-generated EGO slots (EGO has no depth map, so
normal spatial does not exist for it at all; EGO in-context spatial ground
truth has not been generated yet). The pending slots are plain descriptors,
never fabricated inference rows.

## Runtime contract

- Base runtime: pinned `vllm/vllm-openai` 0.28.0 CUDA 12.9 image digest.
- Exactly one visible GPU per model engine (`tensor_parallel_size=1`).
- Every engine is built with `limit_mm_per_prompt={"image": 2}`: normal rows
  still send exactly one image; in-context rows send `[reference, target]`
  in that order (Image 1 then Image 2). The answer bbox is always in Image
  2's own coordinates.
- One unmeasured warm-up question precedes timed inference.
- Timings synchronize CUDA immediately before and after each question.
- PaliGemma 2 answers the question first (both images, composited side by
  side for in-context rows -- see the runbook's "PaliGemma two-stage
  behavior"), then grounds its own predicted label in Image 2 ALONE, never
  Image 1; the recorded latency includes both vLLM calls and never uses the
  ground truth.
- The smoke/acceptance gate (`run_vllm_vqa.py`) measures isolated,
  batch-size-1 per-request latency. The benchmark runner
  (`run_vqa_benchmark.py`) supports configurable batching (default 8) for
  throughput; its per-row timing is always recorded as
  `amortized_latency_ms`, never as the isolated-request `latency_ms` field
  the gates use.
- Interrupted runs resume from validated JSONL rows; the benchmark runner
  additionally rejects duplicate or out-of-shard sample IDs on resume and
  never drops a row silently -- a failing batch is retried row-by-row, and
  any row that still fails lands in `failures.jsonl` instead of being lost.
- Reports retain every question, raw output, parsed bbox, ground-truth bbox,
  IoU, parse error, and latency, plus aggregate parse rate, mean IoU, Acc@0.5,
  and latency statistics.
- Every JSON-capable model receives the same bbox instruction and emits XYXY
  coordinates normalized to 0--1000; the parser performs the one documented
  conversion back to target-image pixels. It does not repair invalid JSON,
  reversed boxes, refusals, or missing boxes.
- PaliGemma necessarily uses its published `answer en` then `detect` interface.
  Its stage-1 label and stage-2 native-location output are both retained in
  `adapter_metadata`; an empty stage-1 answer remains an invalid model result.

## Build once

Run inside the repository, from a clean committed tree:

```bash
export RPX_VQA_IMAGE=vndhiran123/rpx-vqa-smoke
bash docker/vqa-smoke/build_and_push.sh --push
```

This creates immutable `vllm-sha-<git-sha>` and convenience `vllm` tags. Model
weights are downloaded at runtime into the mounted Hugging Face cache, not
baked into the image.

## Run gates

Set the writable server runtime once:

```bash
export RPX_VQA_RUNTIME=/data/narendhiran_rpx/src/vqa-runtime
export RPX_VQA_IMAGE=vndhiran123/rpx-vqa-smoke
read -rsp "HF token: " HF_TOKEN; echo
export HF_TOKEN
```

On professional/datacenter GPUs with an older compatible NVIDIA driver, also
set `RPX_VQA_CUDA_COMPAT=1`. The gate wrapper then enables vLLM's bundled CUDA
forward-compatibility libraries and bypasses the base image's version guard.

Verify the common image and print the roster:

```bash
docker run --rm --gpus 'device=0' "$RPX_VQA_IMAGE:vllm" verify
docker run --rm "$RPX_VQA_IMAGE:vllm" list-models
```

Run smoke before acceptance:

```bash
bash docker/vqa-smoke/run_gate.sh smoke gemma4-12b 0
bash docker/vqa-smoke/run_gate.sh acceptance gemma4-12b 0
```

Build a visual audit gallery for an acceptance run (Image 1 + Image 2 for
in-context rows, target only for normal rows, question, raw output, parsed
prediction, ground truth, prediction/GT overlay, IoU/validity/parse-error/
latency, all embedded in one portable HTML file):

```bash
python3 scripts/build_vqa_acceptance_gallery.py \
  --manifest "$RPX_VQA_RUNTIME/outputs/gemma4-12b/sha-<sha>/acceptance/manifest.jsonl" \
  --predictions "$RPX_VQA_RUNTIME/outputs/gemma4-12b/sha-<sha>/acceptance/predictions.jsonl" \
  --model gemma4-12b \
  --image-cache "$RPX_VQA_CACHE/images" \
  --out gallery.html
```

Build the full benchmark plan once (deterministic, seed 20260908 -- see
`scripts/build_vqa_benchmark_plan.py`'s docstring for the exact MOS/EGO
selection rules), then run each model against one or more shards of
`benchmark_available_31500.jsonl`:

```bash
python3 scripts/build_vqa_benchmark_plan.py \
  --parquet-dir "$RPX_VQA_CACHE/parquets" --out-dir benchmark_plan/

bash docker/vqa-smoke/run_benchmark.sh gemma4-12b 0 \
  benchmark_plan/benchmark_available_31500.jsonl 0 4
```

`SHARD_INDEX`/`SHARD_COUNT` split the manifest deterministically (sorted by
`sample_id`, round-robin) so multiple GPUs can run disjoint shards in
parallel; rerunning the same shard resumes safely. Per-shard outputs
(`predictions.jsonl`, `failures.jsonl`, `report.json`, `run_config.json`)
land under
`$RPX_VQA_RUNTIME/outputs/<model>/sha-<rpx-git-sha>/benchmark/shard-<i>-of-<n>/`.

The frozen keys, in high-latency-first execution order, are:

```text
gemma4-12b
paligemma2-10b
internvl2.5-8b
idefics3-8b
qwen2.5-vl-7b
llava-onevision-7b
gemma4-e4b
phi-3.5-vision-4b
paligemma2-3b
qwen2.5-vl-3b
```

Reports are under:

```text
$RPX_VQA_RUNTIME/outputs/<model>/sha-<rpx-git-sha>/<gate>/report.json
```

Inspect the aggregate metrics and per-sample predictions with `jq`:

```bash
report="$(find "$RPX_VQA_RUNTIME/outputs/gemma4-12b" -path '*/smoke/report.json' -print | sort | tail -1)"
jq 'del(.samples)' "$report"
jq '.samples[] | {question,raw_output,predicted_bbox,ground_truth_bbox,iou,latency_ms,valid,parse_error}' "$report"
```

The smoke/acceptance gate is operational: it requires complete request and
prediction coverage. Parse failures, refusals, invalid boxes, and low accuracy
remain in the report and score as model failures; they do not masquerade as an
infrastructure crash. An optional `--min-parse-rate` diagnostic can still be
requested explicitly, but no capability threshold is imposed by default.

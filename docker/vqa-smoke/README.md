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
- Every bbox model answers and localizes in one scored call containing
  the original question and all required images. Two-image requests explicitly
  label Image 1 as the reference and Image 2 as the target.
- PaliGemma 2 retains its official `answer en` task prefix and uses a visibly
  labelled side-by-side composite for two-image rows, but still receives only
  one scored generation call. Its native `detect <label>` prefix is reserved
  for the explicitly unscored localization diagnostic.
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
- Every JSON-capable model receives the identical direct semantic instruction
  and is asked for XYXY coordinates normalized to 0--1000. The
  model-independent decoder also recognizes the common fractional 0--1
  convention and one redundant singleton bbox list; the selected coordinate
  convention is recorded per row. It does not guess reversed XYXY/XYWH boxes,
  invent labels, or replace refusals/missing boxes.
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

To run acceptance without unloading the model afterward, start one resident
engine per GPU and execute acceptance inside it:

```bash
bash docker/vqa-smoke/start_persistent_engine.sh gemma4-12b 0
bash docker/vqa-smoke/run_persistent_acceptance.sh gemma4-12b

# The model is still resident after the report is written.
docker ps --filter name=rpx-vqa-gemma4-12b
docker logs --tail 30 rpx-vqa-gemma4-12b
```

Stop it explicitly only when that GPU is needed for another model:

```bash
docker stop rpx-vqa-gemma4-12b
docker rm rpx-vqa-gemma4-12b
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

# Populate the shared cache once before launching several model clients.
bash docker/vqa-smoke/prefetch_benchmark.sh \
  benchmark_plan/benchmark_available_31500.jsonl

bash docker/vqa-smoke/run_benchmark.sh gemma4-12b 0 \
  benchmark_plan/benchmark_available_31500.jsonl 0 4
```

If the model already runs under `start_persistent_engine.sh`, reuse it without
reloading or stopping it:

```bash
bash docker/vqa-smoke/run_persistent_benchmark.sh gemma4-12b \
  benchmark_plan/benchmark_available_31500.jsonl 0 1
```

The resident path deliberately uses sequential requests because resident
engines are configured with `max_num_seqs=1`; run one different model per GPU
to obtain four-model concurrency. Each row is retried up to three times before
becoming a terminal `failures.jsonl` entry. A later invocation with
`--retry-failures` archives those entries to `failures.jsonl.history` and
requeues them while retaining every successful row.

`SHARD_INDEX`/`SHARD_COUNT` split the manifest deterministically (sorted by
`sample_id`, round-robin) so multiple GPUs can run disjoint shards in
parallel; rerunning the same shard resumes safely. Per-shard outputs
(`predictions.jsonl`, `failures.jsonl`, `report.json`, `run_config.json`)
land under
`$RPX_VQA_RUNTIME/outputs/<model>/sha-<rpx-git-sha>/benchmark/shard-<i>-of-<n>/`.

Spatial reports include mean IoU, Acc@0.25/0.50/0.75, and
`bbox_mean_accuracy_50_95` (mean single-box success across IoU thresholds
0.50:0.05:0.95). The latter is intentionally **not called mAP**: this protocol
emits one box with no confidence score, so conventional ranked detection AP is
undefined. Text reports include normalized exact match, token F1, and the
joint exact-label + IoU@0.5 rate. Exact match is a one-reference RPX label
metric, not conventional multi-annotator VQA soft accuracy.

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

## Decomposed failure diagnostic

After acceptance, separate semantic target-selection errors from localization
errors without treating these probes as benchmark results.  The diagnostic
asks each resident model (1) for the answer phrase only, (2) to localize that
same predicted phrase, and (3) to localize the disclosed ground-truth label in
the target image. Pass the completed acceptance report to reuse its end-to-end
outputs rather than generating them again:

```bash
bash docker/vqa-smoke/run_persistent_diagnostic.sh MODEL \
  --acceptance-report /outputs/MODEL/sha-ACCEPTANCE_SHA/acceptance/report.json
```

The resulting `diagnostic_report.json` grounds the model's own answer phrase and
the oracle label separately.  Identity selection is judged by whether the
answer phrase grounds to the GT object bbox, not by literal label equality, so
aliases and harmless modifiers do not become false reasoning errors.  Literal
exact match remains informational only.  The diagnostic also records strict
protocol parsing plus JSON/Python-dict and normalized/pixel bbox hypotheses;
the best-coordinate measurements use GT to identify convention mistakes and
are diagnostic leakage.  Oracle and best-coordinate results must never be
reported as benchmark performance.

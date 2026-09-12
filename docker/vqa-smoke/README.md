# RPX VQA: smoke, acceptance and benchmark gates

One pinned runtime image serves the frozen VQA roster, covering both one-image
("normal") and two-image ("in-context") tasks. Each prediction records its
explicit backend and version, Hub repository, and immutable model revision.
Chat-capable models use vLLM. Florence-2, PaliGemma 2, and InternVL 3.5 use
their native Transformers task interfaces. The scored contract is bbox-only: one generation
receives the original question and every required image and returns the region.
Generated text labels are retained for analysis but never affect the bbox score.

## Florence-2 native adapter

`florence2-base` and `florence2-large` use the pinned Microsoft `-ft`
checkpoints. The one-stage scored path invokes native
`<CAPTION_TO_PHRASE_GROUNDING>` with a deterministic referring-expression
rewrite of the original question. The rewrite uses no answer or GT data. It
accepts a result only when exactly one native region lies in the
target panel. Zero or multiple target regions remain parse failures; the
evaluator never uses ground truth to select a candidate. For in-context rows,
the labelled Image 1/Image 2 composite is used in that same call and composite
coordinates are mapped back to Image 2 before scoring.

`diagnostic-remote` additionally runs minimal native `<VQA>` answer generation,
answer-phrase localization, and GT-label oracle localization. Those extra calls
explain whether failures come from question interpretation or native grounding;
they are explicitly unscored and never replace the one-stage prediction.

The container keeps Transformers 4.49.0 in an isolated Florence environment;
the main environment stays on the newer version required by Qwen3-VL/vLLM.

In-context Florence inference is an architecture accommodation rather than a
native multi-image interface. Its per-row metadata records
`florence2_composite_phrase_grounding`, the reference-then-target order, and
`labelled_side_by_side_composite`; normal rows remain native single-image
phrase grounding.

## PaliGemma 2 native adapter

PaliGemma 2 runs through its official Hugging Face processor rather than
vLLM. Its one-stage scored path uses `detect <referring expression derived from
the original question>` and decodes native `<loc>` tokens. Exactly one region
in the target panel is required.
For two-image rows, one labelled composite is passed to one generation and the
native composite-relative coordinates are mapped back to Image 2. The
`answer en` and `detect <known label>` calls exist only in the diagnostic
commands; semantic answers and oracle results are not benchmark predictions.

## DeepSeek-VL2 native adapter

Normal single-image rows use DeepSeek-VL2's documented
`<|ref|>expression<|/ref|>` localization form. In-context rows use its native
`<|grounding|>` mode with the hash-verified Image 1 reference crop followed by
the Image 2 target scene. Both paths make one scored model call and accept
exactly one native `<|det|>` bbox. DeepSeek's discrete 0--999 native grid is
explicitly converted to RPX's 0--1000 JSON contract before common parsing.

## Cosmos Reason2 grounding adapter

`cosmos-reason2-2b` and `cosmos-reason2-8b` run as single-GPU BF16 vLLM
engines from immutable Hugging Face revisions. They use NVIDIA's exact example
system message (`You are a helpful assistant.`), preserve Image 1/Image 2 media
order, and make one deterministic scored call. NVIDIA's official 2D-grounding
prompt requires a bounding box returned as JSON but does not define a complete
machine-readable schema. RPX therefore makes the Qwen3-VL-compatible contract
explicit: exactly one `bbox_2d` XYXY box on a target-relative 0--1000 grid.
Missing, multiple, non-finite, reversed, and out-of-range boxes remain parse
failures. RPX omits Reason2's optional chain-of-thought instruction because the
benchmark scores the direct localization result rather than a reasoning trace.

## InternVL 3.5 native grounding adapter

`internvl3.5-1b` and `internvl3.5-14b` use the official dynamic 448-pixel
tiling and `model.chat` path. A deterministic, GT-free rewrite converts each
question into a referring expression, and the model returns its trained native
`label[[x0,y0,x1,y1]]` box on the 0--1000 grid. Exactly one ordered box is
accepted. Two-image rows receive the hash-verified reference crop followed by
the target scene with explicit `Image-1` and `Image-2` markers.

The 1B checkpoint runs in BF16 on one GPU. The 15.1B-parameter 14B checkpoint
runs unquantized across both Server 3 GPUs using the checkpoint's documented
`device_map="auto"` path; it must not run concurrently with the 1B engine.

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
- Exactly one visible GPU for single-GPU model engines. DeepSeek-VL2 full is
  the explicit exception: its resident engine uses two visible GPUs with
  `tensor_parallel_size=2`; remote gate/benchmark clients reuse that engine.
- Every engine is built with `limit_mm_per_prompt={"image": 2}`: normal rows
  still send exactly one image; in-context rows send `[reference, target]`
  in that order (Image 1 then Image 2). The answer bbox is always in Image
  2's own coordinates.
- One unmeasured warm-up question precedes timed inference.
- Timings synchronize CUDA immediately before and after each question.
- Every scored bbox model answers and localizes in one scored call containing
  the original question and all required images. Two-image requests explicitly
  label Image 1 as the reference and Image 2 as the target.
- Florence-2 uses native `<CAPTION_TO_PHRASE_GROUNDING><referring expression>`;
  PaliGemma 2 uses native `detect <referring expression>`. The deterministic
  rewrite uses the question/type only, with no answer or GT. Their two-image scored
  paths use a visibly labelled composite and remap the selected target-panel
  region into Image 2 coordinates.
- DeepSeek-VL2 uses `<|ref|>` for normal referring-expression grounding and
  `<|grounding|>` for two-image in-context grounding. Native 0--999 boxes are
  converted to the common 0--1000 contract without changing their geometry.
- Cosmos Reason2 uses the official system message plus a strict `bbox_2d` JSON
  contract. The model's optional reasoning mode is disabled for deterministic
  one-call localization, and every serialization decision is recorded in the
  row metadata.
- PaliGemma 2 `answer en`, Florence `<VQA>`, predicted-label grounding, and
  GT-label oracle grounding are explicitly unscored diagnostic calls.
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
- Generic JSON-capable models receive the same direct semantic instruction and
  are asked for XYXY coordinates normalized to 0--1000. Cosmos Reason2 uses its
  explicit `bbox_2d` adapter described above. The generic decoder also recognizes
  the common fractional 0--1 convention and one redundant singleton bbox list;
  the selected coordinate convention is recorded per row. No decoder guesses
  reversed XYXY/XYWH boxes, invents labels, or replaces refusals/missing boxes.
- Diagnostic semantic and oracle outputs are never merged into scored results.

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

To run smoke and acceptance without unloading the model between gates, start
one resident engine per GPU and execute both gates inside it:

```bash
bash docker/vqa-smoke/start_persistent_engine.sh gemma4-12b 0
bash docker/vqa-smoke/run_persistent_smoke.sh gemma4-12b
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

Spatial reports include bbox validity, Acc@0.25/0.50/0.75, mean IoU,
all-row mean generalized IoU, center-in-ground-truth accuracy, and
`bbox_mean_accuracy_50_95` (mean single-box success across IoU thresholds
0.50:0.05:0.95). The latter is intentionally **not called mAP**: this protocol
emits one box with no confidence score, so conventional ranked detection AP is
undefined. Generated-label exact match, token F1, and joint label+IoU fields
are informational diagnostics only and do not enter the bbox result.

Completed full runs can be rescored after metric-code changes without rerunning
inference. The extractor discovers the newest complete shard set per requested
model, verifies manifest identity when `run_config.json` is present, counts
terminal inference failures as invalid predictions, and writes JSON, CSV, and
Markdown reports:

```bash
PYTHONPATH=benchmark python3 benchmark/scripts/extract_vqa_benchmark_metrics.py \
  --manifest /path/to/benchmark_runnable_verified.jsonl \
  --outputs-root "$RPX_VQA_RUNTIME/outputs" \
  --model florence2-base \
  --model florence2-large \
  --out-dir "$RPX_VQA_RUNTIME/metrics/server-name"
```

`bbox_mean_giou` is the comparison-safe all-row value: malformed, missing, or
out-of-image boxes receive the worst possible GIoU, -1.0. The JSON also retains
`bbox_mean_giou_valid`, which is a diagnostic over valid boxes only. Acc@0.25
and Acc@0.50 use IoU greater than or equal to the named threshold. Center-in-GT
tests the predicted-box center against the inclusive ground-truth box; invalid
boxes count as misses.

The frozen keys, in high-latency-first execution order, are:

```text
gemma4-12b
paligemma2-10b
internvl3.5-14b
internvl2.5-8b
idefics3-8b
qwen2.5-vl-7b
qwen3-vl-8b
llava-onevision-7b
gemma4-e4b
phi-3.5-vision-4b
paligemma2-3b
qwen2.5-vl-3b
qwen3-vl-2b
internvl3.5-1b
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

# RPX VQA: vLLM-only smoke and acceptance gates

One pinned runtime image serves the complete ten-model, single-image VQA
roster. There is no Transformers inference fallback. Each prediction records
the backend, vLLM version, Hub repository, and immutable model revision; the
gate rejects results whose backend is not `vllm`.

The current smoke manifest has 14 bbox questions over four RGB frames:

- two General bbox questions for each of CLU, INT, CLN, and EGO;
- two Spatial bbox questions for each of CLU, INT, and CLN.

The current acceptance manifest has 700 questions over exactly 400 RGB frames:

- paired General + Spatial questions on 300 MOS frames (100 scenes x 3 phases);
- one General question on 100 EGO frames.

EGO Spatial and the two in-context types remain excluded until their source
parquets are published. Both samplers require the answer bbox centroid to lie
in the middle 50% of the image in both axes.

## Runtime contract

- Base runtime: pinned `vllm/vllm-openai` 0.28.0 CUDA 12.9 image digest.
- Exactly one visible GPU per model engine (`tensor_parallel_size=1`).
- One unmeasured warm-up question precedes timed inference.
- Timings synchronize CUDA immediately before and after each question.
- PaliGemma 2 answers the question, then grounds its own predicted label; the
  recorded latency includes both vLLM calls and never uses the ground truth.
- Interrupted runs resume from validated JSONL rows.
- Reports retain every question, raw output, parsed bbox, ground-truth bbox,
  IoU, parse error, and latency, plus aggregate parse rate, mean IoU, Acc@0.5,
  and latency statistics.

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

Run each smoke first; run acceptance only if its smoke passes:

```bash
bash docker/vqa-smoke/run_gate.sh smoke gemma4-12b 0
bash docker/vqa-smoke/run_gate.sh acceptance gemma4-12b 0
```

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

The smoke/acceptance gate is mechanical: complete coverage and at least 95%
parseable bbox outputs. Accuracy is reported but no paper-score threshold is
invented before baseline validation.

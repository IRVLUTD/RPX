# RPX VQA smoke fixture v1

This is a deterministic **single-image, bbox-only** mechanical smoke sample,
not a paper score subset. It is generated from the benchmark VQA parquets with
`scripts/build_vqa_smoke_sample.py` and contains 14 questions over four real
RPX RGB frames: two General bbox questions in each of CLU, INT, CLN, and EGO,
plus two Spatial bbox questions in each of CLU, INT, and CLN.

Every answer object has its bbox centroid in the middle 50% of both image axes.
Spatial rows with exact or near ties in their serialized evidence are excluded.
Every row has a stable ID, explicit inclusive-pixel `xyxy` convention, original
image dimensions, and a portable Hub tar-member locator. RGB data is not
duplicated in git.

Rebuild from the raw parquets:

```bash
cd benchmark
PYTHONPATH=. python scripts/build_vqa_smoke_sample.py \
  --parquet-dir /path/to/vqa_parquet_30frame_v2 \
  --out data/vqa_smoke/v1/manifest.jsonl
```

Fetch only the required tar members and create model-neutral requests:

```bash
PYTHONPATH=. python scripts/prepare_vqa_smoke.py \
  --manifest data/vqa_smoke/v1/manifest.jsonl \
  --model paligemma2-3b \
  --image-cache .rpx_cache/vqa-smoke-v1/images \
  --out .rpx_cache/vqa-smoke-v1/paligemma2-3b.requests.jsonl
```

The shared container runs every roster checkpoint through vLLM and writes the
raw output, latency, model key, backend, vLLM version, checkpoint, and immutable
revision for each sample. Score that result without any model-specific
normalization hidden in the evaluator:

```bash
PYTHONPATH=. python scripts/run_vqa_smoke_gate.py \
  --manifest data/vqa_smoke/v1/manifest.jsonl \
  --model paligemma2-3b \
  --predictions predictions.jsonl \
  --report report.json
```

The smoke gate verifies complete one-to-one prediction coverage and a 95%
parse rate. Accuracy thresholds belong to the later acceptance/gate stages;
they must be frozen from validated baseline runs, not invented before the
models run.

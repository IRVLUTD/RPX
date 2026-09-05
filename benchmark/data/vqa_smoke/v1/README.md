# RPX VQA smoke fixture v1

This is a deterministic **single-image** mechanical smoke sample, not a paper
score subset. It is generated from the three `vqa_parquet_30frame_v2` files
with `scripts/build_vqa_smoke_sample.py` and contains 36 questions over five
real RPX RGB frames:

- 4 yes + 4 no for each binary relation (`spatial_lr_binary`,
  `spatial_ud_binary`)
- 4 each for `spatial_lr_extreme`, `depth_closest`, and `spatial_farthest`
- 4 MOS + 4 ego questions for `attr_composition`

All three MOS phases occur. Bbox rows with exact or near ties in the serialized
evidence are excluded. Every row has a stable ID, explicit inclusive-pixel
`xyxy` convention, original image dimensions, and a portable Hub tar-member
locator. RGB data is not duplicated in git.

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

Each container reads a request row and writes
`{"sample_id":"...","raw_output":"..."}`. Score that result without any
model-specific normalization hidden in the evaluator:

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

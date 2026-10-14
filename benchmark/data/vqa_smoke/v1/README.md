# VQA smoke fixture v1

A small, deterministic set of single-image bounding-box questions for checking that a VQA model runs
end to end and produces parseable output. It is not a scoring subset, and its accuracy numbers mean
nothing.

## Contents

`scripts/build_vqa_smoke_sample.py` draws the questions from the benchmark VQA Parquets:

| Question type | Phases | Per phase |
|:--|:--|:--|
| General | Clutter, Interaction, Clean, Ego | 2 |
| Spatial | Clutter, Interaction, Clean | 2 |

It keeps only answers whose box centre lies in the middle half of the image on both axes, and drops
Spatial questions whose evidence has exact or near ties. Each row has a stable ID, a box in
inclusive-pixel `xyxy`, the original image size and a Hub tar-member locator, so no images are stored
in Git.

## Usage

Rebuild the manifest from the Parquets:

```bash
cd benchmark
PYTHONPATH=. python scripts/build_vqa_smoke_sample.py \
  --parquet-dir /path/to/vqa_parquet_30frame_v2 \
  --out data/vqa_smoke/v1/manifest.jsonl
```

Fetch the images the manifest needs and write model-ready requests:

```bash
PYTHONPATH=. python scripts/prepare_vqa_smoke.py \
  --manifest data/vqa_smoke/v1/manifest.jsonl \
  --model paligemma2-3b \
  --image-cache .rpx_cache/vqa-smoke-v1/images \
  --out .rpx_cache/vqa-smoke-v1/paligemma2-3b.requests.jsonl
```

The shared vLLM container runs each model on those requests and records, per sample, the raw output,
latency, model key, backend, vLLM version, checkpoint and its pinned revision. Score the result:

```bash
PYTHONPATH=. python scripts/run_vqa_smoke_gate.py \
  --manifest data/vqa_smoke/v1/manifest.jsonl \
  --model paligemma2-3b \
  --predictions predictions.jsonl \
  --report report.json
```

## Gate rules

The evaluator applies no model-specific clean-up. The gate fails if any sample is missing, duplicated
or extra, or if the provenance changes partway through a run. Parse failures count against the model
rather than failing the gate; `--min-parse-rate` adds a threshold when you want one. Accuracy
thresholds belong to the later acceptance stage and should be set from validated baseline runs.

---

<sub>Part of <a href="../../../../README.md">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset</a></sub>

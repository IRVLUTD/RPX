# VQA benchmark subset v1

Question lists that select rows from the published `vqa/` Parquets into a fixed acceptance sample and
a fixed benchmark plan for tasks T5 (VQA) and T6 (In-context VQA). They reference rows by
`sample_id` and add, remove or change no ground truth. There are no images or Parquets here.

<p align="center">
  <a href="#files">Files</a> ·
  <a href="#what-the-subset-contains">What the subset contains</a> ·
  <a href="#loading-with-the-toolkit">Loading with the toolkit</a> ·
  <a href="#provenance">Provenance</a>
</p>

## Files

| File | Rows | Purpose |
|:--|--:|:--|
| `acceptance_104.jsonl` | 104 | small, fixed acceptance sample |
| `benchmark_available_31500.jsonl` | 31,500 | the runnable benchmark |
| `benchmark_pending_4500.jsonl` | 4,500 | reserved Ego slots, not runnable |
| `manifest_summary.json` | | row counts, per-type breakdowns, hashes |
| `SHA256SUMS` | | checksums for this folder |

Rows in the first two files are complete VQA samples matching `rpx_benchmark.vqa.contract.VQASample`.
Pending rows are plain descriptors (`scene_id`, `frame`, `slot`, `reserved_types`, `reason`) with no
question or answer.

## What the subset contains

Every row asks for a box. Binary spatial questions (`spatial_lr_binary`, `spatial_ud_binary`) are
left out. Boxes are `[xmin, ymin, xmax, ymax]`, inclusive, in the target image's original pixels; for
in-context rows the target is Image 2 and Image 1 is a crop from a single-object scan.

The acceptance sample has two rows for every question type in each condition it applies to (the
three MOS phases plus Ego): 40 single-image attribute, 18 single-image spatial, 40 in-context
attribute and 6 in-context spatial rows. Each answer box is centered in the middle half of the image, and no two rows
collapse to the same question after function-alias normalization.

The benchmark plan has two parts:

- MOS: 300 scene–phase cells × 30 frames × 3 questions = 27,000. Each frame gets one single-image
  attribute question, one single-image spatial question, and one in-context question (attribute or
  spatial, split evenly).
- Ego: 100 scenes × 30 single-image attribute questions, plus 15 in-context attribute questions on
  half of those frames = 4,500.

The full plan is 36,000 questions. The remaining 4,500 Ego slots wait on spatial ground truth that Ego
does not have yet, since Ego has no depth map: 3,000 single-image spatial slots and 1,500 in-context
spatial slots. They are reserved, not filled with made-up rows.

## Loading with the toolkit

These lists point their images at earlier dataset revisions, which the toolkit's VQA loader rejects,
and 30 in-context rows in `acceptance_104.jsonl` use phase paths that do not exist (such as `2.0`
instead of `2`). Until they are republished
([IRVLUTD/RPX#96](https://github.com/IRVLUTD/RPX/issues/96)), use the 74 verified rows that ship
with the toolkit:

```python
import rpx_benchmark as rpx

manifest = rpx.download("T5")   # or "T6"
```

## Provenance

All three lists use seed `20260908` and were built from the `vqa/` Parquets at dataset commit
`5a1702652bc16fc0a6b2699a78cc85a4582a3b93` by
[`build_vqa_acceptance_sample.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/scripts/build_vqa_acceptance_sample.py)
and
[`build_vqa_benchmark_plan.py`](https://github.com/IRVLUTD/RPX/blob/main/benchmark/scripts/build_vqa_benchmark_plan.py).

---

<sub>Part of <a href="https://github.com/IRVLUTD/RPX">RPX</a> ·
<a href="https://irvlutd.github.io/RPX/toolkit-docs/">Toolkit docs</a> ·
<a href="https://huggingface.co/datasets/IRVLUTD/RPX">Dataset card</a></sub>

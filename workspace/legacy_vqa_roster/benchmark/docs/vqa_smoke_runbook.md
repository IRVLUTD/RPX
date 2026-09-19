# VQA smoke-test roster and contracts

This document records the roster decision made from the VQA handoff and its two
attached model tables. The machine-readable sources of truth are
`rpx_benchmark.adapters.vqa_scaffold.VQA_MODEL_CARDS` and
`docker/vqa-smoke/model-matrix.json`.

## Common roster

The common sweep uses all ten model-size rows from `image2.png`: PaliGemma 2
(3B, 10B), Gemma 4 (4B, 12B), Qwen2.5-VL (3B, 7B), InternVL 2.5 8B,
LLaVA-OneVision 7B, Phi-3.5-Vision 4.2B and Idefics3 8B. Every row runs all
three agreed subtasks:

| Subtask | Input | Required normalized output | Primary metric |
|---|---|---|---|
| Directional binary | scene image + top/down or left/right question | `yes` or `no` | accuracy |
| Spatial bbox | scene image + referring expression | one normalized `xyxy` box | IoU >= 0.5 |
| Attribute composition | scene image + composed material/function question | normalized free text | exact-match accuracy |

The embedding score at cosine 0.70 is diagnostic, not a replacement for the
free-text primary metric. Bbox mean IoU, centroid hit and all unparseable rates
are also diagnostics. Binary smoke fixtures must contain both answers so the
known blanket-`yes` failure cannot pass unnoticed.

PaliGemma 2 uses native `<loc>` bbox tokens and Qwen2.5-VL uses native `<box>`
output. All other common-roster models use a strictly parsed prompted bbox.

## Why the other image is not merged into the common roster

`image1.png` is the bbox-comparison shortlist from the handoff. Its additional
GroundingDINO, Florence-2-large, Molmo 2 4B, RoboPoint 13B and CogVLM2 19B rows
are valuable spatial-grounding baselines, but the table does not establish
that they implement all three common VQA contracts. Keep them for a later
bbox-only comparison; adding them to the shared matrix would violate the stated
same-models-for-all-subtasks rule.

## Docker accumulation boundary

There are ten benchmark rows but seven dependency environments. Size variants
within PaliGemma 2, Gemma 4 and Qwen2.5-VL share their family environment. A
cumulative image should add those seven environments in matrix order and keep
weights outside the image. Before implementing each stage, pin and verify the
official source/checkpoint revision and its exact processor/chat template; the
handoff documents that a generic llama.cpp handler previously produced invalid
Gemma output.

This roster step does not copy the pilot's local paths, cached weights or
claimed VRAM/speed estimates into the contract. Those are host-specific and
must be measured by the smoke runner.

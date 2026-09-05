# VQA smoke-test runbook

## Frozen scope

The first supplied roster image is authoritative. In-context/multi-image
evaluation is deferred. The benchmark tasks are:

1. top/down and left/right binary questions;
2. spatial grounding (`lr_extreme`, `closest`, `farthest`) scored by bbox;
3. material+function attribute composition answered with an object name.

GroundingDINO, Florence-2-large, and RoboPoint are bbox-only baselines. They
must not receive binary or attribute rows. The remaining roster entries run
all three task groups. A model's official processor/chat template wraps the
model-neutral prompts; the benchmark must not hand-build image tokens.

## Stages

| Stage | Data | Purpose | Mechanical pass condition |
|---|---|---|---|
| Unit | synthetic rows | parser, coordinates, IoU, schema | all tests pass |
| Smoke | committed 36-row fixture | image load, inference, parse, score | complete IDs, parse rate >= 95%, no crash/OOM |
| Acceptance | frozen stratified development set | catch model/task integration errors | thresholds calibrated per model and task |
| Paper gate | full eligible dataset | publishable result | frozen config, no missing rows, full report |

Smoke is intentionally not an accuracy claim. A weak model can pass plumbing
smoke and still fail acceptance.

## Dataset contract

- One RGB frame per question.
- MOS RGB comes from Hub revision `main`; ego RGB comes from
  `ego-preview-v2`.
- Coordinates are `[xmin,ymin,xmax,ymax]`, inclusive, in the original image.
- The model never sees scene ID, phase, evidence, target IDs, or answers.
- Object underscores are displayed as spaces and normalized back only during
  scoring.
- Raw output is always retained. Parsing failures are metrics, not silently
  repaired answers.
- Spatial near-ties are excluded from smoke. Future full GT should preserve
  full-precision evidence and an explicit winner margin.

## Determinism

Use greedy decoding (`do_sample=false`, `num_beams=1`), pin model and dataset
revisions, record package/container digests, and save the exact request and raw
response for every sample. PaliGemma bboxes are decoded from normalized
`<loc>` tokens in `yxyx` token order. Other generative models return strict
JSON in original-image `xyxy` coordinates; an adapter is responsible for
undoing any processor resize before writing its raw benchmark response.

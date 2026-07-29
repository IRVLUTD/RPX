# Depth Models — Who Owns What, What Works

**Pick your model. Run the install. Smoke it. Update the table.**

For the pinned two-gate launcher, environment matrix and Easy-only rollout,
use [`depth_smoke_runbook.md`](depth_smoke_runbook.md).

Last updated: 2026-07-02

---

## Quick read

We have 20 depth models in the paper (10 single-image, 10 video). The code is
wired up for all of them. **Four single-image models have passed a real GPU
acceptance gate**: DA-V2 Large, Depth Pro, UniDepth V2 and MoGe-2. HyDen reached
the official checkpoint and was denied access; Lotus-2's corrected dependency
recipe still needs a fresh server setup and smoke. The remaining official
adapters need their isolated upstream environment and smoke run.

| | Code wired | Acceptance passed | Unverified-weight block |
| --- | --- | --- | --- |
| Single-image depth | 10 / 10 | 4 / 10 | 0 |
| Video depth | 10 / 10 | 0 / 10 | 2 (D4RT, GemDepth) |

**The two remaining weight-lineage blocks are D4RT and GemDepth.** FE2E now has
an official release; its one-frame Docker/GPU acceptance is still pending.

---

## Single-image depth (10 models)

The team CLI: `python scripts/run_depth.py --model <name> --split easy`

| Model | Owner | Status | What's needed |
| --- | --- | --- | --- |
| **da-v2-large** | — | ✅ Done | Already tested on RTX 5070 — works |
| **da3-metric-l** | — | ⏳ Needs smoke | `pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3` then smoke |
| **depth-pro** | — | ✅ Passed (2026-07-02, pop-os, RTX 5060 8 GB) | Micro 1/1 + acceptance 25/25; acceptance RMSE 0.5981, AbsRel 0.1390, delta1 0.9092 |
| **fe2e** | — | 🐳 Docker GPU validation pending | Official `AMAP-ML/FE2E` release and author-linked `exander/FE2E` checkpoint are pinned in `docker/depth-fe2e`; run the one-frame CUDA acceptance before production |
| **hyden** | — | 🔐 Access blocked (2026-07-02, pop-os) | Environment/import/CUDA checks pass, but the official `facebook/hyden-mogev2-metric-point` checkpoint returns HTTP 403 for the current HF account. Request official access, then rerun on the server; do not substitute weights. |
| **lotus-2** | — | ⚠️ Server retest | Setup exposed a Diffusers/Transformers 5 incompatibility. The recipe now pins compatible Transformers 4.46.3, but the two-attempt local ceiling was reached before inference. Recreate/resume the corrected environment on the server. |
| **zipdepth** | — | 🐳 Docker GPU validation pending | Official `fabiotosi92/ZipDepth` source and GPU checkpoint are pinned in `docker/depth-zipdepth`; pooled `ls_disparity` matches the released inverse-depth evaluator |
| **metric3d-v2** | — | ⏳ Needs smoke | (existing torch.hub) |
| **moge-2-vit-l** | — | ✅ Passed (2026-07-02, pop-os, RTX 5060 8 GB) | Micro 1/1 + acceptance 25/25; acceptance RMSE 0.6235, AbsRel 0.1769, delta1 0.7502 |
| **unidepth-v2** | — | ✅ Passed (2026-07-02, pop-os, RTX 5060 8 GB) | Micro 1/1 + acceptance 25/25; acceptance RMSE 0.4978, AbsRel 0.0707, delta1 0.9479 |

---

## Video depth (10 models)

The team CLI: `python scripts/run_video_depth.py --model <name> --split easy`

| Model | Owner | Status | What's needed |
| --- | --- | --- | --- |
| **chrono-depth** | — | ⏳ Needs smoke | Setup helper pins `github.com/jiahao-shao1/ChronoDepth` and its custom SVD pipeline |
| **d4rt** | — | 🚫 Blocked | Find the official upstream release |
| **da3-video** | — | ⏳ Needs smoke | Same install as `da3-metric-l` (shared weights) |
| **depth-crafter** | — | ⏳ Needs smoke | Setup helper pins the Python-3.11-compatible official v1.0.1 source |
| **gem-depth** | — | 🚫 Blocked | Find the official upstream release |
| **monst3r** | — | ⏳ Needs smoke | `pip install git+https://github.com/Junyi42/monst3r` |
| **rolling-depth** | — | ⏳ Needs smoke | `git clone github.com/prs-eth/rollingdepth && pip install -e .` |
| **vggt-omega** | — | ⏳ Needs smoke | `git clone github.com/facebookresearch/vggt && pip install -e .` |
| **video-da** | — | ⏳ Needs smoke | `git clone github.com/DepthAnything/Video-Depth-Anything && pip install -e .` |
| **vigeo** | — | ⏳ Needs smoke | `git clone github.com/aigc3d/ViGeo && pip install -e .` |

---

## How to smoke your model

Follow [`depth_smoke_runbook.md`](depth_smoke_runbook.md) exactly. It creates
the pinned environment, enforces CUDA and an idle selected GPU, runs the micro
then acceptance gate, and validates all required artefacts. Do not mark a row
passed after an import-only check, a micro gate, a CPU run, a 4-bit DepthLM
diagnostic, or an 8 GB OOM. Do not start Medium/Hard or upload anything while
the Easy pilot is still under review.

---

## When things break

The most common issue: the model loads fine but `predict()` fails because the model's output shape doesn't match what the adapter expects.

What you'll see:
```
AdapterError: <model>: predict returned depth_seq shape (T, X, Y); expected (T, H, W) = (...)
```

The fix is usually 5 lines in the adapter's `predict()` body — squeeze a channel dim, transpose, resize, etc. Edit `scripts/depth_models/<your-model>.py` or `scripts/video_depth_models/<your-model>.py`, push a fix-up PR, re-smoke.

If the error is something else (CUDA OOM, missing package, weights download fails), drop a message in #rpx-eng and we'll triage.

---

## The 3 blocked models — what to do

For **D4RT** and **GemDepth**, the candidate HF repositories still lack verified paper-author lineage. FE2E is no longer in this category: its official AMAP-ML repository now links the `exander/FE2E` checkpoint directly.

To unblock any of them:

1. Find the paper's official GitHub repo (check the paper PDF's "code" link or arxiv "code" tab)
2. Confirm whether the weights match what's on the candidate HF repo (or use the official repo's instructions instead)
3. Update the relevant video adapter (`scripts/video_depth_models/d4rt.py` or `gem_depth.py`) — drop the safety-rail check and wire up the real load path
4. Smoke it, then update this table

---

## Important rule before publication

A model goes into paper Table 3 or Table 4 **only after its row here says ✅ Done**. "Wired up" just means the code is correct — it doesn't mean the model has actually been run on real data. The smoke is what catches the per-model surprises.

---

## Once you're done

After every model in your list shows ✅, we run the downstream pipeline:

1. Aggregate all the `cells.parquet` files from Box into one big table
2. Compute Φ (phase stability) and J (deployment desirability) per model
3. Fill those numbers into Tables 3 and 4 in the paper
4. PDF rebuild + share for review

ETA: 24 hours after the last sweep finishes.

---

## Other tasks (D2–D7) — not started yet

When we start on Detection / Tracking / QA / Pose / NVS, the same pattern repeats. The recipe is in **`benchmark/docs/new_task_playbook.md`** — it's an 8-step checklist so the next task doesn't take as long as D1 did.

Owners for D2–D7? Sign up in #rpx-eng.

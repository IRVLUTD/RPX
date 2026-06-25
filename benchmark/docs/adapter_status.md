# Depth Models — Who Owns What, What Works

**Pick your model. Run the install. Smoke it. Update the table.**

Last updated: 2026-06-25

---

## Quick read

We have 20 depth models in the paper (10 single-image, 10 video). The code is wired up for all of them. **Only 1 has actually been run on a GPU end-to-end so far** — the rest just need somebody to install the upstream package and do a 10-minute test.

| | Code wired | Actually run on GPU | Blocked |
| --- | --- | --- | --- |
| Single-image depth | 10 / 10 | 1 / 10 | 1 (FE2E) |
| Video depth | 10 / 10 | 0 / 10 | 2 (D4RT, GemDepth) |

**The "Blocked" 3 are waiting on someone to find the official model release** — the HF repos we found for them look like community uploads, not the paper authors' actual weights.

---

## Single-image depth (10 models)

The team CLI: `python scripts/run_depth.py --model <name> --split easy`

| Model | Owner | Status | What's needed |
| --- | --- | --- | --- |
| **da-v2-large** | — | ✅ Done | Already tested on RTX 5070 — works |
| **da3-metric-l** | — | ⏳ Needs smoke | `pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3` then smoke |
| **depth-pro** | — | ⏳ Needs smoke | (package already installed in repo) — just run the smoke |
| **depthlm** | — | ⏳ Needs smoke | `pip install transformers torch accelerate` — needs ≥24 GB VRAM |
| **fe2e** | — | 🚫 Blocked | Find the official upstream release (community HF repo looks unofficial) |
| **hyden** | — | ⏳ Needs smoke | (existing) |
| **lotus-2** | — | ⏳ Needs smoke | `pip install diffusers transformers` |
| **metric3d-v2** | — | ⏳ Needs smoke | (existing torch.hub) |
| **moge-2-vit-l** | — | ⏳ Needs smoke | `pip install moge` |
| **unidepth-v2** | — | ⏳ Needs smoke | (existing) |

---

## Video depth (10 models)

The team CLI: `python scripts/run_video_depth.py --model <name> --split easy`

| Model | Owner | Status | What's needed |
| --- | --- | --- | --- |
| **chrono-depth** | — | ⏳ Needs smoke | `git clone github.com/jhShao/ChronoDepth && pip install -e .` |
| **d4rt** | — | 🚫 Blocked | Find the official upstream release |
| **da3-video** | — | ⏳ Needs smoke | Same install as `da3-metric-l` (shared weights) |
| **depth-crafter** | — | ⏳ Needs smoke | `git clone github.com/Tencent/DepthCrafter && pip install -e .` |
| **gem-depth** | — | 🚫 Blocked | Find the official upstream release |
| **monst3r** | — | ⏳ Needs smoke | `pip install git+https://github.com/Junyi42/monst3r` |
| **rolling-depth** | — | ⏳ Needs smoke | `git clone github.com/prs-eth/rollingdepth && pip install -e .` |
| **vggt-omega** | — | ⏳ Needs smoke | `git clone github.com/facebookresearch/vggt && pip install -e .` |
| **video-da** | — | ⏳ Needs smoke | `git clone github.com/DepthAnything/Video-Depth-Anything && pip install -e .` |
| **vigeo** | — | ⏳ Needs smoke | `git clone github.com/aigc3d/ViGeo && pip install -e .` |

---

## How to smoke your model (10 minutes)

```bash
# 1. SSH to the lab GPU box
ssh <lab-gpu>
cd ~/code/RPX/benchmark
git pull

# 2. Install your model's upstream package (see "What's needed" above)
<the install command for your model>

# 3. Run a small test on one scene
PYTHONPATH=. python scripts/run_depth.py \
    --model <your-model> --split easy --max-samples 25

# Or for video models:
PYTHONPATH=. python scripts/run_video_depth.py \
    --model <your-model> --split easy --frame-budget 25 --sampling stride

# 4. Check the output looks reasonable
ls rpx_results/<your-model>/easy/
# Should have: cells.parquet, result.json, summary.md

# 5. If it works, kick off the full sweep:
PYTHONPATH=. python scripts/run_depth.py \
    --model <your-model> --split easy --upload-to-box
# (also do --split medium and --split hard)

# 6. Edit this table: replace "⏳ Needs smoke" with "✅ Done (your-name, date)"
```

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

For **FE2E**, **D4RT**, and **GemDepth** the HF repos we found look unofficial (empty READMEs, author personal handles, no clear link to the paper authors). Running them right now would risk publishing benchmark numbers under those model names that aren't actually the paper's model.

To unblock any of them:

1. Find the paper's official GitHub repo (check the paper PDF's "code" link or arxiv "code" tab)
2. Confirm whether the weights match what's on the candidate HF repo (or use the official repo's instructions instead)
3. Update the adapter file (`scripts/depth_models/fe2e.py` for FE2E, or `scripts/video_depth_models/d4rt.py` / `gem_depth.py` for the other two) — drop the safety-rail check and wire up the real load path
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

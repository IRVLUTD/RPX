# Adapter Status — Live

**Last verified**: 2026-06-25 via WebFetch against each upstream HF model card.
**Updated by**: whoever ships a new adapter / verifies a candidate weights repo / passes a real smoke.

This is the source of truth for "what works today" across all benchmark tasks.
The canonical roster lives in `rpx_benchmark/adapters/depth_scaffold.py:DEPTH_MODEL_CARDS`.

**Two columns to read carefully**:

- **Status**: adapter-code state. "READY" means the adapter file exists, the model_id is verified, and the shape contract is enforced. It does NOT mean the model has actually been run end-to-end.
- **Smoke**: real-GPU end-to-end test state. "passed/<host>" means weights loaded, `predict()` ran on a real RPX clip, `cells.parquet` emitted with reasonable numbers. "pending" means nobody has done this yet on real hardware.

Until a row's Smoke column flips to "passed/<host>", **do not publish numbers from it in Table 3 / 4**.

---

## D1 — Depth (20 / 20 slots filled; 2 / 17 smoke-passed end-to-end)

### Image Depth (10)

| # | Roster key | Display name | Status | Smoke | Upstream | Install hint |
|---|---|---|---|---|---|---|
| 1 | `da-v2-large` | DA-V2 Large | ✓ READY | **✓ passed/RTX 5070** (2026-06-25) | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | `pip install transformers torch timm pillow` |
| 2 | `da3-metric-l` | DA3 Metric-L | ✓ READY (needs pkg) | ○ pending lab GPU | `depth-anything/DA3-LARGE` (ByteDance) | `pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3` |
| 3 | `depth-pro` | Depth Pro | ✓ READY | ○ pending lab GPU | (existing in `scripts/depth_models/depth_pro.py`) | (existing — `pip install depth-pro`) |
| 4 | `depthlm` | DepthLM-12B | ✓ READY (needs pkg, 40+ GB VRAM) | ○ pending lab GPU | `facebook/DepthLM` | `pip install transformers torch accelerate` |
| 5 | `fe2e` | FE2E | ⚠ UNVERIFIED (safety rail) | ✗ blocked — no verified weights | candidate: `exander/FE2E` (empty README) | locate official upstream first |
| 6 | `hyden` | HyDen Metric | ✓ READY (needs pkg) | ○ pending lab GPU | (existing) | (existing) |
| 7 | `lotus-2` | Lotus-2 | ✓ READY (needs pkg) | ○ pending lab GPU | `jingheya/lotus-depth-d-v2-0-disparity` | `pip install diffusers transformers` |
| 8 | `metric3d-v2` | Metric3D-V2-ViT-Giant | ✓ READY | ○ pending lab GPU | (existing torch.hub) | (existing) |
| 9 | `moge-2-vit-l` | MoGe-2-ViTL | ✓ READY (needs pkg) | ○ pending lab GPU | `Ruicheng/moge-2-vitl-normal` | `pip install moge` (or upstream repo) |
| 10 | `unidepth-v2` | UniDepth-V2-ViTL14 | ✓ READY | ○ pending lab GPU | (existing) | (existing) |

**Image Depth smoke summary**: 1 of 10 smoke-passed end-to-end (`da-v2-large`).
Each "pending" row is one team-member-PR away from "passed": `pip install <pkg>` → `python scripts/run_depth.py --model <key> --split easy --max-samples 25` → verify `cells.parquet`.

### Video Depth (10)

| # | Roster key | Display name | Status | Smoke | Upstream | Install hint |
|---|---|---|---|---|---|---|
| 1 | `chrono-depth` | ChronoDepth | ✓ READY (needs pkg) | ○ pending lab GPU | `jhshao/ChronoDepth` + github | `git clone github.com/jhShao/ChronoDepth && pip install -e .` |
| 2 | `d4rt` | D4RT | ⚠ UNVERIFIED (safety rail) | ✗ blocked — no verified weights | candidate: `AlysonIrene/D4RT_checkpoint` | locate official upstream first |
| 3 | `da3-video` | DA3 (video) | ✓ READY (needs pkg) | ○ pending lab GPU | `depth-anything/DA3-LARGE` (same weights as `da3-metric-l`) | `pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3` |
| 4 | `depth-crafter` | DepthCrafter | ✓ READY (needs pkg) | ○ pending lab GPU | `tencent/DepthCrafter` + github | `git clone github.com/Tencent/DepthCrafter && pip install -e .` |
| 5 | `gem-depth` | GemDepth | ⚠ UNVERIFIED (safety rail) | ✗ blocked — no verified weights | candidate: `YuechengLiu/GemDepth` (empty README) | locate official upstream first |
| 6 | `monst3r` | MonST3R | ✓ READY (needs pkg) | ○ pending lab GPU | `Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt` | `pip install git+https://github.com/Junyi42/monst3r` |
| 7 | `rolling-depth` | RollingDepth | ✓ READY (needs pkg) | ○ pending lab GPU | `prs-eth/rollingdepth-v1-0` + github | `git clone github.com/prs-eth/rollingdepth && pip install -e .` |
| 8 | `vggt-omega` | VGGT-Ω | ✓ READY (needs pkg) | ○ pending lab GPU | `facebook/VGGT-1B` + github | `git clone github.com/facebookresearch/vggt && pip install -e .` |
| 9 | `video-da` | Video Depth Anything | ✓ READY (needs pkg) | ○ pending lab GPU | `depth-anything/Video-Depth-Anything-Large` + github | `git clone github.com/DepthAnything/Video-Depth-Anything && pip install -e .` |
| 10 | `vigeo` | ViGeo | ✓ READY (needs pkg) | ○ pending lab GPU | `pkqbajng/ViGeo` + github | `git clone github.com/aigc3d/ViGeo && pip install -e .` |

**Video Depth smoke summary**: 0 of 10 smoke-passed end-to-end. (Adjacent: the `da-v2-video` frame-as-video baseline — outside the canonical roster — passed smoke on RTX 5070 on 2026-06-25, but it isn't one of the paper Table 4 rows.)

**Total smoke status: 1 of 17 real adapters smoke-passed end-to-end on a real GPU.** The other 16 need per-model `pip install` + one-clip smoke on the lab GPU before their numbers ship.

---

## D2–D7 — Other tasks (not yet audited)

Each of the remaining 6 tasks has its own ~10-model roster. These haven't been WebFetch-verified or wired to the canonical bridge yet. **Per-task team owners should follow the playbook in [`new_task_playbook.md`](./new_task_playbook.md) when they start their task** — that doc captures every pattern (canonical bridge, safety rails, smoke protocol, etc.) so the verification work doesn't get redone per task.

| Task | Canonical roster location | Status |
|---|---|---|
| D2 Detection | (none yet — needs the same scaffold treatment as D1) | not started |
| D3 Tracking | (none yet) | not started |
| D4 Scene QA | (none yet) | not started |
| D5 Spatial QA | (none yet) | not started |
| D6 Relative Pose | (none yet — model-side scripts exist under `scripts/pose_models/`) | not started |
| D7 NVS | (none yet — model-side scripts exist under `scripts/nvs_models/`) | not started |

---

## Status legend

**Status column (adapter code state):**

- **✓ READY** — adapter exists, model_id verified against upstream model card, shape contract enforced, can be invoked. Does NOT mean it has actually been run on real weights.
- **✓ READY (needs pkg)** — same as READY, but the upstream Python package must be `pip install`ed first. Without it, the adapter raises a clean `ImportError` with the exact install command.
- **⚠ UNVERIFIED (safety rail)** — adapter exists and CLI resolves the canonical name, but actually running it requires `--acknowledge-unverified` (CLI) or `acknowledge_unverified=True` (constructor). Candidate weights are from author/community HF handles with no documented lineage to the paper authors.

**Smoke column (real-GPU verification state):**

- **✓ passed/&lt;host&gt;** — weights loaded on real GPU, `predict()` ran on a real RPX clip, `cells.parquet` emitted with reasonable numbers. Date + host recorded.
- **○ pending lab GPU** — no real smoke yet. One team-member-PR away.
- **✗ blocked** — cannot smoke because the upstream weights are not verified (safety-railed adapters). Will become "pending" once lineage is confirmed and the safety rail is dropped.

## Publication gate

**Do not publish a row's numbers in paper Table 3 / 4 / etc. until that row's Smoke column reads "✓ passed".** A "READY" adapter is necessary but not sufficient — the predict-output shape, alignment behaviour, and metric values all need real-data validation per model.

## Updating this table

When you ship or fix an adapter — or pass a smoke — update this file in the same PR. The merge bot enforces nothing; the discipline does.

# Adapter Status — Live

**Last verified**: 2026-06-25 via WebFetch against each upstream HF model card.
**Updated by**: whoever ships a new adapter / verifies a candidate weights repo.

This is the source of truth for "what works today" across all benchmark tasks.
The canonical roster lives in `rpx_benchmark/adapters/depth_scaffold.py:DEPTH_MODEL_CARDS`.

---

## D1 — Depth (20 / 20 slots filled)

### Image Depth (10)

| # | Roster key | Display name | Status | Upstream | Install hint |
|---|---|---|---|---|---|
| 1 | `da-v2-large` | DA-V2 Large | ✓ READY | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | `pip install transformers torch timm pillow` |
| 2 | `da3-metric-l` | DA3 Metric-L | ✓ READY (needs pkg) | `depth-anything/DA3-LARGE` (ByteDance) | `pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3` |
| 3 | `depth-pro` | Depth Pro | ✓ READY | (existing) | (existing) |
| 4 | `depthlm` | DepthLM-12B | ✓ READY (needs pkg + 40GB+ VRAM) | `facebook/DepthLM` | `pip install transformers torch accelerate` |
| 5 | `fe2e` | FE2E | ⚠ UNVERIFIED (safety rail) | candidate: `exander/FE2E` (empty README) | locate official upstream first |
| 6 | `hyden` | HyDen Metric | ✓ READY (needs pkg) | (existing) | (existing) |
| 7 | `lotus-2` | Lotus-2 | ✓ READY (needs pkg) | `jingheya/lotus-depth-d-v2-0-disparity` | `pip install diffusers transformers` |
| 8 | `metric3d-v2` | Metric3D-V2-ViT-Giant | ✓ READY | (existing torch.hub) | (existing) |
| 9 | `moge-2-vit-l` | MoGe-2-ViTL | ✓ READY (needs pkg) | `Ruicheng/moge-2-vitl-normal` | `pip install moge` (or upstream repo) |
| 10 | `unidepth-v2` | UniDepth-V2-ViTL14 | ✓ READY | (existing) | (existing) |

### Video Depth (10)

| # | Roster key | Display name | Status | Upstream | Install hint |
|---|---|---|---|---|---|
| 1 | `chrono-depth` | ChronoDepth | ✓ READY (needs pkg) | `jhshao/ChronoDepth` + github | `git clone github.com/jhShao/ChronoDepth && pip install -e .` |
| 2 | `d4rt` | D4RT | ⚠ UNVERIFIED (safety rail) | candidate: `AlysonIrene/D4RT_checkpoint` | locate official upstream first |
| 3 | `da3-video` | DA3 (video) | ✓ READY (needs pkg) | `depth-anything/DA3-LARGE` (same weights as `da3-metric-l`) | `pip install -e git+https://github.com/ByteDance-Seed/depth-anything-3` |
| 4 | `depth-crafter` | DepthCrafter | ✓ READY (needs pkg) | `tencent/DepthCrafter` + github | `git clone github.com/Tencent/DepthCrafter && pip install -e .` |
| 5 | `gem-depth` | GemDepth | ⚠ UNVERIFIED (safety rail) | candidate: `YuechengLiu/GemDepth` (empty README) | locate official upstream first |
| 6 | `monst3r` | MonST3R | ✓ READY (needs pkg) | `Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt` | `pip install git+https://github.com/Junyi42/monst3r` |
| 7 | `rolling-depth` | RollingDepth | ✓ READY (needs pkg) | `prs-eth/rollingdepth-v1-0` + github | `git clone github.com/prs-eth/rollingdepth && pip install -e .` |
| 8 | `vggt-omega` | VGGT-Ω | ✓ READY (needs pkg) | `facebook/VGGT-1B` + github | `git clone github.com/facebookresearch/vggt && pip install -e .` |
| 9 | `video-da` | Video Depth Anything | ✓ READY (needs pkg) | `depth-anything/Video-Depth-Anything-Large` + github | `git clone github.com/DepthAnything/Video-Depth-Anything && pip install -e .` |
| 10 | `vigeo` | ViGeo | ✓ READY (needs pkg) | `pkqbajng/ViGeo` + github | `git clone github.com/aigc3d/ViGeo && pip install -e .` |

**Smoke-tested on RTX 5070 (8 GB) end-to-end with real data**: `da-v2-large` (Image Depth), `da-v2-video` (Video Depth frame-as-video baseline, not in the canonical roster).

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

- **✓ READY** — adapter exists, model_id verified against upstream model card, can be invoked
- **✓ READY (needs pkg)** — same, but the upstream Python package must be `pip install`ed first; adapter raises a clean `ImportError` with the exact install command otherwise
- **⚠ UNVERIFIED (safety rail)** — adapter exists and CLI resolves the canonical name, but actually running it requires `--acknowledge-unverified` (CLI) or `acknowledge_unverified=True` (constructor). The candidate weights are from author/community HF handles with no documented lineage to the paper authors. **Do not publish numbers from a safety-railed adapter without confirming lineage first.**

## Updating this table

When you ship or fix an adapter, update this file in the same PR. The merge bot enforces nothing — the discipline does.

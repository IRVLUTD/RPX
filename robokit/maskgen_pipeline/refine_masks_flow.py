#!/usr/bin/env python3
# Flow-guided SAM2 propagation (seed from first/last correct frame; two-direction meet-in-middle, no fusion)
# -----------------------------------------------------------------------------
# Requirements: opencv-python, numpy, pillow, torch, tqdm
# Project-local deps: robokit.perception.SAM2Predictor, .config.logger, .mask_processing helpers
# -----------------------------------------------------------------------------

import argparse, os, shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image as PILImg
import torch
from tqdm import tqdm
from contextlib import nullcontext
from typing import Optional, Tuple

# Project-local
from robokit.perception import SAM2Predictor
from .config import logger
from .mask_processing import (
    load_palette,
    apply_palette,
    merge_rgb_with_mask,
    merge_rgb_with_mask_with_contours,
)

OUTPUT_DIRS = ["masks", "palette", "rgb_and_mask", "contour_gt_masks"]

# -------------------- utils --------------------

def z5(n: int) -> str: return f"{n:05d}"

def read_faulty_list(fp: Path) -> list[int]:
    if not fp.exists():
        raise FileNotFoundError(f"faulty file not found: {fp}")
    out = []
    for line in fp.read_text().splitlines():
        s = line.strip()
        if s:
            out.append(int(s))
    return sorted(set(out))

def group_contiguous(nums: list[int]) -> list[list[int]]:
    if not nums: return []
    runs, cur = [], [nums[0]]
    for a, b in zip(nums, nums[1:]):
        if b == a + 1: cur.append(b)
        else: runs.append(cur); cur = [b]
    runs.append(cur)
    return runs

def image_gray_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return img

def image_rgb_pil(path: Path) -> PILImg:
    img = PILImg.open(path).convert("RGB")
    return img

def label_png_read(path: Path) -> np.ndarray:
    arr = np.array(PILImg.open(path))
    if arr.dtype != np.uint16:
        # tolerate uint8 instance maps too
        arr = arr.astype(np.uint16)
    return arr

def label_png_write(path: Path, arr: np.ndarray):
    PILImg.fromarray(arr.astype(np.uint16)).save(path)

def ensure_dirs(base: Path, subs: list[str]):
    for s in subs:
        (base / s).mkdir(parents=True, exist_ok=True)

def ids_from_labelmap(lbl: np.ndarray) -> np.ndarray:
    ids = np.unique(lbl)
    return ids[(ids != 0)]

def bbox_from_binary(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def clean_small_regions(binmask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area (4-connectivity)."""
    num, labels = cv2.connectedComponents(binmask.astype(np.uint8), connectivity=4)
    out = np.zeros_like(binmask, dtype=bool)
    for comp in range(1, num):
        area = (labels == comp).sum()
        if area >= min_area:
            out[labels == comp] = True
    return out

def labelmap_to_bboxes(lbl: np.ndarray, min_area_px: int, morph_ksize: int = 3) -> tuple[list[int], list[tuple[int,int,int,int]]]:
    ids = ids_from_labelmap(lbl)
    bboxes, keep_ids = [], []
    if morph_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    for oid in ids:
        binm = (lbl == oid)
        # small-blob cleanup + slight closing to fix holes
        binm = clean_small_regions(binm, min_area_px)
        if morph_ksize > 0:
            binm = cv2.morphologyEx(binm.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        bb = bbox_from_binary(binm)
        if bb is not None:
            keep_ids.append(int(oid))
            bboxes.append(bb)
    return keep_ids, bboxes

def paint_from_masks_bool(masks_bool: np.ndarray, order_ids: list[int]) -> np.ndarray:
    """masks_bool: [K,H,W] bools; order_ids: K instance IDs in same order."""
    H, W = masks_bool.shape[1], masks_bool.shape[2]
    out = np.zeros((H, W), dtype=np.uint16)
    K = min(masks_bool.shape[0], len(order_ids))
    for k in range(K):
        out[masks_bool[k]] = np.uint16(order_ids[k])
    return out

# -------------------- optical flow warping --------------------

def farneback_flow(prev_gray: np.ndarray, next_gray: np.ndarray,
                   pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2) -> np.ndarray:
    """
    Returns flow F such that: pos in prev + F(prev_pos) ~= corresponding pos in next.
    Shape (H,W,2), dtype float32.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
    return flow

def warp_label_from_next_to_prev(next_label: np.ndarray, flow_prev_to_next: np.ndarray) -> np.ndarray:
    """
    We have label L_next at time t, and flow F : prev -> next.
    To get label estimate at prev, sample L_next at coords (x+F_x, y+F_y) for each (x,y) in prev.
    """
    H, W = next_label.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + flow_prev_to_next[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_prev_to_next[..., 1]).astype(np.float32)
    lo = (next_label & 0xFF).astype(np.uint8)
    hi = ((next_label >> 8) & 0xFF).astype(np.uint8)
    lo_warp = cv2.remap(lo, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    hi_warp = cv2.remap(hi, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    warped = (hi_warp.astype(np.uint16) << 8) | lo_warp.astype(np.uint16)
    return warped

def warp_label_from_prev_to_next(prev_label: np.ndarray, flow_prev_to_next: np.ndarray) -> np.ndarray:
    """
    We have label L_prev at time t-1, and flow F : prev -> next.
    To get label estimate at next, sample L_prev at coords (x-F_x, y-F_y) for each (x,y) in next.
    """
    H, W = prev_label.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x - flow_prev_to_next[..., 0]).astype(np.float32)
    map_y = (grid_y - flow_prev_to_next[..., 1]).astype(np.float32)
    lo = (prev_label & 0xFF).astype(np.uint8)
    hi = ((prev_label >> 8) & 0xFF).astype(np.uint8)
    lo_warp = cv2.remap(lo, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    hi_warp = cv2.remap(hi, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    warped = (hi_warp.astype(np.uint16) << 8) | lo_warp.astype(np.uint16)
    return warped

# -------------------- propagation (backward + forward) --------------------

def propagate_run_flow_guided_backward(run_nums: list[int], seed_num: int,
                                       rgb_dir: Path, mask_dir: Path,
                                       sam2: SAM2Predictor, amp_ctx,
                                       out_root: Path,
                                       min_area_px: int = 50,
                                       morph_ksize: int = 3):
    """
    Backward: seed_num is the first correct AFTER the run. Propagates seed -> ... down to run start.
    """
    seed_name = z5(seed_num)
    seed_rgb_p = rgb_dir / f"{seed_name}.png"
    seed_msk_p = mask_dir / f"{seed_name}.png"
    if not seed_rgb_p.exists() or not seed_msk_p.exists():
        logger.warning(f"[run {run_nums}] backward seed missing: {seed_rgb_p} or {seed_msk_p}; skipping backward.")
        return

    cur_lbl = label_png_read(seed_msk_p)
    cur_seed_name = seed_name

    for tgt in sorted(run_nums, reverse=True):
        tgt_name = z5(tgt)
        prev_gray = image_gray_u8(rgb_dir / f"{tgt_name}.png")
        next_gray = image_gray_u8(rgb_dir / f"{cur_seed_name}.png")

        flow = farneback_flow(prev_gray, next_gray)  # prev->next
        warped_lbl_prev = warp_label_from_next_to_prev(cur_lbl, flow)

        order_ids, bboxes = labelmap_to_bboxes(warped_lbl_prev, min_area_px=min_area_px, morph_ksize=morph_ksize)
        if len(bboxes) == 0:
            logger.warning(f"[run {run_nums}] {tgt_name}: no boxes from warped mask; skipping SAM2 and leaving as-is.")
            continue

        tmp_dir = out_root.parent / f"tmp_flow_{cur_seed_name}_to_{tgt_name}"
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        rgb = image_rgb_pil(rgb_dir / f"{tgt_name}.png")
        rgb.save(tmp_dir / f"{tgt_name}.jpg", format="JPEG", quality=100)

        with torch.inference_mode(), amp_ctx:
            seg = sam2.propagate_bbox_prompt_masks_and_save(str(tmp_dir), bboxes)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if f"{tgt_name}.jpg" not in seg:
            logger.warning(f"[run {run_nums}] SAM2 returned no masks for {tgt_name}.")
            continue

        masks = seg[f"{tgt_name}.jpg"]
        masks_arr = np.asarray(masks)
        if masks_arr.ndim == 4: masks_arr = masks_arr.squeeze(1)
        masks_arr = masks_arr.astype(bool)

        lbl_tgt = paint_from_masks_bool(masks_arr, order_ids)

        name_png = f"{tgt_name}.png"
        label_png_write(out_root / "masks" / name_png, lbl_tgt)

        pal = apply_palette(torch.from_numpy(lbl_tgt.astype(np.int64)), load_palette())
        pal.save(out_root / "palette" / name_png)

        rgb_img = image_rgb_pil(rgb_dir / f"{tgt_name}.png")
        blend = merge_rgb_with_mask(rgb_img, pal)
        blend.save(out_root / "rgb_and_mask" / name_png)

        contour = merge_rgb_with_mask_with_contours(rgb_img, pal)
        contour.save(out_root / "contour_gt_masks" / name_png)

        cur_lbl = lbl_tgt
        cur_seed_name = tgt_name  # advance backward chain

def propagate_run_flow_guided_forward(run_nums: list[int], seed_num: int,
                                      rgb_dir: Path, mask_dir: Path,
                                      sam2: SAM2Predictor, amp_ctx,
                                      out_root: Path,
                                      min_area_px: int = 50,
                                      morph_ksize: int = 3):
    """
    Forward: seed_num is the last correct BEFORE the run. Propagates seed -> ... up to run end.
    """
    seed_name = z5(seed_num)
    seed_rgb_p = rgb_dir / f"{seed_name}.png"
    seed_msk_p = mask_dir / f"{seed_name}.png"
    if not seed_rgb_p.exists() or not seed_msk_p.exists():
        logger.warning(f"[run {run_nums}] forward seed missing: {seed_rgb_p} or {seed_msk_p}; skipping forward.")
        return

    cur_lbl = label_png_read(seed_msk_p)
    cur_seed_name = seed_name

    for tgt in sorted(run_nums):  # forward in time
        tgt_name = z5(tgt)
        prev_gray = image_gray_u8(rgb_dir / f"{cur_seed_name}.png")
        next_gray = image_gray_u8(rgb_dir / f"{tgt_name}.png")

        flow = farneback_flow(prev_gray, next_gray)  # prev->next
        warped_lbl_next = warp_label_from_prev_to_next(cur_lbl, flow)

        order_ids, bboxes = labelmap_to_bboxes(warped_lbl_next, min_area_px=min_area_px, morph_ksize=morph_ksize)
        if len(bboxes) == 0:
            logger.warning(f"[run {run_nums}] {tgt_name}: no boxes from warped mask; skipping SAM2 and leaving as-is.")
            continue

        tmp_dir = out_root.parent / f"tmp_flow_{cur_seed_name}_to_{tgt_name}"
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        rgb = image_rgb_pil(rgb_dir / f"{tgt_name}.png")
        rgb.save(tmp_dir / f"{tgt_name}.jpg", format="JPEG", quality=100)

        with torch.inference_mode(), amp_ctx:
            seg = sam2.propagate_bbox_prompt_masks_and_save(str(tmp_dir), bboxes)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if f"{tgt_name}.jpg" not in seg:
            logger.warning(f"[run {run_nums}] SAM2 returned no masks for {tgt_name}.")
            continue

        masks = seg[f"{tgt_name}.jpg"]
        masks_arr = np.asarray(masks)
        if masks_arr.ndim == 4: masks_arr = masks_arr.squeeze(1)
        masks_arr = masks_arr.astype(bool)

        lbl_tgt = paint_from_masks_bool(masks_arr, order_ids)

        name_png = f"{tgt_name}.png"
        label_png_write(out_root / "masks" / name_png, lbl_tgt)

        pal = apply_palette(torch.from_numpy(lbl_tgt.astype(np.int64)), load_palette())
        pal.save(out_root / "palette" / name_png)

        rgb_img = image_rgb_pil(rgb_dir / f"{tgt_name}.png")
        blend = merge_rgb_with_mask(rgb_img, pal)
        blend.save(out_root / "rgb_and_mask" / name_png)

        contour = merge_rgb_with_mask_with_contours(rgb_img, pal)
        contour.save(out_root / "contour_gt_masks" / name_png)

        cur_lbl = lbl_tgt
        cur_seed_name = tgt_name  # advance forward chain

# -------------------- main --------------------

def main(args):
    scene_dir = Path(args.scene_dir)
    rgb_dir   = scene_dir / "rgb"
    sam2_dir  = scene_dir / "sam2"
    mask_dir  = sam2_dir / "masks"
    faulty_fp = sam2_dir / "iter1_faulty.txt"  # keep exactly as in your best-working version

    try:
        faulty = read_faulty_list(faulty_fp)
    except FileNotFoundError as e:
        logger.error(str(e)); return
    if not faulty:
        logger.info("No faulty frames found."); return

    # Output dirs (fresh iter folder)
    out_iter_root = sam2_dir / "mask_refinement" / f"iter_{args.iter}"
    if out_iter_root.exists():
        shutil.rmtree(out_iter_root, ignore_errors=True)
    out_root = out_iter_root / "sam2"
    ensure_dirs(out_root, OUTPUT_DIRS)

    # SAM2 + autocast
    sam2 = SAM2Predictor()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_ctx = torch.autocast(device_type=device_type) if device_type == "cuda" else nullcontext()

    runs = group_contiguous(faulty)
    logger.info(f"Found {len}(runs) contiguous run(s): {runs}")

    for run in runs:
        first_faulty, last_faulty = run[0], run[-1]
        next_seed = last_faulty + 1          # next-good
        prev_seed = first_faulty - 1         # prev-good

        # Check seed availability
        next_ok = (rgb_dir / f"{z5(next_seed)}.png").exists() and (mask_dir / f"{z5(next_seed)}.png").exists()
        prev_ok = (rgb_dir / f"{z5(prev_seed)}.png").exists() and (mask_dir / f"{z5(prev_seed)}.png").exists()

        if next_ok and prev_ok:
            # Meet-in-the-middle split (no overlap, no fusion)
            n = len(run)
            if n % 2 == 1:
                # odd length: backward side owns the true middle
                mid_idx = n // 2
                left_half  = run[:mid_idx]          # forward handles up to before mid
                right_half = run[mid_idx:]          # backward handles mid..end
            else:
                # even length: split evenly, no overlap
                mid_idx = n // 2
                left_half  = run[:mid_idx]          # forward handles first half
                right_half = run[mid_idx:]          # backward handles second half

            if left_half:
                logger.info(f"[run {run}] Forward from prev-good {z5(prev_seed)} over {left_half}")
                propagate_run_flow_guided_forward(
                    run_nums=left_half,
                    seed_num=prev_seed,
                    rgb_dir=rgb_dir,
                    mask_dir=mask_dir,
                    sam2=sam2,
                    amp_ctx=amp_ctx,
                    out_root=out_root,
                    min_area_px=args.min_area,
                    morph_ksize=args.morph_ksize
                )

            if right_half:
                logger.info(f"[run {run}] Backward from next-good {z5(next_seed)} over {right_half}")
                propagate_run_flow_guided_backward(
                    run_nums=right_half,
                    seed_num=next_seed,
                    rgb_dir=rgb_dir,
                    mask_dir=mask_dir,
                    sam2=sam2,
                    amp_ctx=amp_ctx,
                    out_root=out_root,
                    min_area_px=args.min_area,
                    morph_ksize=args.morph_ksize
                )

        elif next_ok:
            # Original behavior (backward only)
            logger.info(f"[run {run}] Seeding from next-good {z5(next_seed)} and propagating backward...")
            propagate_run_flow_guided_backward(
                run_nums=run,
                seed_num=next_seed,
                rgb_dir=rgb_dir,
                mask_dir=mask_dir,
                sam2=sam2,
                amp_ctx=amp_ctx,
                out_root=out_root,
                min_area_px=args.min_area,
                morph_ksize=args.morph_ksize
            )

        elif prev_ok:
            # Forward-only fallback if next-good missing
            logger.info(f"[run {run}] Next-good missing; forward from prev-good {z5(prev_seed)}...")
            propagate_run_flow_guided_forward(
                run_nums=run,
                seed_num=prev_seed,
                rgb_dir=rgb_dir,
                mask_dir=mask_dir,
                sam2=sam2,
                amp_ctx=amp_ctx,
                out_root=out_root,
                min_area_px=args.min_area,
                morph_ksize=args.morph_ksize
            )

        else:
            logger.warning(f"[run {run}] No usable seeds found; skipping run.")

    logger.info("All runs completed.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Flow-guided (Farnebäck) SAM2 mask propagation with two-direction meet-in-the-middle (no fusion).")
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Scene directory containing rgb/ and sam2/ (with masks/ and iter1_faulty.txt).")
    p.add_argument("--iter", type=int, default=1,
                   help="Iteration tag for output folder: sam2/mask_refinement/iter_<iter>/")
    p.add_argument("--min_area", type=int, default=50,
                   help="Minimum area (pixels) to keep warped components when forming bboxes.")
    p.add_argument("--morph_ksize", type=int, default=3,
                   help="Kernel size for morphological closing on warped masks (0 to disable).")
    args = p.parse_args()
    main(args)

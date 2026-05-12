#!/usr/bin/env python3
"""
Run SAM2 on ONLY the verified frames listed in verified_frames.txt.
No video propagation. No bbox_overlay. No logs.

Input structure (your pattern):
base_dir/
  verified_frames.txt
  sceneXX.../
    bboxes_verified/<part>/*.npz
    frames/<part>/rgb/*.png

verified_frames.txt lines look like:
  scene21.jo.4f_0_00026.npz

Behavior:
- For each verified entry:
    - load bboxes from scene/bboxes_verified/<part>/<matching npz>
    - load the corresponding frame image scene/frames/<part>/rgb/<frame_id>.png
    - run SAM2 on that single frame with bbox prompts
    - save results to:
        scene/sam2/<part>/
          masks/<frame_id>.png               (uint16 combined instance mask)
          palette/<frame_id>.png             (colored mask)
          rgb_and_mask/<frame_id>.png        (rgb overlay)
          contour_gt_masks/<frame_id>.png    (rgb overlay + contours)

Note:
- Because we cannot rely on the exact SAM2Predictor single-image API name in your robokit version,
  we run SAM2 through the video API using a TEMP folder with exactly ONE image (so "propagation" = 1 frame).
  This guarantees it works with your existing SAM2Predictor.propagate_bbox_prompt_masks_and_save().
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from robokit.perception import SAM2Predictor

from .mask_processing import (
    load_palette,
    apply_palette,
    combine_masks,
    merge_rgb_with_mask,
    merge_rgb_with_mask_with_contours,
)

OUT_SUBDIRS = ["masks", "palette", "rgb_and_mask", "contour_gt_masks"]


# ------------------------- verified_frames.txt parsing -------------------------


def parse_verified_line(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse:
        scene21.jo.4f_0_00026.npz
    -> (scene_name, part, token_npz_name)
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if not s.endswith(".npz"):
        return None
    try:
        left, part, npz_name = s.rsplit("_", 2)
    except ValueError:
        return None
    return left, part, npz_name


def load_verified_index(base_dir: Path) -> List[Tuple[str, str, str]]:
    vf = base_dir / "verified_frames.txt"
    if not vf.exists():
        raise FileNotFoundError(f"Missing verified_frames.txt at: {vf}")

    out = []
    with open(vf, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_verified_line(line)
            if parsed:
                out.append(parsed)
    return out


# ------------------------------ utilities ------------------------------


def ensure_dirs(scene_dir: Path, part: str) -> Dict[str, Path]:
    out_root = scene_dir / "sam2" / part
    out_root.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in OUT_SUBDIRS:
        d = out_root / s
        d.mkdir(parents=True, exist_ok=True)
        out[s] = d
    return out


def load_npz_bboxes(npz_path: Path) -> List[List[float]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "bboxes" not in data:
            return []
        arr = data["bboxes"]
        return arr.tolist() if hasattr(arr, "tolist") else list(arr)
    except Exception as e:
        print(f"[ERROR] Failed loading {npz_path}: {e}")
        return []


def extract_frame_id_from_npz_name(npz_name: str) -> str:
    """
    "00026.npz" -> "00026"
    or fallback to trailing digits if needed.
    """
    stem = Path(npz_name).stem
    if stem.isdigit():
        return stem
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else stem


def resolve_verified_npz(scene_dir: Path, part: str, token_npz_name: str) -> Optional[Path]:
    """
    verified_frames.txt gives token like 00026.npz.
    But actual file in bboxes_verified/<part>/ might be prefixed.
    We resolve by:
      - exact name if exists
      - else glob "*<stem>.npz"
    """
    part_dir = scene_dir / "bboxes_verified" / part
    if not part_dir.exists():
        return None

    exact = part_dir / token_npz_name
    if exact.exists():
        return exact

    stem = Path(token_npz_name).stem
    matches = sorted(part_dir.glob(f"*{stem}.npz"))
    return matches[0] if matches else None


def resolve_frame_path(scene_dir: Path, part: str, frame_id: str) -> Optional[Path]:
    rgb_dir = scene_dir / "frames" / part / "rgb"
    if not rgb_dir.exists():
        return None

    # Prefer PNG (your dataset)
    p = rgb_dir / f"{frame_id}.png"
    if p.exists():
        return p

    # Fallbacks just in case
    for ext in [".jpg", ".jpeg"]:
        q = rgb_dir / f"{frame_id}{ext}"
        if q.exists():
            return q

    return None


def save_all_outputs(out_dirs: Dict[str, Path], palette, frame_pil: Image.Image, masks_tensor: torch.Tensor, frame_id: str) -> None:
    """
    masks_tensor: [N,H,W] (or [N,1,H,W])
    Saves only for this ONE frame.
    """
    if masks_tensor.ndim == 4:
        masks_tensor = masks_tensor.squeeze(1)

    combined = combine_masks(masks_tensor)  # torch [H,W]
    out_name = f"{frame_id}.png"

    # uint16 instance mask
    Image.fromarray(combined.cpu().numpy().astype(np.uint16)).save(out_dirs["masks"] / out_name)

    # palette + overlays
    pal_img = apply_palette(combined, palette)
    pal_img.save(out_dirs["palette"] / out_name)

    blended = merge_rgb_with_mask(frame_pil, pal_img)
    blended.save(out_dirs["rgb_and_mask"] / out_name)

    contour = merge_rgb_with_mask_with_contours(frame_pil, pal_img)
    contour.save(out_dirs["contour_gt_masks"] / out_name)


# ------------------------------ SAM2 single-frame runner ------------------------------


def sam2_segment_single_frame_via_temp_video(sam2: SAM2Predictor, frame_pil: Image.Image, bboxes: List[List[float]]) -> Optional[torch.Tensor]:
    """
    Robust approach: create a temp folder with exactly ONE jpg image
    and call propagate_bbox_prompt_masks_and_save(video_dir, bboxes).
    This works even if SAM2Predictor doesn't expose a stable single-image API.

    Returns:
        masks_tensor [N,H,W] as torch.Tensor, or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_path = tmpdir / "00000.jpg"  # deterministic
        frame_pil.save(img_path, quality=95)

        try:
            with torch.inference_mode(), torch.autocast(sam2.device):
                segments = sam2.propagate_bbox_prompt_masks_and_save(str(tmpdir), bboxes)
        except Exception as e:
            print(f"[ERROR] SAM2 failed on single frame: {e}")
            return None

        if not segments:
            return None

        # Should be exactly one entry
        (fname, masks) = next(iter(segments.items()))
        masks_tensor = torch.tensor(masks)
        if masks_tensor.ndim == 4:
            masks_tensor = masks_tensor.squeeze(1)
        return masks_tensor


def run_one_verified(base_dir: Path, scene_name: str, part: str, token_npz_name: str, sam2: SAM2Predictor, palette, overwrite: bool) -> None:
    scene_dir = base_dir / scene_name
    if not scene_dir.exists():
        print(f"[SKIP] Missing scene dir: {scene_dir}")
        return

    npz_path = resolve_verified_npz(scene_dir, part, token_npz_name)
    if npz_path is None:
        print(f"[SKIP] Missing verified npz: {scene_name}_{part}_{token_npz_name}")
        return

    bboxes = load_npz_bboxes(npz_path)
    if not bboxes:
        print(f"[SKIP] No bboxes in {npz_path}")
        return

    frame_id = extract_frame_id_from_npz_name(token_npz_name)
    frame_path = resolve_frame_path(scene_dir, part, frame_id)
    if frame_path is None:
        print(f"[SKIP] Missing frame image for {scene_name} part={part} frame={frame_id}")
        return

    out_dirs = ensure_dirs(scene_dir, part)

    # Skip if already done (all outputs exist)
    out_name = f"{frame_id}.png"
    if (not overwrite and
        (out_dirs["masks"] / out_name).exists() and
        (out_dirs["palette"] / out_name).exists() and
        (out_dirs["rgb_and_mask"] / out_name).exists() and
        (out_dirs["contour_gt_masks"] / out_name).exists()):
        return

    frame_pil = Image.open(frame_path).convert("RGB")

    masks_tensor = sam2_segment_single_frame_via_temp_video(sam2, frame_pil, bboxes)
    if masks_tensor is None:
        print(f"[ERROR] No masks returned for {scene_name} part={part} frame={frame_id}")
        return

    save_all_outputs(out_dirs, palette, frame_pil, masks_tensor, frame_id)


# ------------------------------ main ------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 ONLY on verified frames listed in verified_frames.txt (single-frame).")
    parser.add_argument("--base_dir", type=str, required=True, help="faulty_collection_gdino_bboxes root folder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    verified_items = load_verified_index(base_dir)
    if not verified_items:
        print("[WARN] No valid entries in verified_frames.txt")
        return

    # Group by scene for cleaner prints
    by_scene: Dict[str, List[Tuple[str, str]]] = {}
    for scene_name, part, token_npz_name in verified_items:
        by_scene.setdefault(scene_name, []).append((part, token_npz_name))

    palette = load_palette()

    print("[INFO] Initializing SAM2Predictor...")
    sam2 = SAM2Predictor()

    for scene_name in sorted(by_scene.keys()):
        entries = by_scene[scene_name]
        print(f"\n[SCENE] {scene_name}  ({len(entries)} verified frames)")
        for part, token_npz_name in tqdm(entries, desc=f"{scene_name}", leave=False):
            run_one_verified(base_dir, scene_name, part, token_npz_name, sam2, palette, overwrite=args.overwrite)

    print("\n[DONE] Finished SAM2 single-frame runs for verified_frames.txt.")


if __name__ == "__main__":
    main()

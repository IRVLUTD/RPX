# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + Grok 3)
# ----------------------------------------------------------------------------------------------------
import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image as PILImg
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from robokit.perception import SAM2Predictor
from robokit.utils import annotate, overlay_masks, combine_masks
from .config import logger
from .mask_processing import (
    load_palette,
    apply_palette,
    merge_rgb_with_mask,
    merge_rgb_with_mask_with_contours,
)

import argparse
import matplotlib.pyplot as plt

# Define output directories
OUTPUT_DIRS = ["masks", "palette", "rgb_and_mask", "contour_gt_masks", "bbox_overlay"]

def masks_to_bboxes(mask_arr):
    """
    Get bounding boxes (xyxy) for each instance in a mask using NumPy only.

    Args:
        mask_arr (np.ndarray): 2D array (uint8), each unique ID is an instance.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (x1, y1, x2, y2)
    """
    bboxes = []
    instance_ids = np.unique(mask_arr)
    instance_ids = instance_ids[instance_ids != 0]  # Exclude background

    for instance_id in instance_ids:
        ys, xs = np.where(mask_arr == instance_id)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bboxes.append((x1, y1, x2, y2))
    return bboxes

def plot_bboxes(image, bboxes, title="Image with Bounding Boxes"):
    """
    Plot an image with bounding boxes overlaid.

    Args:
        image (PIL.Image): RGB image to display.
        bboxes (List[Tuple[int, int, int, int]]): List of bboxes in (x1, y1, x2, y2) format.
        title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

def main(args):
    """
    Read faulty frames from iter1_faulty.txt, copy RGB images to rgb1/,
    load first frame mask (sam2/masks/00000.png), convert to bboxes,
    propagate through faulty frames using SAM2, and plot bboxes for confirmation.

    Args:
        args: Command-line arguments with scene_dir and iter.
    """
    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    sam2_dir = scene_dir / "sam2"
    mask_dir = sam2_dir / "masks"
    faulty_file = sam2_dir / "iter1_faulty.txt"
    mask_refine_out = sam2_dir / "mask_refinement" / f"iter_{args.iter}"
    
    try:
        shutil.rmtree(mask_refine_out)
    except FileNotFoundError:
        pass

    rgb1_dir = mask_refine_out / "rgb"

    # Create rgb1 directory
    rgb1_dir.mkdir(parents=True, exist_ok=True)

    # Read faulty frames
    if not faulty_file.exists():
        logger.error(f"faulty file not found: {faulty_file}")
        return
    faulty_frames = [line.strip() for line in open(faulty_file) if line.strip()]
    logger.info(f"found {len(faulty_frames)} faulty frames.")

    # Copy RGB PNG images to rgb1 as JPGs, including 00000.png
    rgb_files_to_copy = [f"{fname}.png" for fname in faulty_frames] + ["00000.png"]
    for fname in rgb_files_to_copy:
        src_path = rgb_dir / fname
        dst_fname = fname.replace(".png", ".jpg")
        dst_path = rgb1_dir / dst_fname
        if src_path.exists():
            img = PILImg.open(src_path).convert("RGB")
            img.save(dst_path, format="JPEG", quality=100)
            logger.info(f"Converted and copied {src_path} to {dst_path} as JPG")
        else:
            logger.warning(f"RGB file not found: {src_path}")

    # Load first frame mask (00000.png)
    first_mask_path = mask_dir / "00000.png"
    if not first_mask_path.exists():
        logger.error(f"first frame mask not found: {first_mask_path}")
        return
    first_mask_img = PILImg.open(first_mask_path)
    first_mask_arr = np.array(first_mask_img).astype(np.uint8)
    seed_ids = np.unique(first_mask_arr)
    seed_ids = seed_ids[seed_ids != 0]  # keep objects only

    def bbox_from_id(mask_arr, idv):
        ys, xs = np.where(mask_arr == idv)
        return (xs.min(), ys.min(), xs.max(), ys.max())

    # bboxes in the SAME order as seed_ids
    initial_bboxes = [bbox_from_id(first_mask_arr, idv) for idv in seed_ids]
    logger.info(f"generated {len(initial_bboxes)} bboxes from first frame mask")

    # Plot initial bboxes on first frame RGB image
    first_rgb_path = rgb_dir / "00000.png"
    if first_rgb_path.exists():
        first_rgb_img = PILImg.open(first_rgb_path).convert("RGB")
        # plot_bboxes(first_rgb_img, initial_bboxes, title="Initial Bounding Boxes (Frame 00000.png)")
    else:
        logger.warning(f"first frame rgb image not found: {first_rgb_path}")

    # Initialize SAM2 model
    sam2 = SAM2Predictor()

    # Create output directories
    sam2_out = mask_refine_out / "sam2"
    for subdir in OUTPUT_DIRS:
        (sam2_out / subdir).mkdir(parents=True, exist_ok=True)

    # Load palette
    palette = load_palette()
    video_dir = str(rgb1_dir)

    with torch.inference_mode(), torch.autocast(sam2.device):
        segments = sam2.propagate_bbox_prompt_masks_and_save(
            video_dir, initial_bboxes
        )
    
    for i, (fname, masks) in tqdm(
        enumerate(reversed(list(segments.items()))), desc="Processing frames"
    ):
        frame = PILImg.open(rgb1_dir / fname).convert("RGB")
        masks_tensor = torch.tensor(masks)
        masks_arr = np.asarray(masks)
        if masks_arr.ndim == 4:
            masks_arr = masks_arr.squeeze(1)  # [K,H,W]
        masks_arr = masks_arr.astype(bool)

        # Paint label map using ORIGINAL ids, assuming mask order == seed_ids order
        H, W = masks_arr.shape[1], masks_arr.shape[2]
        combined_mask = np.zeros((H, W), dtype=np.uint16)
        for k, idv in enumerate(seed_ids):
            combined_mask[masks_arr[k]] = np.uint16(idv)
        name = fname.replace(".jpg", ".png")
        PILImg.fromarray(combined_mask.astype(np.uint16)).save(
            sam2_out / "masks" / name
        )

        palette_img = apply_palette(torch.from_numpy(combined_mask.astype(np.int64)), palette)
        palette_img.save(sam2_out / "palette" / name)
        blended = merge_rgb_with_mask(frame, palette_img)
        blended.save(sam2_out / "rgb_and_mask" / name)
        contour_img = merge_rgb_with_mask_with_contours(frame, palette_img)
        contour_img.save(sam2_out / "contour_gt_masks" / name)
        # if i == 0:
        #     conf = [1.0] * len(initial_bboxes)
        #     phrases = ["object"] * len(initial_bboxes)
        #     ann_img = annotate(
        #         overlay_masks(f"{i:05d}.png", masks_tensor),
        #         np.array(initial_bboxes),
        #         conf,
        #         phrases,
        #     )
        #     ann_img.save(sam2_out / "bbox_overlay" / name)
    
    shutil.rmtree(rgb1_dir)
    print(f"[🗑️] Deleted JPG directory: {rgb1_dir}")

    logger.info("completed sam2 mask propagation for faulty frames!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy RGB images to rgb1, propagate first frame mask bboxes through faulty frames, and plot bboxes."
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True, help="Scene directory containing rgb/ and sam2/"
    )
    parser.add_argument(
        "--iter", type=int, default=1, help="Iteration for refinement after initial run"
    )
    args = parser.parse_args()
    main(args)
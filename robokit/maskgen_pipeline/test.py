import os
import yaml
import numpy as np
import cv2
from pathlib import Path
import torch
from time import time
import matplotlib.pyplot as plt
from PIL import Image as PILImg
import argparse

# Add parent directory to sys.path to find robokit
import sys
sys.path.append(str(Path(__file__).parent))

# Custom modules (ensure these are available in your environment)
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor
from robokit.utils import overlay_masks, combine_masks, annotate

from .mask_processing import (
    load_palette,
    apply_palette,
    combine_masks,
    merge_rgb_with_mask,
    merge_rgb_with_mask_with_contours,
)

# Set the correct RDD module path relative to the script
rdd_path = Path(__file__).parent / "rdd"
print(f"Attempting to add RDD path: {rdd_path}")
sys.path.append(str(rdd_path))
print(f"Current sys.path: {sys.path}")

from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper

def points_to_bbox(points):
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    return [x_min, y_min, x_max, y_max]

def apply_mask(img, mask):
    """Apply a binary mask to an image."""
    return cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8) * 255)

def masks_to_bboxes_numpy(mask_arr):
    """
    Get bounding boxes (xyxy) for each instance in a mask using NumPy only.
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

def filter_correspondences(pts0, pts1, conf, mask0_bin, mask1_bin, conf_threshold=0.9):
    """
    Filter correspondences to keep only those within masks and above confidence threshold.
    """
    valid = []
    for i in range(len(pts0)):
        x0, y0 = int(pts0[i, 0]), int(pts0[i, 1])
        x1, y1 = int(pts1[i, 0]), int(pts1[i, 1])
        if (0 <= y0 < mask0_bin.shape[0] and 0 <= x0 < mask0_bin.shape[1] and
            0 <= y1 < mask1_bin.shape[0] and 0 <= x1 < mask1_bin.shape[1]):
            if (mask0_bin[y0, x0] > 0 and mask1_bin[y1, x1] > 0 and conf[i] > conf_threshold):
                valid.append(i)
    return pts0[valid], pts1[valid], conf[valid]

def plot_correspondences(img0, img1, pts0, pts1, conf, title0="Image 0", title1="Image 1", num_lines=50):
    """
    Plot two images side by side with lines showing correspondences.
    """
    img0_rgb = img0
    img1_rgb = img1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img0_rgb)
    ax1.set_title(title0)
    ax1.axis('off')
    ax2.imshow(img1_rgb)
    ax2.set_title(title1)
    ax2.axis('off')

    if len(pts0) > num_lines:
        indices = np.argsort(conf)[-num_lines:]
        pts0 = pts0[indices]
        pts1 = pts1[indices]

    for i in range(len(pts0)):
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        ax1.plot(x0, y0, 'ro', markersize=5)
        ax2.plot(x1, y1, 'ro', markersize=5)
        con = plt.Line2D(
            (x0, x1 + img0.shape[1]),
            (y0, y1),
            transform=fig.transFigure,
            color='cyan',
            linewidth=1,
            alpha=0.5
        )
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(args):
    """
    Main function to reanalyze faulty masks and estimate correspondences between frame pairs.
    """
    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    sam2_dir = scene_dir / "sam2"
    mask_dir = sam2_dir / "masks"
    depth_dir = scene_dir / "depth"
    faulty_file = sam2_dir / "iter1_faulty.txt"

    # Initialize models
    SAM2 = SAM2Predictor()

    # Load config file
    config_path = Path(__file__).parent / "rdd" / "configs" / "default.yaml"
    print(f"Using config path: {config_path}")

    # Set weights path
    weight_path = Path(os.getcwd()) / "ckpts" / "rdd" / "RDD-v2.pth"
    print(f"Using weights path: {weight_path}")
    
    config = read_config(str(config_path))

    RDD_model = build(config=config, weights=weight_path)
    RDD_model.eval()
    RDD_wrap = RDD_helper(RDD_model)

    # Read faulty frames
    if not faulty_file.exists():
        print(f"[❌] Faulty file not found: {faulty_file}")
        return
    faulty_frames = [line.strip() for line in open(faulty_file) if line.strip()]
    print(f"[🔍] Found {len(faulty_frames)} faulty frames.")

    sam2_out = Path(scene_dir) / f"sam2_{args.iter+1}"
    # Create subdirectories
    (sam2_out / "masks").mkdir(parents=True, exist_ok=True)
    (sam2_out / "palette").mkdir(parents=True, exist_ok=True)
    (sam2_out / "rgb_and_mask").mkdir(parents=True, exist_ok=True)
    (sam2_out / "contour_gt_masks").mkdir(parents=True, exist_ok=True)

    is_first = True
    new_masks_combined = None

    for fname in faulty_frames:
        # Load faulty frame (first image)
        frame_idx = int(fname)
        # name0 = f"00000.png"  # Assume previous frame is -1
        name0 = f"{frame_idx - 1:05d}.png"  # Assume previous frame is -1
        rgb_path0 = rgb_dir / name0
        mask_path0 = mask_dir / name0
        depth_path0 = depth_dir / name0
        old_rgb_mask_overlay_path = sam2_dir / "rgb_and_mask" / name0

        name1 = f"{fname}.png"
        rgb_path1 = rgb_dir / name1
        mask_path1 = mask_dir / name1
        depth_path1 = depth_dir / name1

        # Check if files exist
        if not all(p.exists() for p in [rgb_path0, mask_path0, depth_path0, rgb_path1, mask_path1, depth_path1]):
            print(f"[⚠️] Missing files for frame pair {name0} or {name1}")
            continue

        # Load images and masks
        rgb_img0 = PILImg.open(rgb_path0).convert("RGB")
        img0 = np.array(rgb_img0).astype(np.uint8)
        mask_img0 = PILImg.open(mask_path0)
        mask_arr0 = np.array(mask_img0).astype(np.uint8)

        rgb_img1 = PILImg.open(rgb_path1).convert("RGB")
        img1 = np.array(rgb_img1).astype(np.uint8)
        mask_img1 = PILImg.open(mask_path1)
        mask_arr1 = np.array(mask_img1).astype(np.uint8)

        bbox_prompts = []

        palette = load_palette()

        if is_first:
            # For the first frame, use the original mask as the new overlay
            new_masks_combined = mask_arr0.copy()
            is_first = False

        obj_idxs = np.unique(new_masks_combined)[1:]  # Exclude background (0)
        
        predicted_masks = []

        # Process correspondences for each instance in the new mask
        for i in obj_idxs:
            if isinstance(new_masks_combined, torch.Tensor):
                new_masks_combined = new_masks_combined.numpy()
            mask0_bin = (new_masks_combined == i).astype(np.uint8)
            mask1_bin = (mask_arr1 == i).astype(np.uint8)  # Use original mask for second image
            img0_masked = apply_mask(img0, mask0_bin)
            img1_masked = apply_mask(img1, mask1_bin)

            # Compute correspondences
            start = time()
            mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
            print(f"Found {len(mkpts_0)} matches for instance {i} in {time() - start:.2f} seconds")

            # bbox_prompts.append(points_to_bbox(mkpts_1))
            
            # masks, scores, logits = SAM2.predict_mask_in_image_using_point_prompts(
            #         rgb_img1,
            #         mkpts_1,
            #         np.ones(len(mkpts_1), dtype=np.int64) * i,
            # )

            # mean_xy = np.array([np.mean(mkpts_1, axis=0)])

            # Filter correspondences based on mask and confidence
            masks, scores, logits = SAM2.predict_mask_in_image_using_point_prompts(
                    rgb_img1,
                    mean_xy.astype(np.int16),
                    np.ones(len(mean_xy), dtype=np.int64) * i,
            )

            predicted_masks.append(masks)
        
        new_masks = np.array(predicted_masks)

        # Generate new masks with SAM2
        # new_masks, scores, logits = SAM2.predict_mask_in_image(rgb_img1, np.array(bbox_prompts))

        new_masks = torch.tensor(new_masks).unsqueeze(0) if len(new_masks.shape) < 4 else torch.tensor(new_masks)
        combined_mask = combine_masks(new_masks[:, 0, :, :])

        name = f"{fname}.png"
        # Save combined mask
        PILImg.fromarray(combined_mask.cpu().numpy().astype(np.uint16)).save(sam2_out / "masks" / name)

        # Save palette, blended, and contour images
        palette_img = apply_palette(combined_mask, palette)
        palette_img.save(sam2_out / "palette" / name)
        blended = merge_rgb_with_mask(rgb_img1, palette_img)
        blended.save(sam2_out / "rgb_and_mask" / name)
        contour_img = merge_rgb_with_mask_with_contours(rgb_img1, palette_img)
        contour_img.save(sam2_out / "contour_gt_masks" / name)

        # Prepare annotations for visualization
        old_overlay = PILImg.open(old_rgb_mask_overlay_path).convert("RGB") if old_rgb_mask_overlay_path.exists() else None
        new_masks_combined = combine_masks(new_masks[:, 0, :, :])  # Ensure correct shape
        new_overlay = overlay_masks(rgb_img1, new_masks_combined)

        # Plot mask comparison
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # if old_overlay is not None:
        #     axes[0].imshow(old_overlay)
        #     axes[0].set_title(f"Old Mask - {name0}")
        # else:
        #     axes[0].set_title(f"Old Mask Not Found - {name0}")
        # axes[1].imshow(new_overlay)
        # axes[1].set_title(f"New Mask (SAM2) - {name}")
        # for ax in axes:
        #     ax.axis("off")
        # plt.tight_layout()
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reanalyze faulty masks and estimate correspondences.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--refine", action="store_true", help="Enable refinement mode")
    parser.add_argument("--iter", type=int, default=1, help="Iteration for refinement after initial run")
    args = parser.parse_args()
    main(args)

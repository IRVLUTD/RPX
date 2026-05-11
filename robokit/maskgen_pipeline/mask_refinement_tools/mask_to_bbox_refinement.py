#!/usr/bin/env python3
"""
Script: regenerate_masks_from_bboxes.py

Given a directory of input RGB images and corresponding binary masks,
this script computes tight bounding boxes around each mask, then
generates new, refined masks for those boxes using the SAM2Predictor.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
from robokit.perception import SAM2Predictor, mask_to_bbox
from robokit.utils import overlay_masks

def process_image(image_path: Path, mask_dir: Path, output_dir: Path):
    """
    For a single image, load all masks in mask_dir, compute their bboxes,
    predict new masks with SAM2, and save outputs to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    sam = SAM2Predictor()

    # Iterate through each binary mask file
    for mask_file in mask_dir.glob("*.png"):
        mask = np.array(Image.open(mask_file))
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue  # skip if mask was empty

        # Predict a new mask using the detected bbox
        # SAM2Predictor.predict_with_bbox returns a list of mask arrays
        new_masks = sam.predict_with_bbox(img, [bbox])
        new_mask = new_masks[0]

        # Save the new mask (binary PNG)
        out_mask_path = output_dir / mask_file.name
        Image.fromarray((new_mask > 0).astype(np.uint8) * 255).save(out_mask_path)

        # Optionally, save an overlay for quick visual inspection
        overlay = overlay_masks(img, [new_mask])
        overlay.save(output_dir / f"ovr_{mask_file.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Regenerate masks by computing bboxes from existing masks and re-predicting"
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True,
        help="Root folder containing `rgb/` and `sam2/masks/`"
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    old_mask_dir = scene_dir / "sam2" / "masks"
    new_mask_dir = scene_dir / "sam2_new_masks"

    # Process each image in the rgb directory
    for img_file in rgb_dir.glob("*.png"):
        process_image(img_file, old_mask_dir, new_mask_dir)

    print(f"🎉 New masks and overlays saved to {new_mask_dir}")

if __name__ == "__main__":
    main()

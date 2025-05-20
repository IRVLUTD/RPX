# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + GPT-4)
# ----------------------------------------------------------------------------------------------------
import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor
from robokit.utils import annotate, overlay_masks
from .config import OUTPUT_DIRS, logger
from .image_processing import convert_pngs_to_jpgs_reversed, reverse_all_modalities
from .mask_processing import (
    load_palette,
    apply_palette,
    combine_masks,
    merge_rgb_with_mask,
    merge_rgb_with_mask_with_contours,
)
from .bbox_utils import (
    apply_nms,
    sort_boxes_by_area,
    remove_n_largest_bboxes,
    interactive_resize_bboxes,
)
from .visualization import interactive_bbox_selection


def main(scene_dir):
    """
    Run the SAM2 reverse mask propagation pipeline.

    Args:
        scene_dir (str): Directory containing rgb/ folder.
    """
    palette = load_palette()
    jpg_dir = convert_pngs_to_jpgs_reversed(scene_dir)
    img_path = sorted(jpg_dir.glob("*.jpg"))[0]
    img_name = img_path.name
    sam2_out = Path(scene_dir) / "sam2"
    bbox_file = sam2_out / "usr_bbox_prompts.npy"
    sam2_out.mkdir(parents=True, exist_ok=True)
    for subdir in OUTPUT_DIRS:
        (sam2_out / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("Initialize object detectors")
    gdino = GroundingDINOObjectPredictor()
    sam2 = SAM2Predictor()
    video_dir = str(jpg_dir)
    img_pil = Image.open(img_path).convert("RGB")
    text_prompt = "objects"
    logger.info("GDINO: Predict initial bounding boxes, phrases, and confidence scores")
    initial_bboxes, phrases, gdino_conf = gdino.predict(img_pil, text_prompt)
    initial_bboxes, kept_indices = apply_nms(
        initial_bboxes, gdino_conf, iou_threshold=0.5
    )
    phrases = [phrases[i] for i in kept_indices]
    gdino_conf = gdino_conf[kept_indices]
    print(f"{len(initial_bboxes)} Initial BBoxes detected!")
    indices_to_keep = [
        i for i, phrase in enumerate(phrases) if "objects" in phrase.lower()
    ]
    phrases = [phrases[i] for i in indices_to_keep]
    initial_bboxes = initial_bboxes[indices_to_keep]
    gdino_conf = gdino_conf[indices_to_keep]
    print(f"{len(initial_bboxes)} Initial BBoxes filtered!")
    n_rm_large_bboxes = 1
    # initial_bboxes, phrases, gdino_conf = remove_n_largest_bboxes(initial_bboxes, phrases, gdino_conf, n_rm_large_bboxes)
    w, h = img_pil.size
    image_pil_bboxes = gdino.bbox_to_scaled_xyxy(initial_bboxes, w, h)
    image_pil_bboxes, phrases, gdino_conf = sort_boxes_by_area(
        image_pil_bboxes, phrases, gdino_conf
    )
    image_pil_bboxes, phrases, gdino_conf = interactive_bbox_selection(
        img_pil, image_pil_bboxes, phrases, gdino_conf
    )
    image_pil_bboxes = interactive_resize_bboxes(img_pil, image_pil_bboxes)
    print(f"[✅] Keeping {len(image_pil_bboxes)} boxes after filtering and additions.")
    bbox_annotated_pil = annotate(
        img_pil, np.array(image_pil_bboxes), gdino_conf, phrases
    )
    bbox_annotated_pil.save(
        os.path.join(sam2_out / "bbox_overlay", img_name.replace("jpg", "png"))
    )
    bbox_annotated_pil.show()
    with torch.inference_mode(), torch.autocast(sam2.device):
        segments = sam2.propagate_bbox_prompt_masks_and_save(
            video_dir, image_pil_bboxes
        )
    for i, (fname, masks) in tqdm(
        enumerate(reversed(list(segments.items()))), desc="Processing frames"
    ):
        frame = Image.open(jpg_dir / fname).convert("RGB")
        masks_tensor = torch.tensor(masks)
        if masks_tensor.ndim == 4:
            masks_tensor = masks_tensor.squeeze(1)
        combined_mask = combine_masks(masks_tensor)
        name = fname.replace(".jpg", ".png")
        Image.fromarray(combined_mask.cpu().numpy().astype(np.uint16)).save(
            sam2_out / "masks" / name
        )

        palette_img = apply_palette(combined_mask, palette)
        palette_img.save(sam2_out / "palette" / name)
        blended = merge_rgb_with_mask(frame, palette_img)
        blended.save(sam2_out / "rgb_and_mask" / name)
        contour_img = merge_rgb_with_mask_with_contours(frame, palette_img)
        contour_img.save(sam2_out / "contour_gt_masks" / name)
        if i == 0:
            conf = [1.0] * len(image_pil_bboxes)
            phrases = ["object"] * len(image_pil_bboxes)
            ann_img = annotate(
                overlay_masks(frame, masks_tensor),
                np.array(image_pil_bboxes),
                conf,
                phrases,
            )
            ann_img.save(sam2_out / "bbox_overlay" / name)
    shutil.rmtree(jpg_dir)
    print(f"[🗑️] Deleted JPG directory: {jpg_dir}")
    reverse_all_modalities(Path(scene_dir))
    print("🎉 Full SAM2 reverse propagation + color + contour pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full SAM2 reverse mask propagation pipeline."
    )
    parser.add_argument(
        "--scene_dir", type=str, required=True, help="Scene directory containing rgb/"
    )
    args = parser.parse_args()
    main(args.scene_dir)

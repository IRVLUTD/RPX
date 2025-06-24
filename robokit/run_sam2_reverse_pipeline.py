#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + GPT-4)
#----------------------------------------------------------------------------------------------------

import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, Image as PILImg
from tqdm import tqdm
import torch
import requests
import matplotlib.pyplot as plt
import cv2

# 🔇 Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import logging
logging.set_verbosity(logging.ERROR)

from robokit.perception import SAM2Predictor
from robokit.utils import annotate, overlay_masks


def convert_pngs_to_jpgs_reversed(input_dir):
    rgb_dir = Path(input_dir) / "rgb"
    jpg_dir = Path(input_dir) / "jpg"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(rgb_dir.glob("*.png"))[::-1]
    for i, file in enumerate(pngs):
        with Image.open(file) as im:
            rgb_img = im.convert("RGB")
            rgb_img.save(jpg_dir / f"{i:05d}.jpg", "JPEG")
    print(f"[✓] Converted {len(pngs)} PNGs to JPGs in reverse order.")
    return jpg_dir


def get_user_bboxes(image_path, save_path):
    image = Image.open(image_path).convert("RGB")
    bboxes = []
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.axis("off")
    start = []

    def on_press(event):
        if event.inaxes is not None:
            start.clear()
            start.extend([event.xdata, event.ydata])

    def on_release(event):
        if event.inaxes is not None:
            x1, y1 = start
            x2, y2 = event.xdata, event.ydata
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bboxes.append([x1, y1, x2, y2])
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', facecolor='none', lw=2))
            fig.canvas.draw()

    def on_key(event):
        if event.key == 'q':
            bboxes_np = np.array(bboxes)
            xywh = np.column_stack([
                bboxes_np[:, 0],
                bboxes_np[:, 1],
                bboxes_np[:, 2] - bboxes_np[:, 0],
                bboxes_np[:, 3] - bboxes_np[:, 1]
            ])
            np.save(save_path.with_suffix('.npy'), {"xyxy": bboxes_np, "xywh": xywh})
            print(f"[✅] Saved both XYXY and XYWH to: {save_path.with_suffix('.npy')}")
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return np.array(bboxes)


def load_palette():
    url = "https://raw.githubusercontent.com/IRVLUTD/fewsol-toolkit/refs/heads/main/palette.txt"
    response = requests.get(url)
    response.raise_for_status()
    return [tuple(map(int, line.strip().split())) for line in response.text.strip().split("\n")]



def apply_palette(mask, palette):
    """
    Apply a color palette to a segmentation mask.

    Args:
        mask (torch.Tensor): Tensor of shape [H, W] containing integer labels.
        palette (list): List of (R, G, B) tuples.

    Returns:
        np.ndarray: RGB image of shape [H, W, 3].
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = torch.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background (assuming background is 0)
        # Ensure label is an integer before indexing the palette
        label_int = int(label.item())  # Convert to int explicitly
        color = palette[label_int % len(palette)]  # Cycle through palette if needed
        colored_mask[mask == label_int] = color

    return PILImg.fromarray(colored_mask)




# def combine_masks(masks_tensor):
#     """Combine multiple binary masks into a single labeled mask."""
#     masks_tensor = torch.flip(masks_tensor, dims=(0,))
#     combined = torch.zeros_like(masks_tensor[0], dtype=torch.uint8)
#     current_id = 1
#     for m in masks_tensor:
#         mask_area = (m > 0) & (combined == 0)
#         combined[mask_area] = current_id
#         current_id += 1
#     return combined

def combine_masks(gt_masks):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [1,2,0].

    Args:
        gt_masks (torch.Tensor): Tensor of shape [N, H, W] representing multiple bit masks.

    Returns:
        torch.Tensor: Combined mask of shape [H, W].
    """
    try:
        gt_masks = torch.flip(gt_masks, dims=(0,))
        num, h, w = gt_masks.shape
        bin_mask = torch.zeros((h, w), device=gt_masks.device)
        num_instance = len(gt_masks)

        # if there is not any instance, just return a mask full of 0s.
        if num_instance == 0:
            return bin_mask

        for m, object_label in zip(gt_masks, range(1, 1 + num_instance)):
            label_pos = torch.nonzero(m, as_tuple=True)
            bin_mask[label_pos] = object_label
        return bin_mask

    except Exception as e:
        logging.error(f"Error combining masks: {e}")
        raise e


def reverse_filenames(target_dir, ext):
    files = sorted(Path(target_dir).glob(f"*.{ext}"))
    # First pass: rename to temporary names
    for i, f in enumerate(files):
        f.rename(f.parent / f"temp_{i:05d}.{ext}")
    # Second pass: reverse and rename to final names
    temp_files = sorted(Path(target_dir).glob(f"temp_*.{ext}"))
    for i, f in enumerate(reversed(temp_files)):
        f.rename(f.parent / f"{i:05d}.{ext}")
    print(f"[✓] Reversed .{ext} filenames in {target_dir}")

def reverse_all_modalities(base_dir):
    base_dir = Path(base_dir)
    targets = {
        "rgb": "png",
        "depth": "png",
        "fisheye/left": "png",
        "fisheye/right": "png",
        "cam_pose": "npz",
    }

    for subdir, ext in targets.items():
        target_path = base_dir / subdir
        if target_path.exists():
            reverse_filenames(target_path, ext)
        else:
            print(f"[!] Skipping {target_path} (does not exist)")


def merge_rgb_with_mask_with_contours(rgb_pil, masks_pil, bg_alpha=0.4, mask_alpha=0.65):
    rgb_array = np.array(rgb_pil.convert("RGB"), dtype=np.float32)
    mask_rgba = np.array(masks_pil.convert("RGBA"), dtype=np.float32)
    mask_rgb = mask_rgba[..., :3]
    mask_binary = (mask_rgb != [0, 0, 0]).any(axis=-1)

    rgb_darkened = rgb_array * (1 - bg_alpha)
    blended = rgb_darkened.copy()
    blended[mask_binary] = (
        mask_rgb[mask_binary] * mask_alpha +
        rgb_darkened[mask_binary] * (1 - mask_alpha)
    )

    gray_mask = np.array(masks_pil.convert("L"))
    for val in np.unique(gray_mask):
        if val == 0:
            continue
        mask = (gray_mask == val).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blended = cv2.drawContours(np.uint8(blended), contours, -1, (255, 255, 255), 2)

    return PILImg.fromarray(np.uint8(np.clip(blended, 0, 255)))


def merge_rgb_with_mask(rgb_pil, masks_pil):
    """
    Merges the RGB image with the mask while applying a semi-transparent black overlay to the background.

    Args:
        rgb_pil (PIL.Image): The RGB image.
        masks_pil (PIL.Image): The mask image with a palette.

    Returns:
        PIL.Image: Merged image with highlighted masks and darkened background.
    """
    # Convert images to RGBA
    rgb_image = rgb_pil.convert("RGB")
    mask_image = masks_pil.convert("RGBA")

    # Convert images to NumPy arrays
    rgb_array = np.array(rgb_image, dtype=np.float32)
    mask_array = np.array(mask_image, dtype=np.float32)

    # Detect non-black regions in the mask
    mask_binary = (mask_array[..., :3] != [0, 0, 0]).any(axis=-1)

    # Darken the background (apply transparency)
    bg_alpha = 0.6  # Transparency level for background
    rgb_darkened = rgb_array * (1 - bg_alpha)

    # Blend the mask with transparency
    mask_alpha = 0.5  # Transparency for the mask colors
    blended_mask = rgb_darkened.copy()
    blended_mask[mask_binary] = (
        mask_array[mask_binary][:, :3] * mask_alpha + rgb_darkened[mask_binary] * (1 - mask_alpha)
    )

    # Convert back to PIL Image
    blended_image = PILImg.fromarray(np.uint8(blended_mask))
    
    return blended_image


    
def main(scene_dir):
    palette = load_palette()
    jpg_dir = convert_pngs_to_jpgs_reversed(scene_dir)
    img_path = sorted(jpg_dir.glob("*.jpg"))[0]

    sam2_out = Path(scene_dir) / "sam2"
    bbox_file = sam2_out / "usr_bbox_prompts.npy"
    sam2_out.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or collect bboxes
    if bbox_file.exists():
        print(f"[ℹ️] Found existing bbox file: {bbox_file}")
        bbox_data = np.load(bbox_file, allow_pickle=True).item()
        bboxes = bbox_data["xyxy"]

        # ✅ Visualize xyxy boxes
        image = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots()
        ax.imshow(image)
        for box in bboxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            ax.add_patch(plt.Rectangle((x1, y1), w, h, edgecolor='lime', facecolor='none', lw=2))
        plt.title("Existing Bounding Boxes (XYXY)")
        plt.axis("off")
        plt.show()

        use_existing = input("Use existing bounding boxes? (y/n): ").strip().lower()
        if use_existing != 'y':
            bboxes = get_user_bboxes(img_path, bbox_file)
    else:
        bboxes = get_user_bboxes(img_path, bbox_file)

    # Step 2: Setup output dirs
    mask_out = sam2_out / "masks"
    rgbm_out = sam2_out / "rgb_and_mask"
    palette_out = sam2_out / "palette"
    overlay_out = sam2_out / "bbox_overlay"
    contour_out = sam2_out / "contour_gt_masks"
    for p in [mask_out, rgbm_out, palette_out, overlay_out, contour_out]:
        p.mkdir(parents=True, exist_ok=True)

    # Step 3: Run SAM2
    sam2 = SAM2Predictor()
    
    with torch.inference_mode(), torch.autocast(sam2.device):
        video_dir = str(jpg_dir)
        segments = sam2.propagate_bbox_prompt_masks_and_save(video_dir, bboxes)

    # Step 4: Save masks + overlays
    for i, (fname, masks) in tqdm(enumerate(reversed(list(segments.items())))):
        frame = Image.open(jpg_dir / fname).convert("RGB")
        masks_tensor = torch.tensor(masks)
        if masks_tensor.ndim == 4:
            masks_tensor = masks_tensor.squeeze(1)

        combined_mask = combine_masks(masks_tensor)

        name = fname.replace(".jpg", ".png")
        Image.fromarray(combined_mask.cpu().numpy(), mode="L").save(mask_out / name)

        palette_img = apply_palette(combined_mask, palette)
        palette_img.save(palette_out / name)

        # RGB + mask blend
        frame_np = np.array(frame.convert("RGB"), dtype=np.float32)
        mask_np = np.array(palette_img.convert("RGB"), dtype=np.float32)
        mask_binary = (mask_np != [0, 0, 0]).any(axis=-1)
        frame_np[mask_binary] = 0.6 * frame_np[mask_binary] + 0.4 * mask_np[mask_binary]
        blended = PILImg.fromarray(np.uint8(np.clip(frame_np, 0, 255)))
        blended.save(rgbm_out / name)

        # Contour overlays
        contour_img = merge_rgb_with_mask_with_contours(frame, palette_img)
        contour_img.save(contour_out / name)

        if i == 0:
            conf = [1.0] * len(bboxes)
            phrases = ["object"] * len(bboxes)
            ann_img = annotate(overlay_masks(frame, masks_tensor), bboxes, conf, phrases)
            ann_img.save(overlay_out / name)


    shutil.rmtree(jpg_dir)
    print(f"[🗑️] Deleted JPG directory: {jpg_dir}")
    reverse_all_modalities(Path(scene_dir))
    print("🎉 Full SAM2 reverse propagation + color + contour pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full SAM2 reverse mask propagation pipeline.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Scene directory containing rgb/")
    args = parser.parse_args()
    main(args.scene_dir)

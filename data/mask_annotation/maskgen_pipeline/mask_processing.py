#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + GPT-4)
#----------------------------------------------------------------------------------------------------
import numpy as np
import torch
import cv2
from PIL import Image
import requests

def load_palette():
    """
    Load color palette from remote URL.
    
    Returns:
        list: List of (R, G, B) tuples.
    """
    url = "https://raw.githubusercontent.com/IRVLUTD/fewsol-toolkit/refs/heads/main/palette.txt"
    response = requests.get(url)
    response.raise_for_status()
    return [tuple(map(int, line.strip().split())) for line in response.text.strip().split("\n")]

def apply_palette(mask, palette):
    """
    Apply a color palette to a segmentation mask.
    
    Args:
        mask (torch.Tensor): Tensor of shape [H, W] with integer labels.
        palette (list): List of (R, G, B) tuples.
    
    Returns:
        Image: RGB image of shape [H, W, 3].
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = torch.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue
        label_int = int(label.item())
        color = palette[label_int % len(palette)]
        colored_mask[mask == label_int] = color
    return Image.fromarray(colored_mask)

def combine_masks(gt_masks):
    """
    Combine multiple bit masks [N, H, W] into a single mask [H, W].
    
    Args:
        gt_masks (torch.Tensor): Tensor of shape [N, H, W].
    
    Returns:
        torch.Tensor: Combined mask of shape [H, W].
    """
    try:
        gt_masks = torch.flip(gt_masks, dims=(0,))
        num, h, w = gt_masks.shape
        bin_mask = torch.zeros((h, w), device=gt_masks.device)
        if num == 0:
            return bin_mask
        for m, object_label in zip(gt_masks, range(1, 1 + num)):
            label_pos = torch.nonzero(m, as_tuple=True)
            bin_mask[label_pos] = object_label
        return bin_mask
    except Exception as e:
        print(f"[ERROR] Combining masks: {e}")
        raise

def merge_rgb_with_mask_with_contours(rgb_pil, masks_pil, bg_alpha=0.4, mask_alpha=0.65):
    """
    Merge RGB image with mask, adding contours.
    
    Args:
        rgb_pil (Image): RGB image.
        masks_pil (Image): Mask image with palette.
        bg_alpha (float): Background transparency.
        mask_alpha (float): Mask transparency.
    
    Returns:
        Image: Merged image with contours.
    """
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
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))

def merge_rgb_with_mask(rgb_pil, masks_pil, bg_alpha=0.6, mask_alpha=0.5):
    """
    Merge RGB image with mask, applying a semi-transparent black overlay to the background.
    
    Args:
        rgb_pil (Image): RGB image.
        masks_pil (Image): Mask image with palette.
        bg_alpha (float): Background transparency.
        mask_alpha (float): Mask transparency.
    
    Returns:
        Image: Merged image.
    """
    rgb_array = np.array(rgb_pil.convert("RGB"), dtype=np.float32)
    mask_array = np.array(masks_pil.convert("RGBA"), dtype=np.float32)
    mask_binary = (mask_array[..., :3] != [0, 0, 0]).any(axis=-1)
    rgb_darkened = rgb_array * (1 - bg_alpha)
    blended_mask = rgb_darkened.copy()
    blended_mask[mask_binary] = (
        mask_array[mask_binary][:, :3] * mask_alpha + rgb_darkened[mask_binary] * (1 - mask_alpha)
    )
    return Image.fromarray(np.uint8(blended_mask))
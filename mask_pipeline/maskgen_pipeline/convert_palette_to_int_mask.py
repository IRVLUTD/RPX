import numpy as np
import argparse
from PIL import Image
import requests
from pathlib import Path
from tqdm import tqdm
from mask_processing import load_palette


def convert_palette_mask_to_int(mask_img, palette):
    """
    Convert a color-coded RGB mask to an integer label mask.
    
    Label 0 → background (palette[0])
    Label 1, 2, ... → objects (palette[1:], in order)

    Args:
        mask_img (PIL.Image or np.ndarray): RGB mask image of shape (H, W, 3)
        palette (np.ndarray): (N, 3) array of RGB tuples

    Returns:
        np.ndarray: (H, W) integer mask with values in [0, N-1]
    """
    # Ensure NumPy array
    if not isinstance(mask_img, np.ndarray):
        mask_img = np.array(mask_img, dtype=np.uint8)
    
    h, w, _ = mask_img.shape
    flat_rgb = mask_img.reshape(-1, 3)

    # Convert each RGB triplet to 24-bit unique ID
    flat_ids = (flat_rgb[:, 0].astype(np.uint32) << 16) | \
               (flat_rgb[:, 1].astype(np.uint32) << 8) | \
               flat_rgb[:, 2].astype(np.uint32)

    # Build LUT for RGB → label index (0-based)
    lut = np.zeros(2**24, dtype=np.uint16)
    palette_ids = (palette[:, 0].astype(np.uint32) << 16) | \
                  (palette[:, 1].astype(np.uint32) << 8) | \
                  palette[:, 2].astype(np.uint32)
    lut[palette_ids] = np.arange(len(palette), dtype=np.uint16)

    # Apply LUT to mask
    label_mask = lut[flat_ids].reshape(h, w)
    max_id = np.max(label_mask)
    _len = np.unique(label_mask).size

    if max_id != _len-1:
        label_mask[label_mask==max_id] = _len-1

    return label_mask

def main(scene_dir):
    palette = np.array(load_palette(), dtype=np.uint8)
    palette_dir = Path(scene_dir) / "sam2" / "palette"
    output_dir = Path(scene_dir) / "sam2" / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(palette_dir.glob("*.png"))
    for mask_file in tqdm(mask_files, desc="Converting masks"):
        mask_img = Image.open(mask_file).convert("RGB")
        int_mask = convert_palette_mask_to_int(mask_img, palette)
        Image.fromarray(int_mask).save(output_dir / mask_file.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast palette-to-integer mask conversion.")
    parser.add_argument("--scene_dir", required=True, help="Path to scene directory")
    args = parser.parse_args()
    main(args.scene_dir)
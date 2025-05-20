import numpy as np
import argparse
from PIL import Image
import requests
from pathlib import Path
from tqdm import tqdm
from .mask_processing import load_palette


def convert_palette_mask_to_int(mask_img, palette):
    """
    Convert an RGB mask to an int mask using a fast LUT-based vectorized approach.
    """
    rgb = np.array(mask_img, dtype=np.uint8)
    h, w, _ = rgb.shape
    flat_rgb = rgb.reshape(-1, 3)

    # Encode RGB to unique 24-bit integer
    flat_ids = (flat_rgb[:, 0].astype(np.uint32) << 16) | \
               (flat_rgb[:, 1].astype(np.uint32) << 8) | \
               flat_rgb[:, 2].astype(np.uint32)

    # Build LUT: 2^24 possible RGB values (16M) → label index
    lut_size = 2**24
    lut = np.zeros(lut_size, dtype=np.uint16)
    palette_ids = (palette[:, 0].astype(np.uint32) << 16) | \
                  (palette[:, 1].astype(np.uint32) << 8) | \
                  palette[:, 2].astype(np.uint32)
    lut[palette_ids] = np.arange(1, len(palette) + 1, dtype=np.uint16)

    # Map pixel IDs to labels using LUT
    flat_labels = lut[flat_ids]
    return flat_labels.reshape(h, w)

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
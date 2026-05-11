# ----------------------------------------------------------------------------------------------------
# Mask to Object Validator (Skips Faulty Frames)
# ----------------------------------------------------------------------------------------------------
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def overlay_labels(rgb_img, palette_img, mask_map):
    """Overlay object names at the centroid of each mask."""
    img = np.array(rgb_img.convert("RGB"))
    mask_arr = np.array(palette_img.convert("RGB"))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")

    for entry in mask_map.values():
        color = tuple(entry["color"])
        obj_name = entry["object"]["name"]
        obj_id = entry["object"]["id"]

        mask = np.all(mask_arr == np.array(color), axis=-1)
        if not mask.any():
            continue
        y, x = np.mean(np.argwhere(mask), axis=0)

        ax.text(x, y, f"{obj_name} ({obj_id})",
                color="yellow", fontsize=10, weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))

    fig.suptitle("Mask Validation (Valid Frame)", fontsize=14, color="cyan")
    plt.show()


def validate_mask_to_object(iter_dir):
    iter_dir = Path(iter_dir)
    mapping_path = iter_dir / "sam2" / "mask_to_object.json"
    palette_dir = iter_dir / "sam2" / "palette"
    rgb_dir = iter_dir / "sam2" / "rgb_and_mask"
    faulty_file = iter_dir / "sam2" / "iter1_faulty.txt"

    if not mapping_path.exists():
        raise FileNotFoundError(f"mask_to_object.json not found at {mapping_path}")

    with open(mapping_path, "r") as f:
        mask_map = json.load(f)

    palette_files = sorted(palette_dir.glob("*.png"))
    rgb_files = sorted(rgb_dir.glob("*.png"))

    if not palette_files:
        raise RuntimeError("No palette frames found to validate.")

    # Load faulty frame names (without extension)
    faulty_frames = set()
    if faulty_file.exists():
        with open(faulty_file, "r") as f:
            faulty_frames = {line.strip() for line in f if line.strip()}

    # Filter out faulty frames
    valid_indices = [
        i for i, pf in enumerate(palette_files)
        if pf.stem not in faulty_frames
    ]

    if not valid_indices:
        raise RuntimeError("No valid frames available (all marked faulty).")

    # Pick a random valid frame
    idx = random.choice(valid_indices)
    palette_img = Image.open(palette_files[idx])
    rgb_img = Image.open(rgb_files[idx])

    print(f"[INFO] Showing validation for valid frame: {palette_files[idx].name}")
    overlay_labels(rgb_img, palette_img, mask_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate mask_to_object.json on a random non-faulty frame.")
    parser.add_argument("--iter_dir", required=True, help="Path to iteration folder (e.g., scene63.ecsw.axxess-atrium/0)")
    args = parser.parse_args()

    validate_mask_to_object(args.iter_dir)

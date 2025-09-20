#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + GPT-4)
#----------------------------------------------------------------------------------------------------
from pathlib import Path
from PIL import Image
import os

def convert_pngs_to_jpgs_reversed(input_dir):
    """
    Convert PNG images in rgb/ to JPG in jpg/ in reverse order.
    
    Args:
        input_dir (str or Path): Directory containing rgb/ folder.
    
    Returns:
        Path: Path to jpg/ directory.
    """
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

def reverse_filenames(target_dir, ext):
    """
    Reverse the order of filenames with the given extension in the target directory.
    
    Args:
        target_dir (str or Path): Directory containing files.
        ext (str): File extension (e.g., 'png', 'npz').
    """
    files = sorted(Path(target_dir).glob(f"*.{ext}"))
    for i, f in enumerate(files):
        f.rename(f.parent / f"temp_{i:05d}.{ext}")
    temp_files = sorted(Path(target_dir).glob(f"temp_*.{ext}"))
    for i, f in enumerate(reversed(temp_files)):
        f.rename(f.parent / f"{i:05d}.{ext}")
    print(f"[✓] Reversed .{ext} filenames in {target_dir}")

def reverse_all_modalities(base_dir):
    """
    Reverse filenames for all modalities in the base directory.
    
    Args:
        base_dir (str or Path): Base directory containing modality folders.
    """
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
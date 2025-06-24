#!/usr/bin/env python3
"""
reverse_rgb_filenames.py

Safely reverses the order of .png filenames in a specified rgb/ directory.

Example:
    python reverse_rgb_filenames.py --rgb_dir /path/to/rgb
"""

import os
import argparse

def reverse_filenames(rgb_dir):
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

    if not rgb_files:
        print("[ERROR] No .png files found in:", rgb_dir)
        return

    # Step 1: Temporary renaming to avoid overwrite
    for i, f in enumerate(rgb_files):
        old_path = os.path.join(rgb_dir, f)
        temp_path = os.path.join(rgb_dir, f"temp_{i:6d}.png")
        os.rename(old_path, temp_path)

    # Step 2: Final renaming in reversed order
    temp_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith("temp_")])
    for i, f in enumerate(reversed(temp_files)):
        new_name = f"{i:04d}.png"
        os.rename(os.path.join(rgb_dir, f), os.path.join(rgb_dir, new_name))

    print(f"[✅] Reversed {len(rgb_files)} .png filenames in: {rgb_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse .png filenames in a directory.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Path to the rgb/ directory.")
    args = parser.parse_args()

    if not os.path.isdir(args.rgb_dir):
        print("[ERROR] Invalid directory:", args.rgb_dir)
    else:
        reverse_filenames(args.rgb_dir)

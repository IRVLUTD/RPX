import os
import yaml
import numpy as np
import cv2
from pathlib import Path
import torch
from time import time
import matplotlib.pyplot as plt
from PIL import Image as PILImg

# Add parent directory to sys.path to find robokit
import sys
import pickle
sys.path.append(str(Path(__file__).parent))


# Custom modules (ensure these are available in your environment)
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor
from robokit.utils import overlay_masks, combine_masks, annotate

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


if __name__ == "__main__":
    # Load the data
    if not os.path.exists("test.pkl"):
        raise FileNotFoundError("test.pkl file not found. Please ensure it exists in the current directory.")
    with open("test.pkl", "rb") as f:
        data = pickle.load(f)
        point_prompts = data["points"]
        point_labels = data["labels"]
        img = data["img"]

    # Initialize SAM2 predictor
    sam2 = SAM2Predictor()
    # predictor = sam2.img_predictor

    bbox_prompts = []

    with torch.inference_mode():
        for obj_idx, obj_pts_prompts in enumerate(point_prompts):
            # pts_labels = point_labels[obj_idx]
            # obj_pts_prompts = obj_pts_prompts#.astype(np.int)
            
            bbox = points_to_bbox(obj_pts_prompts)
            bbox_prompts.append(bbox)

        # predictor.set_image(img)
        new_masks, scores, logits = sam2.predict_mask_in_image(img, np.array(bbox_prompts))

        new_masks = torch.tensor(new_masks).unsqueeze(0) if len(new_masks.shape) < 4 else torch.tensor(new_masks)
        new_masks_combined = combine_masks(new_masks[:, 0, :, :])

        # Plot mask
        plt.figure(figsize=(8, 6))
        plt.imshow(new_masks_combined, cmap='gray')

        # Final touches
        plt.axis('off')
        plt.title("Mask + Points + Bounding Box")
        plt.show()

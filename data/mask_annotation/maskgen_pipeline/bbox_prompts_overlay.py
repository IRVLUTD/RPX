#!/usr/bin/env python3

import os
import json
import argparse
from PIL import Image as PILImg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths, e.g. <scene>/0/rgb/00249.png and <scene>/0/bbox_prompts.json
parser = argparse.ArgumentParser(description="Draw saved bbox prompts on an image.")
parser.add_argument("image", help="RGB frame to annotate")
parser.add_argument("bboxes", help="bbox_prompts.json written by collect_bbox_prompts.py")
parser.add_argument("--output", help="output image (default: annotated.png next to the bbox JSON)")
args = parser.parse_args()
image_path = args.image
bbox_json_path = args.bboxes
output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(bbox_json_path)), "annotated.png")

# Load image and bboxes
image = PILImg.open(image_path)
with open(bbox_json_path, 'r') as f:
    bboxes = json.load(f)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)
ax.axis('off')

# Draw each box
for box in bboxes:
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

plt.tight_layout()
# Save the annotated image
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print(f"Annotated image saved to: {output_path}")
plt.show()

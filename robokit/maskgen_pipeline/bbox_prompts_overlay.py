#!/usr/bin/env python3

import os
import json
from PIL import Image as PILImg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
image_path = "/home/itaykadosh/Desktop/scene97.jsom.atrium/0/rgb/00249.png"
bbox_json_path = "/home/itaykadosh/Desktop/scene97.jsom.atrium/0/bbox_prompts.json"
output_path = "/home/itaykadosh/Desktop/scene97.jsom.atrium/0/annotated.png"

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

#!/usr/bin/env python3

import os
import json
from PIL import Image as PILImg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path to your image and output JSON
image_path = "/home/itaykadosh/Desktop/scene97.jsom.atrium/0/rgb/00249.png"
output_json = "/home/itaykadosh/Desktop/scene97.jsom.atrium/0/bbox_prompts.json"

# Load the image
image = PILImg.open(image_path)

# Create figure & axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)
ax.axis('off')

# Temporary storage
temp_points = []   # holds up to two clicks
bboxes = []        # final list of {x1,y1,x2,y2}

def on_click(event):
    # only respond to left mouse button inside axes
    if event.inaxes is not ax or event.button != 1:
        return
    x, y = int(event.xdata), int(event.ydata)
    temp_points.append((x, y))

    if len(temp_points) == 2:
        # compute top-left/bottom-right
        (x1, y1), (x2, y2) = temp_points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        # draw rectangle
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()
        temp_points.clear()

def on_key(event):
    # press 'q' to finish
    if event.key.lower() == 'q':
        plt.close(fig)

# connect events
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

print("Click top-left and bottom-right of each box. Press 'q' when done.")
plt.show()

# ensure output directory exists
os.makedirs(os.path.dirname(output_json), exist_ok=True)
# save to JSON
with open(output_json, 'w') as f:
    json.dump(bboxes, f, indent=2)

print(f"Saved {len(bboxes)} bounding box(es) to {output_json}")
for i, box in enumerate(bboxes, start=1):
    print(f"{i}: x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}")

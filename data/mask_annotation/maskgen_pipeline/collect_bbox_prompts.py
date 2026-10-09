#!/usr/bin/env python3

import os
import json
import argparse
from PIL import Image as PILImg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Image to annotate (e.g. <scene>/0/rgb/00249.png) and output JSON
parser = argparse.ArgumentParser(description="Click bbox prompts on an image and save them as JSON.")
parser.add_argument("image", help="RGB frame to annotate")
parser.add_argument("--output", help="output JSON (default: bbox_prompts.json in the phase directory, i.e. two levels above the image)")
args = parser.parse_args()
image_path = args.image
output_json = args.output or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(image_path))), "bbox_prompts.json")

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

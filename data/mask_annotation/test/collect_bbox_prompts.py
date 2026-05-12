# imports
import numpy as np
from PIL import Image as PILImg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image using PIL
# image_path = "./imgs/sam2-test/rgb/000000.jpg"  # Replace with the path to your image
image_path = "/home/jishnu/Projects/iTeach-UOIS/uois-models/UnseenObjectsWithMeanShift/data/humanplay_data/scene_0424T194846/jpg/000000.jpg"  # Replace with the path to your image

print(image_path)

image = PILImg.open(image_path)

# Display the image using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)
ax.axis('off')  # Hide axes for better visualization

# List to store rectangles (each rectangle defined by two clicks: top-left and bottom-right)
rectangles = []
temp_points = []  # To store the current two points temporarily

# Function to handle clicks on the image
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        temp_points.append((x, y))
        print(f"Clicked at: (x={x}, y={y})")

        # After 2 points are clicked, define a rectangle
        if len(temp_points) == 2:
            (x1, y1), (x2, y2) = temp_points
            top_left_x = min(x1, x2)
            top_left_y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            # Store rectangle as (x, y, width, height)
            rectangles.append((top_left_x, top_left_y, width, height))

            # Draw the rectangle on the plot
            rect = patches.Rectangle((top_left_x, top_left_y), width, height,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            fig.canvas.draw()

            # Clear temp points for next rectangle
            temp_points.clear()

# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)

# Access the collected rectangles after plot is closed
def print_rectangles():
    print("\nCollected Rectangles:")
    for i, (x, y, w, h) in enumerate(rectangles):
        print(f"{i + 1}: (x={x}, y={y}, width={w}, height={h})")

plt.show()

# After closing the plot, call print_rectangles() manually if needed
# print_rectangles()

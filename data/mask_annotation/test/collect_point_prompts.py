# imports
import numpy as np
from PIL import Image as PILImg
import matplotlib.pyplot as plt

def get_clicked_points():
    clicked_points = [
        # x, y, pos or neg
        [(271, 195, 1)],
        [(331, 208, 1)],
        [(392, 207, 1)],
        [(453, 211, 1)],
        [(462, 300, 1)],
        [(402, 307, 1)],
        [(373, 330, 1)],
        [(296, 318, 1)],
        [(377, 430, 1)],
        [(436, 431, 1)],
        [(578, 463, 0)],
        [(558, 437, 1)],
        [(513, 288, 1)],
        [(162, 417, 1)],
        [(97, 437, 1)],
        [(40, 470, 1)]
    ]
    return clicked_points

# Load the image using PIL
image_path = "./imgs/sam2-test/rgb/000000.jpg"  # Replace with the path to your image
image = PILImg.open(image_path)

clicked_points = [] #get_clicked_points()

# Enable Matplotlib's interactive mode
# %matplotlib widget

# Display the image using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)
ax.axis('off')  # Hide axes for better visualization
# ax.set_title("Click on pixels to collect their locations", fontsize=16)

# """
# Connect the click event to the on_click function
# List to store clicked pixel locations
clicked_points = []

# Function to handle clicks on the image
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))
        print(f"Clicked at: (x={x}, y={y})")  # Print for immediate feedback

fig.canvas.mpl_connect('button_press_event', on_click)

# Access the collected points after the plot is closed
def print_points():
    print("\nCollected Points:")
    for i, (x, y) in enumerate(clicked_points):
        print(f"{i + 1}: (x={x}, y={y})")

# """

# Plot each point as a red marker
for obj_points in clicked_points:
    for x, y, label in obj_points:
        marker = 'g*' if label == 1 else 'r*'
        plt.plot(x, y, marker, markersize=10)  # 'ro' means red circles

plt.show()


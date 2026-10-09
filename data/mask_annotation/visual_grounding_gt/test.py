import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# Load the image
# Print the mask IDs in one mask PNG, e.g. <scene>/2/sam2/masks/00002.png
if len(sys.argv) != 2:
    sys.exit("usage: python test.py MASK_PNG")
img_path = sys.argv[1]
img = Image.open(img_path)


print(np.unique(np.array(img)), np.array(img).shape)  # Print the shape of the image array
# Plot the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Hide axes for better visualization
plt.show()
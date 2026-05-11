import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# Load the image
img_path = '/home/itaykadosh/Desktop/SCENES_W_MASKS/fill_ins_2/scene64.ecsw.axxess-atrium/2/sam2/masks/00002.png'
img = Image.open(img_path)


print(np.unique(np.array(img)), np.array(img).shape)  # Print the shape of the image array
# Plot the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Hide axes for better visualization
plt.show()
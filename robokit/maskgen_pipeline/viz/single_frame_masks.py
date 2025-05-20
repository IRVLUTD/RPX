"""
Usage: python single_frame_mask.py /path/to/your/mask.png
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_mask(mask_path):
    # Load mask image
    mask = np.array(Image.open(mask_path))

    # Plot with colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='nipy_spectral', interpolation='nearest')
    plt.colorbar(label='Instance ID')
    plt.title('Instance Segmentation Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mask.py <mask_path>")
        sys.exit(1)
    
    plot_mask(sys.argv[1])

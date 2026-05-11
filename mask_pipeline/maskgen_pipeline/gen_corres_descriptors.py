import numpy as np
import cv2
from pathlib import Path
from scipy.io import loadmat
import torch
from time import time
import matplotlib.pyplot as plt

# Assuming RDD.RDD and RDD.RDD_helper are defined elsewhere
from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8) * 255)

def filter_correspondences(pts0, pts1, conf, mask0_bin, mask1_bin, conf_threshold=0.9):
    """
    Filter correspondences to keep only those within masks and above confidence threshold.
    
    Args:
        pts0, pts1: Nx2 arrays of corresponding points.
        conf: N array of confidence scores.
        mask0_bin, mask1_bin: Binary masks (H x W).
        conf_threshold: Minimum confidence for keeping a match.
    
    Returns:
        Filtered pts0, pts1, and conf arrays.
    """
    valid = []
    for i in range(len(pts0)):
        x0, y0 = int(pts0[i, 0]), int(pts0[i, 1])
        x1, y1 = int(pts1[i, 0]), int(pts1[i, 1])
        if (0 <= y0 < mask0_bin.shape[0] and 0 <= x0 < mask0_bin.shape[1] and
            0 <= y1 < mask1_bin.shape[0] and 0 <= x1 < mask1_bin.shape[1]):
            if (mask0_bin[y0, x0] > 0 and mask1_bin[y1, x1] > 0 and conf[i] > conf_threshold):
                valid.append(i)
    return pts0[valid], pts1[valid], conf[valid]

def plot_correspondences(img0, img1, pts0, pts1, conf, num_lines=50):
    """
    Plot two images side by side with lines showing correspondences.
    
    Args:
        img0, img1: Input images (BGR format).
        pts0, pts1: Nx2 arrays of corresponding points (x, y).
        conf: Confidence scores for the correspondences.
        num_lines: Number of correspondence lines to draw (to avoid clutter).
    """
    # Convert BGR to RGB for Matplotlib
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display the images
    ax1.imshow(img0_rgb)
    ax1.set_title('Image 0')
    ax1.axis('off')
    ax2.imshow(img1_rgb)
    ax2.set_title('Image 1')
    ax2.axis('off')

    # Select a subset of correspondences to avoid clutter (e.g., top N by confidence)
    if len(pts0) > num_lines:
        indices = np.argsort(conf)[-num_lines:]  # Select top N correspondences by confidence
        pts0 = pts0[indices]
        pts1 = pts1[indices]

    # Plot points and lines
    for i in range(len(pts0)):
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        # Plot points on both images
        ax1.plot(x0, y0, 'ro', markersize=5)  # Red points on image 0
        ax2.plot(x1, y1, 'ro', markersize=5)  # Red points on image 1

        # Draw a line connecting the points (using figure coordinates)
        con = plt.Line2D(
            (x0, x1 + img0.shape[1]),  # x-coordinates (shift x1 by image width)
            (y0, y1),                  # y-coordinates
            transform=fig.transFigure,
            color='cyan',
            linewidth=1,
            alpha=0.5
        )
        fig.add_artist(con)

    # Adjust the layout to align images properly
    plt.tight_layout()
    plt.show()

def main(args):
    """
    Main function to estimate relative pose between two frames and visualize results.

    Args:
        args: Command-line arguments with scene_dir and mode.
    """
    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    sam2_dir = scene_dir / "sam2"
    mask_dir =  sam2_dir / "masks"
    depth_dir = scene_dir / "depth"
    # K_path = scene_dir / "cam_K.txt"

    # names = [('00000', '00001')]
    names = [('00132', '00133')]
    # names = [('00132', '00134')]
    # names = [('00152', '00153')]
    # names = [('00100', '00133')]
    # names = [('00000', '00133')]

    for _name0, _name1 in names:
        name0 = f"{_name0}.png"
        name1 = f"{_name1}.png"
        img0 = cv2.imread(str(rgb_dir / name0))
        img1 = cv2.imread(str(rgb_dir / name1))
        mask0 = cv2.imread(str(mask_dir / name0), cv2.IMREAD_UNCHANGED)
        mask1 = cv2.imread(str(mask_dir / name1), cv2.IMREAD_UNCHANGED)
        depth0 = cv2.imread(str(depth_dir / name0), cv2.IMREAD_UNCHANGED)
        depth1 = cv2.imread(str(depth_dir / name1), cv2.IMREAD_UNCHANGED)

        # K = load_intrinsics(K_path, img0.shape)
        # print("Camera Intrinsics K:\n", K)

        assert img0 is not None and img1 is not None, "Images not loaded"
        assert mask0 is not None and mask1 is not None, "Masks not loaded"
        assert depth0 is not None and depth1 is not None, "Depth images not loaded"

        # mask0_bin = (mask0 > 0).astype(np.uint8)
        # mask1_bin = (mask1 > 0).astype(np.uint8)

        RDD_model = build(weights='./weights/RDD-v2.pth')
        RDD_model.eval()
        RDD_wrap = RDD_helper(RDD_model)

        for i in range(1,8):

            mask1_bin = (mask1 == i).astype(np.uint8)
            mask0_bin = (mask0 == i).astype(np.uint8)
            img0_masked = apply_mask(img0, mask0_bin)
            img1_masked = apply_mask(img1, mask1_bin)

            start = time()
            # mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0, img1, resize=1024)

            # mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
            # mkpts_0, mkpts_1, conf = RDD_wrap.match(img0_masked, img1_masked, resize=1024)
            # mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
            
            # mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0, img1, resize=1024)
            mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
            print(f"Found {len(mkpts_0)} matches in {time() - start:.2f} seconds")
            plot_correspondences(img0_masked, img1, mkpts_0, mkpts_1, conf, num_lines=50)

            # Filter correspondences
            # pts0, pts1, conf = filter_correspondences(mkpts_0, mkpts_1, conf, mask0_bin, mask1_bin, conf_threshold=0.01)
            # print(f"Retained {len(pts0)} valid matches after filtering")

            # Visualize correspondences
            # plot_correspondences(img0_masked, img1, pts0, pts1, conf, num_lines=50)
            # if len(pts0) >= 8:
                # plot_correspondences(img0_masked, img1_masked, pts0, pts1, conf, num_lines=50)
            #     pass
            # else:
                # print(f"Warning: Only {len(pts0)} valid correspondences. Need at least 8 to visualize.")
                # pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory")
    args = parser.parse_args()
    main(args)
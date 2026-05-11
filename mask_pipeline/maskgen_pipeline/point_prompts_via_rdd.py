import os
import yaml
import numpy as np
import cv2
from pathlib import Path
import torch
from time import time
import matplotlib.pyplot as plt
from PIL import Image as PILImg

# Add parent directory to sys.path to find robokit
import sys
import pickle
sys.path.append(str(Path(__file__).parent))


# Custom modules (ensure these are available in your environment)
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor
from robokit.utils import overlay_masks, combine_masks, annotate

# Set the correct RDD module path relative to the script
rdd_path = Path(__file__).parent / "rdd"
print(f"Attempting to add RDD path: {rdd_path}")
sys.path.append(str(rdd_path))
print(f"Current sys.path: {sys.path}")

from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def points_to_bbox(points):
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    return [x_min, y_min, x_max, y_max]


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
    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    sam2_dir = scene_dir / "sam2"
    mask_dir = sam2_dir / "masks"
    depth_dir = scene_dir / "depth"
    old_rgb_mask_dir = sam2_dir / "rgb_and_mask"

    iter1_faulty_txt = sam2_dir / "iter1_faulty.txt"

    if iter1_faulty_txt.exists():
        with open(iter1_faulty_txt, 'r') as f:
            faulty_frames = [line.strip() for line in f if line.strip()]
        print(f"Found {len(faulty_frames)} faulty frames in {iter1_faulty_txt}")
    
    # gdino = GroundingDINOObjectPredictor()
    sam2 = SAM2Predictor()

    import pdb; pdb.set_trace()
    for curr_frame in faulty_frames:
        prev_frame = str(int(curr_frame) - 1).zfill(5)
        img0 = cv2.imread(str(rgb_dir / f"{prev_frame}.png"))
        img1 = cv2.imread(str(rgb_dir / f"{curr_frame}.png"))
        mask0 = cv2.imread(str(mask_dir / f"{prev_frame}.png"), cv2.IMREAD_UNCHANGED)
        mask1 = cv2.imread(str(mask_dir / f"{curr_frame}.png"), cv2.IMREAD_UNCHANGED)
        depth0 = cv2.imread(str(depth_dir / f"{prev_frame}.png"), cv2.IMREAD_UNCHANGED)
        depth1 = cv2.imread(str(depth_dir / f"{curr_frame}.png"), cv2.IMREAD_UNCHANGED)

        assert img0 is not None and img1 is not None, "Images not loaded"
        assert mask0 is not None and mask1 is not None, "Masks not loaded"
        assert depth0 is not None and depth1 is not None, "Depth images not loaded"

        mask0_bin = (mask0 > 0).astype(np.uint8)
        mask1_bin = (mask1 > 0).astype(np.uint8)
        img0_masked = apply_mask(img0, mask0_bin)
        img1_masked = apply_mask(img1, mask1_bin)

        # Load config file
        config_path = Path(__file__).parent / "rdd" / "configs" / "default.yaml"
        print(f"Using config path: {config_path}")

        # Set weights path
        weight_path = Path(__file__).parent.parent / "ckpts" / "rdd" / "RDD-v2.pth"
        print(f"Using weights path: {weight_path}")
        
        config=read_config(str(config_path))

        RDD_model = build(config=config, weights=weight_path)
        RDD_model.eval()
        RDD_wrap = RDD_helper(RDD_model)

        point_prompts = []
        point_labels = []

        obj_idxs = np.unique(mask1.astype(np.uint8))[1:]  # Exclude background (0)

        for i in obj_idxs:
            curr_point_labels = []
            
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
            # plot_correspondences(img0_masked, img1, mkpts_0, mkpts_1, conf, num_lines=50)

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
            
            # Collect point prompts for SAM2
            point_prompts.append(mkpts_1)
            point_labels.append(np.ones(mkpts_1.shape[0], dtype=int))  # All points are positive for the current object


            #############
            # point_prompts = np.concatenate(point_prompts, axis=0)
            # point_labels = np.concatenate(point_labels, axis=0)
            # _img1 = PILImg.fromarray(img1)

            # import pdb; pdb.set_trace()

            # new_masks, scores, logits = sam2.predict_mask_in_image_using_point_prompts(_img1, point_prompts, point_labels)
            # new_masks = torch.tensor(new_masks).unsqueeze(0) if len(new_masks.shape) < 4 else torch.tensor(new_masks)
            # new_masks_combined = combine_masks(new_masks[:, 0, :, :])

            # old_rgb_mask_overlay_path = old_rgb_mask_dir / f"{curr_frame}.png"
            # old_overlay = PILImg.open(old_rgb_mask_overlay_path) if old_rgb_mask_overlay_path.exists() else None

            # new_overlay = overlay_masks(_img1, new_masks_combined)

            # import pdb; pdb.set_trace()

            # # Plot mask comparison
            # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # if old_overlay is not None:
            #     axes[0].imshow(old_overlay)
            #     axes[0].set_title(f"Old Mask - {curr_frame}")
            # else:
            #     axes[0].set_title(f"Old Mask Not Found - {curr_frame}")
            # axes[1].imshow(new_overlay)
            # axes[1].set_title(f"New Mask (SAM2) - {curr_frame}")
            # for ax in axes:
            #     ax.axis("off")
            # plt.tight_layout()
            # plt.show()
            #############

        # exit()
        # point_prompts = np.concatenate(point_prompts, axis=0)
        # point_labels = np.concatenate(point_labels, axis=0)

        # convert bgr to rgb for PIL
        import pdb; pdb.set_trace()
        _img1 = PILImg.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

        data = {"img":_img1, "points": point_prompts, "labels": point_labels}
        with open("test.pkl", "wb") as f:
            pickle.dump(data, f)

        new_masks, scores, logits = sam2.predict_mask_in_image_using_point_prompts(_img1, point_prompts, point_labels)
        new_masks = torch.tensor(new_masks).unsqueeze(0) if len(new_masks.shape) < 4 else torch.tensor(new_masks)
        new_masks_combined = combine_masks(new_masks[:, 0, :, :])

        old_rgb_mask_overlay_path = old_rgb_mask_dir / f"{curr_frame}.png"
        old_overlay = PILImg.open(old_rgb_mask_overlay_path) if old_rgb_mask_overlay_path.exists() else None
        new_masks_combined = combine_masks(new_masks[:, 0, :, :])

        new_overlay = overlay_masks(_img1, new_masks_combined)

        import pdb; pdb.set_trace()

        # Plot mask comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if old_overlay is not None:
            axes[0].imshow(old_overlay)
            axes[0].set_title(f"Old Mask - {curr_frame}")
        else:
            axes[0].set_title(f"Old Mask Not Found - {curr_frame}")
        axes[1].imshow(new_overlay)
        axes[1].set_title(f"New Mask (SAM2) - {curr_frame}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True, help="Path to scene dir containing rgb/, masks/, depth/, cam_K.txt")
    args = parser.parse_args()
    main(args)
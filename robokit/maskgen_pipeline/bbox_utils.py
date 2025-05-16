#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + GPT-4)
#----------------------------------------------------------------------------------------------------
import numpy as np
import torch
import torchvision.ops as ops
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from .config import logger

def apply_nms(bboxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        bboxes (torch.Tensor): Tensor of shape [N, 4] in (x1, y1, x2, y2).
        scores (torch.Tensor): Confidence scores.
        iou_threshold (float): Overlap threshold.
    
    Returns:
        tuple: Filtered bboxes and keep indices.
    """
    keep_indices = ops.nms(bboxes, scores, iou_threshold)
    return bboxes[keep_indices], keep_indices

def remove_n_largest_bboxes(initial_bboxes, phrases, gdino_conf, n):
    """
    Remove n largest bounding boxes by area.
    
    Args:
        initial_bboxes (torch.Tensor): Tensor of shape [N, 4].
        phrases (list): Corresponding phrases.
        gdino_conf (torch.Tensor): Confidence scores.
        n (int): Number of boxes to remove.
    
    Returns:
        tuple: Filtered bboxes, phrases, and scores.
    """
    if n <= 0:
        return initial_bboxes, phrases, gdino_conf
    bbox_areas = initial_bboxes[:, 2] * initial_bboxes[:, 3]
    largest_indices = torch.argsort(bbox_areas, descending=True)[:n]
    keep_indices = torch.ones(len(initial_bboxes), dtype=torch.bool)
    keep_indices[largest_indices] = False
    return (
        initial_bboxes[keep_indices],
        [phrases[i] for i in range(len(phrases)) if keep_indices[i]],
        gdino_conf[keep_indices]
    )

def sort_boxes_by_area(boxes, phrases, scores):
    """
    Sort bounding boxes by area in ascending order.
    
    Args:
        boxes (list or np.ndarray): Boxes in [x1, y1, x2, y2].
        phrases (list): Corresponding phrases.
        scores (torch.Tensor or np.ndarray): Confidence scores.
    
    Returns:
        tuple: Sorted boxes, phrases, and scores.
    """
    boxes_np = np.array(boxes)
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    sorted_indices = np.argsort(areas)
    return (
        boxes_np[sorted_indices].tolist(),
        [phrases[i] for i in sorted_indices],
        scores[sorted_indices]
    )

def interactive_resize_bboxes(image_pil, bboxes):
    """
    Interactively resize bounding boxes.
    
    Args:
        image_pil (Image): Image to display.
        bboxes (list): List of [x1, y1, x2, y2] boxes.
    
    Returns:
        list: Resized bounding boxes.
    """
    resized_bboxes = []
    print(f"[🔧] Starting interactive resizing of {len(bboxes)} bounding boxes...")
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_pil)
        ax.set_title(f"Resize Box {i+1}/{len(bboxes)} — Close the window to confirm")
        ax.axis("off")
        rect_patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='lime', facecolor='none', lw=2)
        ax.add_patch(rect_patch)
        current_box = [x1, y1, x2, y2]
        def on_select(eclick, erelease):
            cx1, cy1 = eclick.xdata, eclick.ydata
            cx2, cy2 = erelease.xdata, erelease.ydata
            current_box[0] = min(cx1, cx2)
            current_box[1] = min(cy1, cy2)
            current_box[2] = max(cx1, cx2)
            current_box[3] = max(cy1, cy2)
        toggle_selector = RectangleSelector(ax, on_select,
                                           useblit=True,
                                           button=[1],
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
        plt.show()
        resized_bboxes.append(current_box)
    print(f"[✅] Resizing complete. {len(resized_bboxes)} boxes updated.")
    return resized_bboxes
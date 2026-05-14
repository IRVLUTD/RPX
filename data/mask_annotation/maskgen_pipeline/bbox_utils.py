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
    Interactively resize bounding boxes — single window, step through with N/P.

    Senior-engineer rev: image axis uses ``set_aspect('equal')`` so the
    rendered image NEVER stretches when the window is maximized. Layout uses
    explicit figure-fraction rects → deterministic at any window size.
    """
    from .visualization import (
        _BG, _BRAND_P, _DIM, _TEXT, _TEXT_DIM,
        _maximize_window, _style_image_axis,
    )

    bboxes = [list(b) for b in bboxes]
    n = len(bboxes)
    if n == 0:
        return bboxes

    state = {"target": 0, "selector": None}

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(_BG)
    try:
        fig.canvas.manager.set_window_title("RPX · Annotator")
    except Exception:
        pass

    # Image axis — same fixed rect as Step 1, aspect-preserved
    ax = fig.add_axes([0.025, 0.05, 0.95, 0.86])
    ax.imshow(image_pil)
    _style_image_axis(ax)

    fig.text(
        0.025, 0.955, "Resize Boxes",
        color=_TEXT, fontsize=14, weight="bold",
    )
    subtitle = fig.text(
        0.025, 0.928,
        f"L-drag to redraw  ·  Q / Enter / → : next box  ·  ← : back  ·  Esc to skip remaining  ·  Resizing 1 / {n}",
        color=_TEXT_DIM, fontsize=11,
    )

    # Pre-draw all boxes dimmed; reveal active in pink
    rects, tags = [], []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        r = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor=_DIM, facecolor="none",
                          lw=1.2, linestyle="--", alpha=0.6)
        ax.add_patch(r)
        rects.append(r)
        t = ax.text(
            x1 + 4, y1 + 18, f"{i + 1:02d}",
            color="white", fontsize=9, weight="bold", alpha=0.5,
            bbox=dict(facecolor=_DIM, edgecolor="none",
                      boxstyle="round,pad=0.25", alpha=0.6),
        )
        tags.append(t)

    def _activate(idx):
        for j in range(n):
            if j == idx:
                rects[j].set_edgecolor(_BRAND_P)
                rects[j].set_linewidth(2.5)
                rects[j].set_linestyle("-")
                rects[j].set_alpha(1.0)
                tags[j].set_alpha(1.0)
                tags[j].set_bbox(dict(facecolor=_BRAND_P, edgecolor="none",
                                      boxstyle="round,pad=0.25"))
            else:
                rects[j].set_edgecolor(_DIM)
                rects[j].set_linewidth(1.2)
                rects[j].set_linestyle("--")
                rects[j].set_alpha(0.6)
                tags[j].set_alpha(0.5)
                tags[j].set_bbox(dict(facecolor=_DIM, edgecolor="none",
                                      boxstyle="round,pad=0.25", alpha=0.6))
        state["target"] = idx
        subtitle.set_text(
            f"L-drag to redraw  ·  Q / Enter / → : next box  ·  ← : back  ·  "
            f"Esc to skip remaining  ·  Resizing {idx + 1} / {n}"
        )

        if state["selector"] is not None:
            try:
                state["selector"].set_active(False)
                state["selector"].disconnect_events()
            except Exception:
                pass

        def on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return
            x1 = min(eclick.xdata, erelease.xdata)
            y1 = min(eclick.ydata, erelease.ydata)
            x2 = max(eclick.xdata, erelease.xdata)
            y2 = max(eclick.ydata, erelease.ydata)
            bboxes[state["target"]] = [x1, y1, x2, y2]
            rects[idx].set_xy((x1, y1))
            rects[idx].set_width(x2 - x1)
            rects[idx].set_height(y2 - y1)
            tags[idx].set_position((x1 + 4, y1 + 18))
            fig.canvas.draw_idle()

        state["selector"] = RectangleSelector(
            ax, on_select,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords="pixels", interactive=True,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        # Esc — skip remaining boxes immediately, keep edits made so far
        if event.key == "escape":
            plt.close(fig)
            return
        # Q / Enter / N / right-arrow — confirm THIS box, advance to next.
        # Past the last box, that confirmation closes the window and the
        # pipeline moves on to SAM2 propagation.
        if event.key in ("q", "Q", "enter", "n", "N", "right"):
            if state["target"] >= n - 1:
                plt.close(fig)
            else:
                _activate(state["target"] + 1)
            return
        # Left-arrow / P — back to previous box
        if event.key in ("p", "P", "left"):
            _activate(max(0, state["target"] - 1))

    fig.canvas.mpl_connect("key_press_event", on_key)
    _activate(0)
    print(f"[🔧] Resizing — {n} box(es). N/P to navigate, drag to redraw, Q to confirm.")
    plt.show()
    print(f"[✅] Resizing complete. {n} boxes updated.")
    return bboxes
#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + GPT-4)
#----------------------------------------------------------------------------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from .config import SUPERVISION_COLORMAP, logger

def get_user_bboxes(image_path, save_path):
    """
    Interactively collect bounding boxes from user and save as numpy file.
    
    Args:
        image_path (str or Path): Path to input image.
        save_path (str or Path): Path to save .npy file.
    
    Returns:
        np.ndarray: Array of [x1, y1, x2, y2] boxes.
    """
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    bboxes = []
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.axis("off")
    start = []
    def on_press(event):
        if event.inaxes is not None:
            start.clear()
            start.extend([event.xdata, event.ydata])
    def on_release(event):
        if event.inaxes is not None:
            x1, y1 = start
            x2, y2 = event.xdata, event.ydata
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bboxes.append([x1, y1, x2, y2])
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', facecolor='none', lw=2))
            fig.canvas.draw()
    def on_key(event):
        if event.key == 'q':
            bboxes_np = np.array(bboxes)
            xywh = np.column_stack([
                bboxes_np[:, 0],
                bboxes_np[:, 1],
                bboxes_np[:, 2] - bboxes_np[:, 0],
                bboxes_np[:, 3] - bboxes_np[:, 1]
            ])
            np.save(save_path.with_suffix('.npy'), {"xyxy": bboxes_np, "xywh": xywh})
            print(f"[✅] Saved both XYXY and XYWH to: {save_path.with_suffix('.npy')}")
            plt.close()
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return np.array(bboxes)

def interactive_bbox_selection(image_pil, bboxes, phrases, gdino_conf):
    """
    Interactively select and add bounding boxes with Supervision-inspired colors.
    
    Args:
        image_pil (Image): Image to display.
        bboxes (list): List of [x1, y1, x2, y2] boxes.
        phrases (list): Corresponding phrases.
        gdino_conf (torch.Tensor): Confidence scores.
    
    Returns:
        tuple: Filtered bboxes, phrases, and scores.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_pil)
    plt.title("Toggle Boxes to Keep + Right Click to Add (Close to Confirm)")
    plt.axis("off")
    keep_flags = [True] * len(bboxes)
    patches = []
    new_boxes = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = SUPERVISION_COLORMAP(i % SUPERVISION_COLORMAP.N)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                            edgecolor=color, facecolor='none', lw=3)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Box {i+1}", color=color, fontsize=12, weight="bold")
        patches.append(rect)
    labels = [f"Box {i+1}" for i in range(len(bboxes))]
    rax = plt.axes([0.85, 0.3, 0.13, 0.4])
    check = CheckButtons(rax, labels, keep_flags)
    for i, text in enumerate(check.labels):
        color = SUPERVISION_COLORMAP(i % SUPERVISION_COLORMAP.N)
        text.set_color(color)
    def toggle(label):
        i = labels.index(label)
        keep_flags[i] = not keep_flags[i]
        patches[i].set_visible(keep_flags[i])
        fig.canvas.draw()
    check.on_clicked(toggle)
    start = []
    def on_mouse_press(event):
        if event.button == 3 and event.inaxes == ax:
            start.clear()
            start.extend([event.xdata, event.ydata])
    def on_mouse_release(event):
        if event.button == 3 and event.inaxes == ax and start:
            x1, y1 = start
            x2, y2 = event.xdata, event.ydata
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            idx = len(bboxes) + len(new_boxes)
            color = SUPERVISION_COLORMAP(idx % SUPERVISION_COLORMAP.N)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                edgecolor=color, facecolor='none', lw=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Box {idx}", color=color, fontsize=10, weight="bold")
            new_boxes.append([x1, y1, x2, y2])
            fig.canvas.draw()
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    plt.show()
    filtered_bboxes = [box for box, keep in zip(bboxes, keep_flags) if keep]
    filtered_phrases = [p for p, keep in zip(phrases, keep_flags) if keep]
    filtered_conf = gdino_conf[[i for i, keep in enumerate(keep_flags) if keep]]
    if new_boxes:
        filtered_bboxes.extend(new_boxes)
        filtered_phrases += ["object"] * len(new_boxes)
        filtered_conf = torch.cat([filtered_conf, torch.ones(len(new_boxes))], dim=0)
        print(f"[➕] Added {len(new_boxes)} new boxes interactively.")
    return filtered_bboxes, filtered_phrases, filtered_conf
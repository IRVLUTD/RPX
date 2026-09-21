#----------------------------------------------------------------------------------------------------
# Please check the licenses of the respective works utilized here before using this script.
#----------------------------------------------------------------------------------------------------
"""
Interactive bbox UIs for the SAM2 mask annotator.

Key design decisions (senior-engineer mode — set once, never iterate):

  • One module-level dark theme via mpl.rcParams. Applied once at import.
  • Image axes use ``ax.set_aspect("equal")`` so the image NEVER stretches
    when the window is resized / maximized. Whitespace is added around the
    image instead of distortion.
  • Layout is deterministic via explicit ``fig.add_axes(rect)`` figure-fraction
    rectangles. Window resize re-flows the rects but proportions stay locked.
  • Best-effort window maximization (Qt / Tk / wx). Never crashes if it fails.
  • Drop-in compatible: same signatures + return types as the original module.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from .config import SUPERVISION_COLORMAP, logger

# --------------------------------------------------------------------------- #
# Theme — once, at import time. All plt.figure() below inherits this.
# --------------------------------------------------------------------------- #
_BG = "#0f1116"          # window background
_PANEL = "#1a1d26"       # sidebar panel
_TEXT = "#f0f2f8"        # primary text
_TEXT_DIM = "#a8acbf"    # secondary text
_BRAND_P = "#ec4899"     # active-state pink
_DIM = "#5a5d72"         # dimmed dashed border

mpl.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor": _BG,
    "savefig.facecolor": _BG,
    "text.color": _TEXT,
    "axes.edgecolor": _PANEL,
    "axes.labelcolor": _TEXT_DIM,
    "axes.titlecolor": _TEXT,
    "xtick.color": _TEXT_DIM,
    "ytick.color": _TEXT_DIM,
    "font.family": ["Roboto", "DejaVu Sans", "sans-serif"],
})


# --------------------------------------------------------------------------- #
# Window maximize — best-effort across mpl backends
# --------------------------------------------------------------------------- #
def _go_fullscreen(fig):
    """Open the figure in fullscreen / maximized mode.

    Best-effort across mpl backends:
      • Qt   → ``showFullScreen()`` (true fullscreen — no OS chrome)
      • Tk   → ``attributes('-fullscreen', True)``
      • wx   → ``Maximize(True)``

    Safe: any backend that doesn't support fullscreen silently falls back
    to the default windowed mode.
    """
    try:
        mng = fig.canvas.manager
        backend = mpl.get_backend().lower()
        if "qt" in backend:
            try:
                mng.window.showFullScreen()
                return
            except Exception:
                mng.window.showMaximized()
                return
        if "tk" in backend:
            try:
                mng.window.attributes("-fullscreen", True)
                return
            except Exception:
                pass
            try:
                mng.window.attributes("-zoomed", True)
                return
            except Exception:
                pass
            try:
                mng.window.state("zoomed")
                return
            except Exception:
                pass
            try:
                mng.resize(
                    mng.window.winfo_screenwidth(),
                    mng.window.winfo_screenheight(),
                )
                return
            except Exception:
                pass
        if "wx" in backend:
            mng.frame.Maximize(True)
    except Exception:
        pass


# Backwards-compat alias — bbox_utils still imports the old name
_maximize_window = _go_fullscreen


def _style_image_axis(ax):
    """Configure an axis to host an image without stretching it.

    - ``aspect='equal'`` pins the image to its native aspect (no horizontal
      stretch when the window is wide; no vertical stretch when tall).
    - ``anchor='C'`` centres the image inside the axes rect.
    - All spines hidden; ticks off; backgrund = window background.
    """
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_facecolor(_BG)
    ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(False)


# =========================================================================== #
# Step 1 — bbox curation (single window: image left + sidebar checks right)
# =========================================================================== #
def interactive_bbox_selection(image_pil, bboxes, phrases, gdino_conf):
    """Interactively select bboxes; right-click+drag to add; close/Q to confirm.

    Drop-in for the legacy function. Same args/returns.
    """
    # Sort by confidence DESC — easy triage of low-conf at the bottom.
    confs_list = (gdino_conf.tolist() if isinstance(gdino_conf, torch.Tensor)
                  else list(gdino_conf))
    order = sorted(range(len(bboxes)), key=lambda k: -float(confs_list[k]))
    bboxes = [bboxes[k] for k in order]
    phrases = [phrases[k] for k in order]
    gdino_conf = torch.tensor([float(confs_list[k]) for k in order], dtype=torch.float32)

    n = len(bboxes)
    keep_flags = [True] * n
    new_boxes: list[list[float]] = []

    # One figure, two axes — image (left) and sidebar checkbuttons (right).
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(_BG)
    try:
        fig.canvas.manager.set_window_title("RPX · Annotator")
    except Exception:
        pass
    # Window opens at the natural 12×7 figsize — no fullscreen / maximize.
    # The user can resize the window manually if they want bigger.

    # IMAGE axis — fixed figure-fraction rect, aspect-preserved
    ax = fig.add_axes([0.025, 0.05, 0.74, 0.86])
    ax.imshow(image_pil)
    _style_image_axis(ax)

    # SIDEBAR — hand-drawn rows in figure-fraction coords (no CheckButtons
    # widget; that doesn't give us swatches, confidence bars, or hover state).
    sidebar_x = 0.795
    sidebar_w = 0.185
    row_h = 0.052          # height per row in figure-fraction
    sidebar_top = 0.86

    # Sidebar card background — only as tall as needed for n items
    card_h = min(0.86, n * row_h + 0.08)
    card_y = sidebar_top - card_h + 0.04
    sidebar_card = FancyBboxPatch(
        (sidebar_x, card_y), sidebar_w, card_h,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor=_PANEL, edgecolor="#23262f", linewidth=1, zorder=5,
    )
    fig.add_artist(sidebar_card)

    # Header inside the card
    fig.text(
        sidebar_x + 0.012, sidebar_top + 0.005,
        f"DETECTED ({n})",
        color=_TEXT_DIM, fontsize=10, weight="bold", zorder=6,
    )
    fig.text(
        sidebar_x + 0.012, sidebar_top - 0.018,
        "click a row to toggle",
        color=_TEXT_DIM, fontsize=9, zorder=6, alpha=0.7,
    )

    # Page header (above both axes)
    fig.text(
        0.025, 0.955, "Curate Boxes",
        color=_TEXT, fontsize=14, weight="bold",
    )
    fig.text(
        0.025, 0.928,
        "click a sidebar row to drop  ·  right-click + drag to add  ·  Q to confirm",
        color=_TEXT_DIM, fontsize=11,
    )

    # Draw the boxes on the image
    patches = []
    label_texts = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = SUPERVISION_COLORMAP(i % SUPERVISION_COLORMAP.N)
        face = (color[0], color[1], color[2], 0.10)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             edgecolor=color, facecolor=face, lw=2)
        ax.add_patch(rect)
        tag = ax.text(
            x1 + 4, y1 + 18, f"{i + 1:02d}",
            color="white", fontsize=10, weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
        )
        patches.append(rect)
        label_texts.append(tag)

    # ---------- Sidebar rows — drawn ONCE, mutated in place on toggle ----------
    # Pre-create every artist per row; cache them in row_handles. On click
    # we ONLY update their attributes (facecolor, alpha, visibility). No
    # artist add/remove cycles — that's what was causing the "collapse" bug.
    row_handles: dict[int, dict] = {}
    row_top0 = sidebar_top - 0.05
    row_colors: dict[int, tuple] = {}

    swatch_x = sidebar_x + 0.014
    swatch_w = 0.012
    label_x = swatch_x + swatch_w + 0.008
    bar_x_end = sidebar_x + sidebar_w - 0.030
    bar_h = 0.0028

    for i in range(n):
        cy = row_top0 - i * row_h
        color = SUPERVISION_COLORMAP(i % SUPERVISION_COLORMAP.N)
        row_colors[i] = color
        conf = float(gdino_conf[i])
        conf_pct = int(round(conf * 100))

        # Swatch (mutable facecolor on toggle)
        swatch = FancyBboxPatch(
            (swatch_x, cy - 0.009), swatch_w, 0.018,
            boxstyle="round,pad=0.001,rounding_size=0.003",
            transform=fig.transFigure,
            facecolor=color, edgecolor="none", zorder=11,
        )
        fig.add_artist(swatch)

        # Label (mutable color on toggle)
        label = fig.text(
            label_x, cy + 0.004, f"obj_{i + 1:02d}",
            color=_TEXT, fontsize=10, weight="bold",
            va="center", zorder=12,
        )

        # Confidence percentage
        conf_text = fig.text(
            sidebar_x + sidebar_w - 0.012, cy + 0.004, f"{conf_pct}%",
            color=_TEXT_DIM, fontsize=10, ha="right", va="center", zorder=12,
        )

        # Confidence bar — track + filled portion
        track = plt.Rectangle(
            (label_x, cy - 0.011), bar_x_end - label_x, bar_h,
            color="#272a35", transform=fig.transFigure, zorder=11,
        )
        fig.add_artist(track)
        fill = plt.Rectangle(
            (label_x, cy - 0.011),
            (bar_x_end - label_x) * max(0.0, min(1.0, conf)), bar_h,
            color=color, transform=fig.transFigure, zorder=12,
        )
        fig.add_artist(fill)

        # Hidden strikethrough — toggled visible on drop
        strike = plt.Line2D(
            [label_x, label_x + 0.045],
            [cy + 0.005, cy + 0.005],
            color=_TEXT_DIM, linewidth=1, transform=fig.transFigure, zorder=13,
            visible=False,
        )
        fig.add_artist(strike)

        row_handles[i] = {
            "swatch": swatch,
            "label": label,
            "conf_text": conf_text,
            "fill": fill,
            "strike": strike,
            "cy": cy,
        }

    # Click handler — direct coord check, no picker (matplotlib's picker is
    # unreliable when artists span figure-fraction coords)
    def on_click(event):
        if event.button != 1:
            return
        if event.x is None or event.y is None:
            return
        # Convert pixel coords → figure-fraction
        fx = event.x / fig.bbox.width
        fy = event.y / fig.bbox.height
        if not (sidebar_x <= fx <= sidebar_x + sidebar_w):
            return
        # Locate which row was clicked
        for i in range(n):
            cy = row_handles[i]["cy"]
            if cy - row_h / 2 <= fy <= cy + row_h / 2:
                keep_flags[i] = not keep_flags[i]
                kept = keep_flags[i]
                # Mutate artist attributes in place
                color = row_colors[i]
                h = row_handles[i]
                h["swatch"].set_facecolor(color if kept else (0.45, 0.45, 0.5, 0.6))
                h["label"].set_color(_TEXT if kept else _TEXT_DIM)
                h["conf_text"].set_color(_TEXT_DIM if kept else "#5a5d72")
                h["fill"].set_color(color if kept else (0.40, 0.42, 0.50, 1.0))
                h["strike"].set_visible(not kept)
                # Sync the bbox on the image
                patches[i].set_visible(kept)
                label_texts[i].set_visible(kept)
                fig.canvas.draw_idle()
                return

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Right-click + drag to add a new box
    rdrag_start: dict[str, float | None] = {"x": None, "y": None}

    def on_press(event):
        if event.button == 3 and event.inaxes == ax:
            rdrag_start["x"], rdrag_start["y"] = event.xdata, event.ydata

    def on_release(event):
        if (event.button == 3 and event.inaxes == ax
                and rdrag_start["x"] is not None and event.xdata is not None):
            x1, y1 = rdrag_start["x"], rdrag_start["y"]
            x2, y2 = event.xdata, event.ydata
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if (x2 - x1) >= 6 and (y2 - y1) >= 6:
                idx = n + len(new_boxes)
                color = SUPERVISION_COLORMAP(idx % SUPERVISION_COLORMAP.N)
                face = (color[0], color[1], color[2], 0.10)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor=color, facecolor=face, lw=2)
                ax.add_patch(rect)
                ax.text(
                    x1 + 4, y1 + 18, f"{idx + 1:02d}",
                    color="white", fontsize=10, weight="bold",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.3"),
                )
                new_boxes.append([x1, y1, x2, y2])
                fig.canvas.draw_idle()
            rdrag_start["x"] = rdrag_start["y"] = None

    def on_key(event):
        if event.key in ("q", "Q", "enter", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    filtered_bboxes = [b for b, k in zip(bboxes, keep_flags) if k]
    filtered_phrases = [p for p, k in zip(phrases, keep_flags) if k]
    keep_idx = [i for i, k in enumerate(keep_flags) if k]
    filtered_conf = gdino_conf[keep_idx]
    if new_boxes:
        filtered_bboxes.extend(new_boxes)
        filtered_phrases += ["object"] * len(new_boxes)
        filtered_conf = torch.cat([filtered_conf, torch.ones(len(new_boxes))], dim=0)
        print(f"[➕] Added {len(new_boxes)} new boxes interactively.")
    return filtered_bboxes, filtered_phrases, filtered_conf


# =========================================================================== #
# Legacy helper kept for parity (not used in the main pipeline)
# =========================================================================== #
def get_user_bboxes(image_path, save_path):
    """Click-drag boxes onto an image; press Q to save xyxy/xywh npz."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    bboxes = []
    fig, ax = plt.subplots(figsize=(12, 9))
    _maximize_window(fig)
    ax.imshow(image)
    _style_image_axis(ax)
    fig.text(0.025, 0.955, "Draw Boxes", color=_TEXT, fontsize=14, weight="bold")
    fig.text(0.025, 0.928, "click + drag to add  ·  Q to save",
             color=_TEXT_DIM, fontsize=11)
    start = []

    def on_press(event):
        if event.inaxes is not None:
            start.clear()
            start.extend([event.xdata, event.ydata])

    def on_release(event):
        if event.inaxes is not None and start:
            x1, y1 = start
            x2, y2 = event.xdata, event.ydata
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bboxes.append([x1, y1, x2, y2])
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor=_BRAND_P, facecolor="none", lw=2))
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            if not bboxes:
                plt.close(fig)
                return
            bboxes_np = np.array(bboxes)
            xywh = np.column_stack([
                bboxes_np[:, 0], bboxes_np[:, 1],
                bboxes_np[:, 2] - bboxes_np[:, 0],
                bboxes_np[:, 3] - bboxes_np[:, 1],
            ])
            np.save(save_path.with_suffix(".npy"), {"xyxy": bboxes_np, "xywh": xywh})
            print(f"[✅] Saved both XYXY and XYWH to: {save_path.with_suffix('.npy')}")
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return np.array(bboxes)

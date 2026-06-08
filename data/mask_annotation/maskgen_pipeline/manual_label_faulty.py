"""Manual per-frame labeling for the stubborn frames auto-refinement can't fix.

For each frame in ``iter{N}_faulty.txt``:

  1. Look up the nearest *verified* frame's mask → extract per-instance
     bboxes as a starting point (so object IDs stay consistent).
  2. Open the polished ``interactive_resize_bboxes`` UI on the faulty
     frame with those starting bboxes. The user drags to redraw each
     box if needed (N / Enter / Q to advance, P / ← to go back).
  3. Run SAM2's *image* predictor on each finalised bbox → per-instance mask.
  4. Combine the per-instance masks, save the result to:

        <phase>/sam2/masks/<frame>.png            (uint16 ID map — primary)
        <phase>/sam2/palette/<frame>.png          (palette-coloured PNG)
        <phase>/sam2/rgb_and_mask/<frame>.png     (RGB blended)
        <phase>/sam2/contour_gt_masks/<frame>.png (RGB + contour overlay)

     Also drops a copy in ``masks_verified/`` so the reviewer skips it
     on the next ``--no_verified`` pass and ``gen_faulty_from_verified``
     no longer flags it.

Usage::

    python -m maskgen_pipeline.manual_label_faulty \\
        --scene_dir <phase_dir> --iter 2

If ``--iter`` is omitted, the tool processes every frame in
``sam2/masks/`` that isn't already in ``masks_verified/``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from robokit.perception import SAM2Predictor
from .config import OUTPUT_DIRS, logger
from .visualization import (
    _BG, _BRAND_P, _DIM, _TEXT, _TEXT_DIM,
)
from .mask_processing import (
    apply_palette,
    combine_masks,
    load_palette,
    merge_rgb_with_mask,
    merge_rgb_with_mask_with_contours,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_int_list(fp: Path) -> list[int]:
    if not fp.exists():
        return []
    out = []
    for line in fp.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError:
            continue
    return sorted(set(out))


def _find_faulty_frames(phase_dir: Path, iter_num: int | None) -> list[int]:
    """Resolve the list of faulty frames to relabel."""
    masks_dir = phase_dir / "sam2" / "masks"
    verified_dir = phase_dir / "sam2" / "masks_verified"

    if iter_num is not None:
        fp = phase_dir / "sam2" / f"iter{iter_num}_faulty.txt"
        return _read_int_list(fp)

    # Fall-back: everything in masks/ that isn't already verified
    all_frames = {int(p.stem) for p in masks_dir.glob("*.png") if p.stem.isdigit()}
    verified = set()
    if verified_dir.is_dir():
        verified = {int(p.stem) for p in verified_dir.glob("*.png") if p.stem.isdigit()}
    return sorted(all_frames - verified)


def _bboxes_from_mask(mask: np.ndarray) -> dict[int, list[float]]:
    """Per-instance xyxy bboxes by ID (skips background 0)."""
    out: dict[int, list[float]] = {}
    for obj_id in np.unique(mask):
        if int(obj_id) == 0:
            continue
        ys, xs = np.where(mask == obj_id)
        if xs.size == 0:
            continue
        out[int(obj_id)] = [float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())]
    return out


def _seed_bboxes_for(phase_dir: Path, frame_id: int) -> tuple[list[int], list[list[float]]]:
    """Get starting bboxes from the closest verified frame's uint16 ID map.

    ``masks_verified/`` stores the CONTOUR OVERLAY (3-channel RGB), not the
    uint16 ID map. We use it only as the verified-set index and always read
    the actual ID map from ``masks/<id>.png``.

    Returns ``(object_ids, bboxes)`` — parallel lists.
    """
    verified_dir = phase_dir / "sam2" / "masks_verified"
    masks_dir = phase_dir / "sam2" / "masks"

    # 1. Find the nearest VERIFIED frame id (file stem)
    verified_ids: list[int] = []
    if verified_dir.is_dir():
        verified_ids = sorted(
            int(p.stem) for p in verified_dir.glob("*.png") if p.stem.isdigit()
        )

    if verified_ids:
        nearest_id = min(verified_ids, key=lambda v: abs(v - frame_id))
        src_path = masks_dir / f"{nearest_id:05d}.png"
    else:
        # No verified neighbours yet — fall back to the faulty frame's own mask
        src_path = masks_dir / f"{frame_id:05d}.png"

    if not src_path.is_file():
        return [], []

    # 2. Read the uint16 ID map. Defensive: collapse to 2D if anything else
    mask = np.array(Image.open(src_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    elif mask.ndim != 2:
        return [], []

    boxes = _bboxes_from_mask(mask)
    ids = sorted(boxes.keys())
    return ids, [boxes[i] for i in ids]


def interactive_relabel_frame(
    image_pil,
    seed_ids: list[int],
    seed_bboxes: list[list[float]],
):
    """Single-window relabel UI: resize, ADD, and REMOVE per-frame bboxes.

    Same chrome as ``interactive_resize_bboxes`` plus:
      • **Right-click + drag in image** → add a new bbox (gets the next
        available object id).
      • **D** or **Delete / Backspace** → remove the active bbox.

    Returns ``(final_ids, final_bboxes)`` — parallel lists, IDs preserved
    where possible.
    """
    bboxes = [list(b) for b in seed_bboxes]
    ids = list(seed_ids)
    next_id = (max(ids) + 1) if ids else 1
    state = {"target": 0, "selector": None}

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(_BG)
    try:
        fig.canvas.manager.set_window_title("RPX · Manual Relabel")
    except Exception:
        pass

    ax = fig.add_axes([0.025, 0.05, 0.95, 0.86])
    ax.imshow(image_pil)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_facecolor(_BG)
    ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.025, 0.955, "Manual Relabel", color=_TEXT, fontsize=14, weight="bold")
    subtitle = fig.text(
        0.025, 0.928,
        "L-drag: redraw active  ·  R-drag: add box  ·  D/Del: remove active  ·  "
        "N/Enter/Q: next  ·  P/←: prev  ·  Esc: skip frame",
        color=_TEXT_DIM, fontsize=10,
    )
    counter = fig.text(
        0.025, 0.905,
        f"obj_{ids[0]:02d}  ·  1 / {len(bboxes)}" if bboxes else "no boxes",
        color=_BRAND_P, fontsize=11, weight="bold",
    )

    rects: list[plt.Rectangle] = []
    tags: list = []

    def _draw_box(i):
        x1, y1, x2, y2 = bboxes[i]
        r = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor=_DIM, facecolor="none",
                          lw=1.2, linestyle="--", alpha=0.6)
        ax.add_patch(r)
        t = ax.text(
            x1 + 4, y1 + 18, f"{ids[i]:02d}",
            color="white", fontsize=9, weight="bold", alpha=0.5,
            bbox=dict(facecolor=_DIM, edgecolor="none",
                      boxstyle="round,pad=0.25", alpha=0.6),
        )
        rects.append(r)
        tags.append(t)

    for i in range(len(bboxes)):
        _draw_box(i)

    cancelled = {"v": False}

    def _refresh_styles():
        tgt = state["target"]
        for j in range(len(bboxes)):
            if j == tgt:
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

        if bboxes:
            counter.set_text(f"obj_{ids[tgt]:02d}  ·  {tgt + 1} / {len(bboxes)}")
        else:
            counter.set_text("no boxes — R-drag to add one")
        fig.canvas.draw_idle()

    def _activate(idx):
        if not bboxes:
            return
        idx = max(0, min(len(bboxes) - 1, idx))
        state["target"] = idx

        # Reset RectangleSelector to the active box
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
            rects[state["target"]].set_xy((x1, y1))
            rects[state["target"]].set_width(x2 - x1)
            rects[state["target"]].set_height(y2 - y1)
            tags[state["target"]].set_position((x1 + 4, y1 + 18))
            fig.canvas.draw_idle()

        state["selector"] = RectangleSelector(
            ax, on_select,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords="pixels", interactive=True,
        )
        _refresh_styles()

    rdrag_start: dict[str, float | None] = {"x": None, "y": None}

    def on_press(event):
        # Right-click drag → add new bbox
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
                nonlocal next_id
                new_id = next_id
                next_id += 1
                ids.append(new_id)
                bboxes.append([x1, y1, x2, y2])
                _draw_box(len(bboxes) - 1)
                _activate(len(bboxes) - 1)
            rdrag_start["x"] = rdrag_start["y"] = None

    def _remove_active():
        if not bboxes:
            return
        tgt = state["target"]
        # Remove the visual rect + tag
        try: rects[tgt].remove()
        except Exception: pass
        try: tags[tgt].remove()
        except Exception: pass
        bboxes.pop(tgt)
        ids.pop(tgt)
        rects.pop(tgt)
        tags.pop(tgt)
        # Move target to a still-valid index
        new_tgt = min(tgt, len(bboxes) - 1)
        state["target"] = max(0, new_tgt)
        if bboxes:
            _activate(state["target"])
        else:
            _refresh_styles()

    def on_key(event):
        if event.key == "escape":
            cancelled["v"] = True
            plt.close(fig)
            return
        if event.key in ("d", "D", "delete", "backspace"):
            _remove_active()
            return
        if event.key in ("q", "Q", "enter", "n", "N", "right"):
            if state["target"] >= len(bboxes) - 1:
                plt.close(fig)
            else:
                _activate(state["target"] + 1)
            return
        if event.key in ("p", "P", "left"):
            _activate(max(0, state["target"] - 1))

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    if bboxes:
        _activate(0)
    plt.show()

    if cancelled["v"]:
        return None, None
    return list(ids), [list(b) for b in bboxes]


def _save_outputs(
    phase_dir: Path,
    frame_id: int,
    rgb: Image.Image,
    combined_mask: torch.Tensor,
    palette,
) -> Path:
    name = f"{frame_id:05d}.png"
    sam2_out = phase_dir / "sam2"

    # uint16 ID map (primary output the benchmark reads)
    mask_np = combined_mask.cpu().numpy().astype(np.uint16)
    Image.fromarray(mask_np).save(sam2_out / "masks" / name)

    # Palette, blended, contour versions for visual review
    palette_img = apply_palette(combined_mask, palette)
    palette_img.save(sam2_out / "palette" / name)
    blended = merge_rgb_with_mask(rgb, palette_img)
    blended.save(sam2_out / "rgb_and_mask" / name)
    contour_img = merge_rgb_with_mask_with_contours(rgb, palette_img)
    contour_img.save(sam2_out / "contour_gt_masks" / name)

    # Drop into masks_verified/ so the reviewer + bridge skip it next round
    verified_dir = sam2_out / "masks_verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sam2_out / "contour_gt_masks" / name, verified_dir / name)

    return sam2_out / "masks" / name


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Manually relabel stubborn faulty frames one at a time."
    )
    ap.add_argument("--scene_dir", type=str, required=True,
                    help="Phase directory (contains rgb/, sam2/, etc.)")
    ap.add_argument("--iter", type=int, default=None,
                    help="If set, reads sam2/iter{N}_faulty.txt. "
                         "Otherwise relabels everything in masks/ that isn't in masks_verified/.")
    args = ap.parse_args()

    phase_dir = Path(args.scene_dir)
    rgb_dir = phase_dir / "rgb"
    if not rgb_dir.is_dir():
        sys.exit(f"rgb/ not found under {phase_dir}")

    faulty = _find_faulty_frames(phase_dir, args.iter)
    if not faulty:
        print("No faulty frames to relabel — every mask is already verified.")
        return

    print(f"\n[manual] {len(faulty)} faulty frame(s) to relabel in {phase_dir.name}/")
    print(f"[manual] Loading SAM2 (image predictor)…")
    sam2 = SAM2Predictor()
    palette = load_palette()

    done = 0
    skipped = 0
    for frame_id in faulty:
        rgb_path = rgb_dir / f"{frame_id:05d}.png"
        if not rgb_path.is_file():
            print(f"  [skip] no rgb for frame {frame_id:05d}")
            skipped += 1
            continue

        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb)

        ids, seed_boxes = _seed_bboxes_for(phase_dir, frame_id)
        if not seed_boxes:
            print(f"  [skip] no seed bboxes for frame {frame_id:05d}")
            skipped += 1
            continue

        print(f"\n[manual] frame {frame_id:05d} — {len(seed_boxes)} seed bbox(es). "
              "L-drag resize · R-drag add · D/Del remove · Q/N advance · ←/P back · Esc skip.")

        # Interactive: redraw / add / remove
        final_ids, final_boxes = interactive_relabel_frame(rgb, ids, seed_boxes)
        if final_boxes is None:
            print(f"  [skip] frame {frame_id:05d} cancelled")
            skipped += 1
            continue
        if len(final_boxes) == 0:
            print(f"  [skip] frame {frame_id:05d} ended with 0 boxes")
            skipped += 1
            continue

        # SAM2 single-image: bbox → mask per object
        with torch.inference_mode(), torch.autocast(sam2.device):
            sam2.img_predictor.set_image(rgb_np)
            per_obj_masks = []
            for bb in final_boxes:
                masks_np, scores, _ = sam2.img_predictor.predict(
                    box=np.array(bb)[None, :], multimask_output=False
                )
                per_obj_masks.append(torch.from_numpy(masks_np[0].astype(np.uint8)))

        # Combine per-object masks using the IDs returned by the relabel UI
        # (seeds keep their IDs; user-added boxes got fresh IDs)
        masks_tensor = torch.stack(per_obj_masks, dim=0)
        combined = torch.zeros(masks_tensor.shape[-2:], dtype=torch.int32)
        for obj_id, bin_mask in zip(final_ids, masks_tensor):
            combined[bin_mask.bool()] = int(obj_id)

        out_path = _save_outputs(phase_dir, frame_id, rgb, combined, palette)
        print(f"  [✓] saved {out_path}")
        done += 1

    print(f"\n[manual] done — relabeled {done}, skipped {skipped}, total {len(faulty)}")
    print(f"[manual] All relabeled frames added to {phase_dir / 'sam2' / 'masks_verified'}")


if __name__ == "__main__":
    main()

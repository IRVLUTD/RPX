# ----------------------------------------------------------------------------------------------------
# Interactive Mask → Object Mapper (Indexed Masks 1–7)
# Same cosmetics as your latest script, now uses indexed masks instead of palette colors
# ----------------------------------------------------------------------------------------------------
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from collections import OrderedDict

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass


def load_indexed_mask(mask_path):
    """Load an indexed mask (0=background, 1–7=objects)."""
    mask_img = Image.open(mask_path).convert("L")
    return np.array(mask_img, dtype=np.uint8)


def overlay_mask_indexed(rgb_img, mask_array, obj_idx):
    """Overlay a mask (indexed) with white outline + red tint."""
    rgb = np.array(rgb_img.convert("RGB"))
    overlay = rgb.copy()

    # Binary mask for object
    obj_mask = (mask_array == obj_idx).astype(np.uint8) * 255

    # White outline
    edges = cv2.Canny(obj_mask, 50, 150)
    outline = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    overlay[outline > 0] = [255, 255, 255]

    # Red tint interior
    tinted = overlay.copy()
    tinted[obj_mask > 0] = [255, 0, 0]
    blended = (0.6 * overlay + 0.4 * tinted).astype(np.uint8)

    return blended, obj_mask


def load_existing_mapping(iter_dir):
    """Load previous mask_to_object.json if available."""
    map_path = iter_dir / "sam2" / "mask_to_object.json"
    if map_path.exists():
        try:
            with open(map_path, "r") as f:
                data = json.load(f)
                return {int(k) - 1: v["object"] for k, v in data.items()}
        except Exception as e:
            print(f"[WARN] Could not load existing mapping: {e}")
    return {}


def interactive_label_masks(iter_dir, objects):
    mask_path = sorted((iter_dir / "sam2/masks").glob("*.png"))[0]
    rgb_path = sorted((iter_dir / "rgb").glob("*.png"))[0]

    mask_array = load_indexed_mask(mask_path)
    rgb_img = Image.open(rgb_path)
    object_indices = list(range(1, 8))  # masks 1–7

    existing_map = load_existing_mapping(iter_dir)
    mapping = OrderedDict(existing_map)

    current_idx = [0]
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"Mask → Object Labeling | {iter_dir.name}")

    def draw_frame():
        fig.clf()
        idx = current_idx[0]
        obj_idx = object_indices[idx]

        # --- Layout Panels ---
        ax_orig = fig.add_axes([0.05, 0.15, 0.35, 0.7])
        ax_mask = fig.add_axes([0.45, 0.15, 0.35, 0.7])
        ax_list = fig.add_axes([0.82, 0.15, 0.17, 0.7])
        ax_list.axis("off")

        # --- Left: Original RGB ---
        ax_orig.imshow(rgb_img)
        ax_orig.axis("off")
        ax_orig.add_patch(mpatches.Rectangle((0, 0), rgb_img.width, rgb_img.height,
                                             fill=False, edgecolor="Gray", linewidth=10))
        ax_orig.set_title("Original RGB", fontsize=12, weight="bold",
                          backgroundcolor="black", color="white")

        # --- Right: Mask Overlay ---
        blended, obj_mask = overlay_mask_indexed(rgb_img, mask_array, obj_idx)
        ax_mask.imshow(blended)
        ax_mask.axis("off")
        ax_mask.add_patch(mpatches.Rectangle((0, 0), rgb_img.width, rgb_img.height,
                                             fill=False, edgecolor="Gray", linewidth=10))
        ax_mask.set_title(f"Mask {idx+1}/7", fontsize=12, weight="bold",
                          backgroundcolor="black", color="white")

        # Display object name at centroid
        # current_object = mapping.get(idx, {}).get("name", None)
        # ys, xs = np.where(obj_mask > 0)
        # if len(xs) > 0:
        #     cx, cy = xs.mean(), ys.mean()
        #     name_display = current_object.upper() if current_object else "UNLABELED"
        #     ax_mask.text(cx, cy, name_display,
        #                  fontsize=14, weight="bold", color="yellow",
        #                  ha="center", va="center",
        #                  bbox=dict(facecolor="black", alpha=0.7, pad=3))

        # --- Object List (Compact with Highlight) ---
        start_y, step_y = 0.7, 0.06
        current_name = mapping.get(idx, {}).get("name", None)

        for i, obj in enumerate(objects):
            y = start_y - i * step_y
            label_text = f"[{i+1}] {obj['name']} ({obj['id']})"
            if current_name == obj["name"]:
                ax_list.add_patch(mpatches.Rectangle((-0.05, y - 0.025), 1.1, 0.05,
                                                     transform=ax_list.transAxes, color="yellow", alpha=0.3))
                ax_list.text(0, y, label_text, color="red", fontsize=9, weight="bold",
                             transform=ax_list.transAxes)
            else:
                ax_list.text(0, y, label_text, color="Black", fontsize=7,
                             transform=ax_list.transAxes)

        fig.suptitle("Mask → Object Mapping | Press ←/→ to navigate | 1–7 to assign | Q to quit",
                     fontsize=13, weight="bold", color="Black", backgroundcolor="White")
        plt.draw()

    def on_key(event):
        idx = current_idx[0]
        if event.key == "right":
            current_idx[0] = min(idx + 1, len(object_indices) - 1)
            draw_frame()
        elif event.key == "left":
            current_idx[0] = max(idx - 1, 0)
            draw_frame()
        elif event.key.isdigit():
            choice = int(event.key) - 1
            if 0 <= choice < len(objects):
                mapping[idx] = objects[choice]
                print(f"[✔] Mask {idx+1} → {objects[choice]['name']} ({objects[choice]['id']})")
                current_idx[0] = min(idx + 1, len(object_indices) - 1)
                draw_frame()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame()
    plt.show()

    # --- Canonicalize: phase-local pixel index → scene-level ref_idx ---
    # `objects` is the same list/order for every phase of this scene, so its
    # position IS the canonical ref_idx. Remapping here means every phase's
    # mask_to_object.json (and mask pixels) agree from the moment they're
    # authored — no downstream cross-phase realignment pass needed.
    canon_ref = {obj["id"]: i + 1 for i, obj in enumerate(objects)}
    remap = {}
    for i in range(len(object_indices)):
        old_px = i + 1
        obj = mapping.get(i)
        new_px = canon_ref.get(obj["id"]) if obj else None
        remap[old_px] = new_px if new_px is not None else old_px  # unlabeled: leave as-is

    if any(old != new for old, new in remap.items()):
        masks_dir = iter_dir / "sam2" / "masks"
        mask_files = sorted(masks_dir.glob("*.png"))
        for mp in mask_files:
            arr = np.array(Image.open(mp))
            new_arr = np.zeros_like(arr)
            for old_px, new_px in remap.items():
                new_arr[arr == old_px] = new_px
            Image.fromarray(new_arr.astype(arr.dtype)).save(mp)
        print(f"[↔] Canonicalized {len(mask_files)} masks in {iter_dir.name}: {remap}")

    # --- Save Mapping (canonical keys) ---
    mask_map = {
        str(remap[i + 1]): {
            "mask_index": remap[i + 1],
            "object": mapping.get(i, {"id": "unknown", "name": "unlabeled"})
        }
        for i in range(len(object_indices))
    }

    out_path = iter_dir / "sam2" / "mask_to_object.json"
    with open(out_path, "w") as f:
        json.dump(mask_map, f, indent=2)
    print(f"[✅] Saved mapping to {out_path}")


def main(scene_dir, json_file=None, objects_json=None):
    scene_dir = Path(scene_dir)

    if objects_json is not None:
        with open(objects_json, "r") as f:
            objects = json.load(f)
    else:
        with open(json_file, "r") as f:
            scenes = json.load(f)
        scene_id = int(scene_dir.name.split("scene")[-1].split(".")[0])
        scene_entry = next(s for s in scenes if s["scene_id"] == scene_id)
        objects = scene_entry["objects"]

    for iter_folder in sorted(scene_dir.glob("[0-9]")):
        print(f"\n=== Iteration {iter_folder.name} ===")
        interactive_label_masks(iter_folder, objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Mask → Object Mapper (Indexed 1–7)")
    parser.add_argument("--scene_dir", required=True, help="Path to scene folder")
    parser.add_argument("--json", help="Path to scenes.json (with scene_id lookup)")
    parser.add_argument("--objects_json", help="Path to flat objects list JSON [{id, name}, ...] — bypasses scenes.json")
    args = parser.parse_args()
    if args.objects_json is None and args.json is None:
        parser.error("Provide --json or --objects_json")
    main(args.scene_dir, json_file=args.json, objects_json=args.objects_json)

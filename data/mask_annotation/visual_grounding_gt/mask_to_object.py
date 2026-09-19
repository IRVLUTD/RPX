# ----------------------------------------------------------------------------------------------------
# Interactive Mask → Object Mapper (Indexed Masks)
# Same cosmetics as your latest script, now uses indexed masks instead of palette colors
# ----------------------------------------------------------------------------------------------------
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from collections import OrderedDict



def load_indexed_mask(mask_path):
    """Load an indexed mask (0=background, 1–7=objects)."""
    with Image.open(mask_path) as image:
        arr = np.array(image)
    if arr.ndim != 2 or arr.dtype.kind not in "ui":
        raise ValueError(f"Expected an indexed integer mask: {mask_path}")
    return arr


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
    mask_files = sorted((iter_dir / "sam2/masks").glob("*.png"))
    if not mask_files:
        raise ValueError(f"No masks in {iter_dir}")
    # Include IDs that only appear later in the clip. Show their first frame.
    representative = {}
    for path in mask_files:
        for label in np.unique(load_indexed_mask(path)):
            if label:
                representative.setdefault(int(label), path)
    object_indices = sorted(representative)
    if not object_indices:
        raise ValueError("Masks contain no foreground objects")
    if len(objects) > 9:
        raise ValueError("The keyboard mapper supports at most nine scene objects")

    existing_map = load_existing_mapping(iter_dir)
    mapping = OrderedDict(existing_map)

    current_idx = [0]
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"Mask → Object Labeling | {iter_dir.name}")

    def draw_frame():
        fig.clf()
        idx = current_idx[0]
        obj_idx = object_indices[idx]

        mask_path = representative[obj_idx]
        rgb_path = iter_dir / "rgb" / mask_path.name
        mask_array = load_indexed_mask(mask_path)
        rgb_img = Image.open(rgb_path)

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
        ax_mask.set_title(f"Mask {obj_idx} ({idx+1}/{len(object_indices)})", fontsize=12, weight="bold",
                          backgroundcolor="black", color="white")

        # Display object name at centroid
        # current_object = mapping.get(obj_idx - 1, {}).get("name", None)
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
        current_name = mapping.get(obj_idx - 1, {}).get("name", None)

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

        fig.suptitle("Mask → Object Mapping | Press ←/→ to navigate | 1–9 to assign | Q to quit",
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
        elif event.key and event.key.isdigit():
            choice = int(event.key) - 1
            if 0 <= choice < len(objects):
                mapping[object_indices[idx] - 1] = objects[choice]
                print(f"[✔] Mask {idx+1} → {objects[choice]['name']} ({objects[choice]['id']})")
                current_idx[0] = min(idx + 1, len(object_indices) - 1)
                draw_frame()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame()
    plt.show()

    # Validate the entire clip before touching any original file.
    assignments = {label: mapping.get(label - 1) for label in object_indices}
    remap = build_canonical_remap(object_indices, assignments, objects)
    mask_map = {
        str(remap[label]): {"mask_index": remap[label], "object": assignments[label]}
        for label in object_indices
    }
    masks_dir = iter_dir / "sam2" / "masks"
    if any(old != new for old, new in remap.items()):
        backup = masks_dir.with_name("masks_before_canonicalization")
        if backup.exists():
            raise ValueError(f"Preserve or move existing backup before relabeling: {backup}")
        staging = Path(tempfile.mkdtemp(prefix=".canonical-masks-", dir=masks_dir.parent))
        try:
            for path in mask_files:
                Image.fromarray(remap_mask(load_indexed_mask(path), remap)).save(staging / path.name)
            masks_dir.rename(backup)
            try:
                staging.rename(masks_dir)
            except BaseException:
                backup.rename(masks_dir)
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print(f"Canonicalized masks; originals preserved at {backup}")

    out_path = iter_dir / "sam2" / "mask_to_object.json"
    temporary = out_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(mask_map, indent=2) + "\n")
    os.replace(temporary, out_path)
    print(f"Saved mapping to {out_path}")


def build_canonical_remap(labels, assignments, objects):
    """Reject incomplete/non-injective mappings before mask files are changed."""
    ids = [obj["id"] for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Scene object IDs must be unique")
    canonical = {obj_id: index + 1 for index, obj_id in enumerate(ids)}
    remap = {}
    for label in labels:
        obj = assignments.get(label)
        if not obj or obj.get("id") not in canonical:
            raise ValueError(f"Mask {label} must be assigned to a known scene object")
        remap[label] = canonical[obj["id"]]
    if len(set(remap.values())) != len(remap):
        raise ValueError("Each mask must map to a different scene object")
    return remap


def remap_mask(arr, remap):
    """Apply a simultaneous label permutation, preserving background and dtype."""
    missing = set(np.unique(arr)) - {0} - set(remap)
    if missing:
        raise ValueError(f"Unmapped mask IDs: {sorted(missing)}")
    if max(remap.values(), default=0) > np.iinfo(arr.dtype).max:
        raise ValueError("Canonical IDs exceed the mask dtype")
    result = np.zeros_like(arr)
    for old, new in remap.items():
        result[arr == old] = new
    return result


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
    parser = argparse.ArgumentParser(description="Interactive Mask → Object Mapper (Indexed Masks)")
    parser.add_argument("--scene_dir", required=True, help="Path to scene folder")
    parser.add_argument("--json", help="Path to scenes.json (with scene_id lookup)")
    parser.add_argument("--objects_json", help="Path to flat objects list JSON [{id, name}, ...] — bypasses scenes.json")
    args = parser.parse_args()
    if args.objects_json is None and args.json is None:
        parser.error("Provide --json or --objects_json")
    main(args.scene_dir, json_file=args.json, objects_json=args.objects_json)

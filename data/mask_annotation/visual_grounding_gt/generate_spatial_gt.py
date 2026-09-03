"""Shared geometry and depth utilities for ground-truth generation.

Loads segmentation masks and depth maps, resolves the mask-index -> object
mapping, and turns them into per-object instances (pixel centroid, area,
median depth) that the task-specific generators in vqa_gt/ build questions
from.
"""
import json
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def imread_mask(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


# Depth PNGs in this dataset are natively encoded in millimeters. Confirmed
# by sampling 200 frames across 5 scenes: per-object MEDIAN depth never
# exceeds ~4600mm (consistent with a robot workspace), but a frame's raw MAX
# value routinely hits 65535 -- the standard 16-bit "no return" sensor
# sentinel, not a real distance. An earlier version of this function divided
# the whole frame by 1000 whenever that raw max exceeded 5000, meant to
# catch a different-unit frame -- but since the max is so often a sentinel
# rather than a real reading, that heuristic fired on ~63% of ordinary
# frames and silently corrupted their units. Comparisons that only depend on
# ordering (which object is closer, which is occluded) were unaffected,
# because the bug scaled an entire frame uniformly -- but anything using an
# absolute distance threshold (a minimum depth separation, a graspability
# clearance in mm) was not safe under it.
DEPTH_MAX_VALID_MM = 10000.0


def imread_depth(path):
    """Load a depth PNG as a float32 array of millimeters. 0 = no reading."""
    im = Image.open(path)
    if im.mode not in ("L", "I;16", "I"):
        im = im.convert("I")
    arr = np.array(im).astype(np.float32)
    arr[arr > DEPTH_MAX_VALID_MM] = 0.0
    return arr


def load_mapping(path):
    """mask_index -> {"name": str, "oid": fewsol_id str}, from a scene
    phase's mask_to_object.json."""
    with open(path) as f:
        data = json.load(f)
    mp = {}
    for k, v in data.items():
        try:
            key = int(k)
        except ValueError:
            continue
        obj = v.get("object", v)
        mp[key] = {
            "name": obj.get("name", str(key)),
            "oid": str(obj.get("id", key)),
        }
    return mp


def compute_instances(mask, mapping, depth=None):
    """One entry per labeled object visible in this frame: mask_id, name,
    oid, pixel area, pixel centroid (cx, cy), and median depth in mm (None
    if no depth map was given, or the object has no valid depth pixels)."""
    instances = {}
    for mid in np.unique(mask).tolist():
        if mid == 0 or mid not in mapping:
            continue
        m = mask == mid
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        inst = {
            "mask_id": mid,
            "name": mapping[mid]["name"],
            "oid": mapping[mid]["oid"],
            "area": int(m.sum()),
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "mask": m,  # kept in memory for occlusion checks; not serialized
        }
        if depth is not None:
            vals = depth[ys, xs]
            valid = vals[vals > 0]
            inst["median_depth"] = float(np.median(valid)) if len(valid) > 0 else None
        else:
            inst["median_depth"] = None
        instances[mid] = inst
    return instances


def load_phase(scene_dir, phase):
    root = os.path.join(scene_dir, str(phase))
    mask_dir = os.path.join(root, "sam2", "masks")
    map_path = os.path.join(root, "sam2", "mask_to_object.json")
    fids = sorted([
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(mask_dir, "*.png"))
    ])
    mapping = load_mapping(map_path)
    return root, fids, mapping


def select_frames(fids, n=10):
    """Evenly-spaced sample of n frame ids from a sorted frame list -- not a
    random sample, so repeated runs over the same staged scene are stable
    and coverage spreads across the whole sequence instead of clustering."""
    if len(fids) <= n:
        return fids
    step = len(fids) / n
    return [fids[int(i * step)] for i in range(n)]


DILATE_PX = 8            # pixels to dilate mask B when searching for adjacent occluders
MIN_OCC_FRACTION = 0.10  # min fraction of B's area that must be adjacent to a closer object


def detect_occlusion(instances):
    """
    B is occluded by A if:
      - A's mask intersects the dilation of B's mask (they are adjacent)
      - A's median depth < B's median depth (A is closer)
    Returns dict: name -> {"is_occluded": bool, "occluder": str|None, "fraction": float}

    Ordering-only comparison (A closer than B), so unaffected by the old
    depth-unit bug above -- kept here as a reusable building block even
    though the current three GT tasks (gt_spatial.py) don't call it.
    """
    inst_list = list(instances.values())
    results = {}

    for b in inst_list:
        if b["median_depth"] is None:
            results[b["name"]] = {"is_occluded": False, "occluder": None, "fraction": 0.0}
            continue

        dilated_b = binary_dilation(b["mask"], iterations=DILATE_PX)
        best_occluder = None
        best_fraction = 0.0

        for a in inst_list:
            if a["mask_id"] == b["mask_id"]:
                continue
            if a["median_depth"] is None or a["median_depth"] >= b["median_depth"]:
                continue

            overlap = dilated_b & a["mask"]
            overlap_count = int(overlap.sum())
            if overlap_count == 0:
                continue

            fraction = overlap_count / b["area"]
            if fraction > best_fraction:
                best_fraction = fraction
                best_occluder = a["name"]

        is_occluded = best_fraction >= MIN_OCC_FRACTION
        results[b["name"]] = {
            "is_occluded": is_occluded,
            "occluder": best_occluder if is_occluded else None,
            "fraction": round(best_fraction, 3),
        }

    return results

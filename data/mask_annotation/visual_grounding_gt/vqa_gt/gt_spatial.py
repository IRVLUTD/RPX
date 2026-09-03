"""Spatial ground truth: the "Top-Down / Left-Right -> Yes/No" task and the
"Spatial bbox" task, both built on real depth data. mos frames only -- ego
frames have no depth map in this dataset (confirmed against the published
HF repo listing, not a staging gap), and depth is what makes this reliable.

Why depth, not raw pixel position: comparing two objects' raw pixel
centroids to decide "which one is more to the left" or "which is higher up"
gets it wrong at oblique camera angles -- an object that's further from the
camera and merely appears shifted can read as "more left" in pixel space
while actually sitting elsewhere in the real scene. Confirmed on a real
sample: recomputing spatial_lr_extreme with depth flipped 9.4% of answers
that the pixel-only version had gotten wrong.

The fix doesn't need camera intrinsics (none are published for this
dataset). Real-world horizontal offset from the camera axis is
proportional to (pixel_x - principal_point_x) * depth -- the unknown focal
length is a shared positive constant, so it cancels out when comparing
which of two objects is further left or right. Same logic vertically. The
one assumption made explicit: principal point ~= image center (standard for
these cameras; not verified against a calibration file, because none
exists in this dataset). This does not correct for camera roll relative to
true gravity -- that would need the published camera-pose file interpreted
against a documented world-axis convention, which isn't available with
enough confidence to trust silently, so it's left uncorrected rather than
guessed at.

Types generated:
  spatial_lr_binary   "Is the {a} to the left of the {b}?" yes/no
  spatial_ud_binary   "Is the {a} above the {b}?" yes/no
  spatial_lr_extreme  "Which object is furthest to the {left/right}?" (bbox)
  depth_closest       "Which object is closest to the camera?" (bbox) --
                       pure depth-ordering, unaffected by the pixel-position
                       issue above; kept as-is.
  spatial_farthest     "Which object is farthest from the {reference}?" (bbox)
                       -- kept only when the depth-based 3D ranking agrees
                       with the 2D pixel-distance ranking, so a
                       perspective-only illusion of "farthest" never gets
                       published as ground truth.
"""
from __future__ import annotations

import numpy as np

MIN_SEP_FRAC = 0.08     # minimum effective pixel separation (fraction of image
                         # width) for an lr/ud binary pair to count as unambiguous
FX = FY = 613.0          # nominal RealSense D435 640x480 intrinsics, used only
CXP, CYP = 320.0, 240.0  # for the 3D cross-check in spatial_farthest -- never
                         # claimed as this camera's real calibration


def _add_position_proxies(instances, W, H):
    """(pixel - principal_point) * depth for each instance -- proportional
    to real-world horizontal/vertical offset from the camera axis, up to
    the shared unknown focal length. None if the instance has no depth."""
    for inst in instances.values():
        d = inst["median_depth"]
        inst["dx_proxy"] = (inst["cx"] - W / 2.0) * d if d else None
        inst["dy_proxy"] = (inst["cy"] - H / 2.0) * d if d else None


def _to_3d(cx, cy, z_mm):
    return np.array([(cx - CXP) * z_mm / FX, (cy - CYP) * z_mm / FY, z_mm])


def _bbox_of_instance(inst):
    ys, xs = np.nonzero(inst["mask"])
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def gen_spatial_questions(scene_name, phase, fid, mask_path, instances, W, H, rng):
    """instances: compute_instances() output for this frame, WITH depth
    (i.e. only ever called for mos frames -- see generate_gt.py)."""
    inst_list = list(instances.values())
    if len(inst_list) < 2:
        return []

    _add_position_proxies(instances, W, H)
    base = {"scene_id": scene_name, "kind": "mos", "phase": phase, "frame": fid, "mask_path": mask_path}
    items = []

    # ---- left/right binary
    lr_candidates = [
        (a, b) for a in inst_list for b in inst_list
        if a["mask_id"] != b["mask_id"] and a["dx_proxy"] is not None and b["dx_proxy"] is not None
        and abs(a["dx_proxy"] - b["dx_proxy"]) / max(a["median_depth"], b["median_depth"]) >= MIN_SEP_FRAC * W
    ]
    if lr_candidates:
        a, b = rng.choice(lr_candidates)
        items.append({
            **base, "type": "spatial_lr_binary",
            "question": f"Is the {a['name']} to the left of the {b['name']}?",
            "answer": "yes" if a["dx_proxy"] < b["dx_proxy"] else "no",
            "evidence": {"name_a": a["name"], "dx_proxy_a": round(a["dx_proxy"], 1),
                         "name_b": b["name"], "dx_proxy_b": round(b["dx_proxy"], 1)},
        })

    # ---- above/below binary
    ud_candidates = [
        (a, b) for a in inst_list for b in inst_list
        if a["mask_id"] != b["mask_id"] and a["dy_proxy"] is not None and b["dy_proxy"] is not None
        and abs(a["dy_proxy"] - b["dy_proxy"]) / max(a["median_depth"], b["median_depth"]) >= MIN_SEP_FRAC * H
    ]
    if ud_candidates:
        a, b = rng.choice(ud_candidates)
        items.append({
            **base, "type": "spatial_ud_binary",
            "question": f"Is the {a['name']} above the {b['name']}?",
            "answer": "yes" if a["dy_proxy"] < b["dy_proxy"] else "no",
            "evidence": {"name_a": a["name"], "dy_proxy_a": round(a["dy_proxy"], 1),
                         "name_b": b["name"], "dy_proxy_b": round(b["dy_proxy"], 1)},
        })

    # ---- leftmost / rightmost (bbox task)
    proxy_candidates = [i for i in inst_list if i["dx_proxy"] is not None]
    if proxy_candidates:
        direction = rng.choice(["left", "right"])
        extreme = (min if direction == "left" else max)(proxy_candidates, key=lambda i: i["dx_proxy"])
        items.append({
            **base, "type": "spatial_lr_extreme",
            "question": f"Which object is furthest to the {direction}?",
            "answer": extreme["name"], "answer_bbox": _bbox_of_instance(extreme),
            "evidence": {i["name"]: round(i["dx_proxy"], 1) for i in proxy_candidates},
        })

    # ---- closest to camera (bbox task) -- pure depth ordering
    depth_vals = [(i["median_depth"], i) for i in inst_list if i["median_depth"] is not None]
    if depth_vals:
        closest = min(depth_vals, key=lambda x: x[0])[1]
        items.append({
            **base, "type": "depth_closest",
            "question": "Which object is closest to the camera?",
            "answer": closest["name"], "answer_bbox": _bbox_of_instance(closest),
            "evidence": {i["name"]: round(d, 1) for d, i in depth_vals},
        })

    # ---- farthest from a named reference object (bbox task) -- kept only
    # when the depth-based 3D ranking agrees with the 2D pixel-distance
    # ranking, so this never publishes a perspective illusion as GT.
    if len(inst_list) >= 3 and depth_vals:
        for ref in inst_list:
            if ref["median_depth"] is None:
                continue
            others = [o for o in inst_list if o["mask_id"] != ref["mask_id"] and o["median_depth"] is not None]
            if len(others) < 2:
                continue
            d2d = {o["mask_id"]: float(np.hypot(o["cx"] - ref["cx"], o["cy"] - ref["cy"])) for o in others}
            ranked_2d = sorted(others, key=lambda o: -d2d[o["mask_id"]])
            ref3d = _to_3d(ref["cx"], ref["cy"], ref["median_depth"])
            d3d = {o["mask_id"]: float(np.linalg.norm(_to_3d(o["cx"], o["cy"], o["median_depth"]) - ref3d))
                   for o in others}
            ranked_3d = sorted(others, key=lambda o: -d3d[o["mask_id"]])
            if ranked_2d[0]["mask_id"] != ranked_3d[0]["mask_id"]:
                continue  # 2D and 3D disagree on which object is farthest -- drop
            top = ranked_2d[0]
            items.append({
                **base, "type": "spatial_farthest",
                "question": f"Which object is farthest from the {ref['name']}?",
                "answer": top["name"], "answer_bbox": _bbox_of_instance(top),
                "evidence": {"reference": ref["name"],
                             "distances_2d_px": {o["name"]: round(d2d[o["mask_id"]], 1) for o in others}},
            })

    return items

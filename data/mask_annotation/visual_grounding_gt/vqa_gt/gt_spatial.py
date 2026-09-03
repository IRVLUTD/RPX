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
    assert len(xs) > 0, (
        f"instance {inst.get('name')!r} has no mask pixels in this frame -- a "
        "question is about to reference an object that isn't actually visible "
        "here. Should be unreachable: every instance in instances came from "
        "compute_instances() iterating this exact frame's mask array."
    )
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _cap(candidates, max_per_type, rng):
    if max_per_type is None or len(candidates) <= max_per_type:
        return candidates
    return rng.sample(candidates, max_per_type)


def gen_spatial_questions(scene_name, phase, fid, mask_path, instances, W, H, rng, max_per_type=5):
    """instances: compute_instances() output for this frame, WITH depth
    (i.e. only ever called for mos frames -- see generate_gt.py).

    max_per_type caps spatial_lr_binary, spatial_ud_binary, and
    spatial_farthest (each capped independently, at most one question per
    distinct object pair / reference object). spatial_lr_extreme always
    produces both directions (leftmost and rightmost) unconditionally --
    there are only ever two, so there's nothing to cap. depth_closest is a
    single scene-global question (closest to the *camera*, not to another
    object), so it stays at one. None = no cap."""
    inst_list = list(instances.values())
    if len(inst_list) < 2:
        return []

    _add_position_proxies(instances, W, H)
    base = {"scene_id": scene_name, "kind": "mos", "phase": phase, "frame": fid, "mask_path": mask_path,
            "img_w": int(W), "img_h": int(H)}
    items = []

    # ---- left/right binary. Dedupe to unordered pairs first (a,b) and
    # (b,a) ask about the same real-world relationship, so sampling the cap
    # from ordered pairs would waste slots on near-duplicates -- the
    # question's left/right phrasing direction is picked per selected pair.
    lr_pairs = {}
    for a in inst_list:
        for b in inst_list:
            if a["mask_id"] == b["mask_id"] or a["dx_proxy"] is None or b["dx_proxy"] is None:
                continue
            if abs(a["dx_proxy"] - b["dx_proxy"]) / max(a["median_depth"], b["median_depth"]) < MIN_SEP_FRAC * W:
                continue
            key = frozenset((a["mask_id"], b["mask_id"]))
            lr_pairs.setdefault(key, (a, b))
    for a, b in _cap(list(lr_pairs.values()), max_per_type, rng):
        if rng.random() < 0.5:
            a, b = b, a
        items.append({
            **base, "type": "spatial_lr_binary",
            "question": f"Is the {a['name']} to the left of the {b['name']}?",
            "answer": "yes" if a["dx_proxy"] < b["dx_proxy"] else "no",
            "evidence": {"name_a": a["name"], "dx_proxy_a": round(a["dx_proxy"], 1),
                         "name_b": b["name"], "dx_proxy_b": round(b["dx_proxy"], 1)},
        })

    # ---- above/below binary, same dedupe-then-cap approach.
    ud_pairs = {}
    for a in inst_list:
        for b in inst_list:
            if a["mask_id"] == b["mask_id"] or a["dy_proxy"] is None or b["dy_proxy"] is None:
                continue
            if abs(a["dy_proxy"] - b["dy_proxy"]) / max(a["median_depth"], b["median_depth"]) < MIN_SEP_FRAC * H:
                continue
            key = frozenset((a["mask_id"], b["mask_id"]))
            ud_pairs.setdefault(key, (a, b))
    for a, b in _cap(list(ud_pairs.values()), max_per_type, rng):
        if rng.random() < 0.5:
            a, b = b, a
        items.append({
            **base, "type": "spatial_ud_binary",
            "question": f"Is the {a['name']} above the {b['name']}?",
            "answer": "yes" if a["dy_proxy"] < b["dy_proxy"] else "no",
            "evidence": {"name_a": a["name"], "dy_proxy_a": round(a["dy_proxy"], 1),
                         "name_b": b["name"], "dy_proxy_b": round(b["dy_proxy"], 1)},
        })

    # ---- leftmost AND rightmost (bbox task) -- both directions, always.
    proxy_candidates = [i for i in inst_list if i["dx_proxy"] is not None]
    if proxy_candidates:
        for direction in ("left", "right"):
            extreme = (min if direction == "left" else max)(proxy_candidates, key=lambda i: i["dx_proxy"])
            items.append({
                **base, "type": "spatial_lr_extreme",
                "question": f"Which object is furthest to the {direction}?",
                "answer": extreme["name"], "answer_bbox": _bbox_of_instance(extreme),
                "evidence": {i["name"]: round(i["dx_proxy"], 1) for i in proxy_candidates},
            })

    # ---- closest to camera (bbox task) -- pure depth ordering, single
    # scene-global question (the reference is the camera, not another
    # object, so there's nothing to vary this by).
    depth_vals = [(i["median_depth"], i) for i in inst_list if i["median_depth"] is not None]
    if depth_vals:
        closest = min(depth_vals, key=lambda x: x[0])[1]
        items.append({
            **base, "type": "depth_closest",
            "question": "Which object is closest to the camera?",
            "answer": closest["name"], "answer_bbox": _bbox_of_instance(closest),
            "evidence": {i["name"]: round(d, 1) for d, i in depth_vals},
        })

    # ---- farthest from a named reference object (bbox task) -- one
    # candidate per reference object, capped; kept only when the
    # depth-based 3D ranking agrees with the 2D pixel-distance ranking, so
    # this never publishes a perspective illusion as GT.
    if len(inst_list) >= 3 and depth_vals:
        farthest_candidates = []
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
            farthest_candidates.append((ref, ranked_2d[0], d2d, others))

        for ref, top, d2d, others in _cap(farthest_candidates, max_per_type, rng):
            items.append({
                **base, "type": "spatial_farthest",
                "question": f"Which object is farthest from the {ref['name']}?",
                "answer": top["name"], "answer_bbox": _bbox_of_instance(top),
                "evidence": {"reference": ref["name"],
                             "distances_2d_px": {o["name"]: round(d2d[o["mask_id"]], 1) for o in others}},
            })

    return items

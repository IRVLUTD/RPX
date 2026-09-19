"""
DINOv2/v3-based mask correspondence: ref crop → full target image.

For each ref mask (known object identity from mask_to_object.json):
  1. Crop the ref image to that mask's bounding box (+ padding), upscale to
     model_size×model_size.  Other objects in the crop stay for context.
  2. Filter the 1369 crop patches to only those whose centre falls on
     mask_idx pixels (dense, all within the object).
  3. Run the FULL target image through the model once → 1369 target patches.
  4. Match each ref query patch → best patch in the full target image.
  5. Record which target mask index each matched patch lands on.
  6. correctness = fraction of ref patches whose match lands on the expected
     target mask (same index).
     confidence  = mean cosine similarity of all matches.

This design avoids the "mask→mask" assumption: we search the whole target
scene for where the known ref object appears, then verify the SAM2 tracking.

Outputs under sam2/dino_output/:
  masked_rgb/      black-background RGB for every frame (full image)
  correspondence/  per-frame JSON
  palette_labeled/ palette + confidence labels
  keypoints/       full-frame side-by-side vis for every frame
  crop_masks/      per-mask: ref crop | full target with matched dots
                   ({stem}_mask{idx}.png)

Usage:
    conda run -n rkit-rpx python3 dino_correspondence.py \\
        --scene_dir <path/to/scene/phase> \\
        [--model facebook/dinov3-vitl16-pretrain-lvd1689m] \\
        [--threshold 0.5] \\
        [--max_kp_per_mask 20] \\
        [--crop_padding 20] \\
        [--hf_token hf_xxx]
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PALETTE_URL = "https://raw.githubusercontent.com/IRVLUTD/fewsol-toolkit/refs/heads/main/palette.txt"
_PALETTE = None

def get_palette():
    global _PALETTE
    if _PALETTE is None:
        print("Loading fewsol palette...")
        _PALETTE = [tuple(map(int, l.strip().split()))
                    for l in requests.get(_PALETTE_URL).text.strip().split("\n")]
    return _PALETTE


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_mask(path):
    return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def black_bg(rgb, mask):
    out = rgb.copy()
    out[mask == 0] = 0
    return out

def load_partitioned_frames(sam2_dir):
    """Return sorted list of frame stems that appear in verified_masks.txt."""
    vmt = sam2_dir / "verified_masks.txt"
    stems = []
    if not vmt.exists():
        return stems
    with open(vmt) as f:
        for line in f:
            s = line.strip()
            if s:
                stems.append(s.split(" | ")[0].strip())
    return sorted(set(stems))


def find_reference_frame(masks_dir, sam2_dir, n_objects, phase):
    """
    Find a reference frame that is:
      - not in the partitioned set
      - has exactly n_objects non-zero mask indices

    Phase "1"   → last such frame.
    Phase "0"/"2" → first such frame.
    """
    vmt = sam2_dir / "verified_masks.txt"
    partitioned = set()
    if vmt.exists():
        with open(vmt) as f:
            for line in f:
                s = line.strip()
                if s:
                    partitioned.add(s.split(" | ")[0].strip())

    good = [mp for mp in sorted(masks_dir.glob("*.png"))
            if mp.stem not in partitioned
            and len(set(np.unique(load_mask(mp)).tolist()) - {0}) == n_objects]

    if not good:
        all_masks = sorted(masks_dir.glob("*.png"))
        chosen = all_masks[-1] if phase == "1" else all_masks[0]
        print(f"  [ref frame] {chosen.name}  (phase={phase}, fallback static)")
        return chosen

    chosen = good[-1] if phase == "1" else good[0]
    print(f"  [ref frame] {chosen.name}  "
          f"(phase={phase}, {'last' if phase == '1' else 'first'} clean frame with all {n_objects} masks)")
    return chosen

def find_rgb_path(phase_dir, stem):
    for sub in ["rgb", os.path.join("frames", "rgb")]:
        for ext in [".png", ".jpg"]:
            p = phase_dir / sub / (stem + ext)
            if p.exists():
                return p
    return None


# ── DINOv2/v3 ────────────────────────────────────────────────────────────────

_FALLBACK_PROCESSOR = "facebook/dinov2-with-registers-large"

def load_dino(model_id, hf_token=None):
    print(f"Loading {model_id}...")
    kwargs = {"token": hf_token} if hf_token else {}
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
    except ValueError:
        # Model doesn't register its processor type (e.g. DINOv3).
        # Use the DINOv2-with-registers processor — same architecture.
        print(f"  [processor] AutoImageProcessor failed, using {_FALLBACK_PROCESSOR} processor")
        processor = AutoImageProcessor.from_pretrained(_FALLBACK_PROCESSOR)
    model = AutoModel.from_pretrained(model_id, **kwargs).to(DEVICE).eval()
    n_reg = getattr(model.config, "num_register_tokens", 0)
    if hasattr(processor, "do_resize"):      processor.do_resize = False
    if hasattr(processor, "do_center_crop"): processor.do_center_crop = False

    native = 518
    dummy  = Image.fromarray(np.zeros((native, native, 3), dtype=np.uint8))
    inp    = {k: v.to(DEVICE) for k, v in processor(images=dummy, return_tensors="pt").items()}
    with torch.no_grad():
        out = model(**inp)
    n_patches  = out.last_hidden_state.shape[1] - 1 - n_reg
    pps        = int(n_patches ** 0.5)
    patch_size = getattr(model.config, "patch_size", 14)
    model_size = pps * patch_size
    print(f"  patch_size={patch_size}, model_size={model_size} ({pps}×{pps} patches), register_tokens={n_reg}")
    return processor, model, model_size, n_reg, patch_size


@torch.no_grad()
def extract_features(pil_img, processor, model, model_size, n_reg):
    """Returns (N_patches, D) on CPU. Input is resized to model_size×model_size."""
    img = pil_img.resize((model_size, model_size), Image.BILINEAR)
    inp = {k: v.to(DEVICE) for k, v in processor(images=img, return_tensors="pt").items()}
    out = model(**inp)
    return out.last_hidden_state[0, 1 + n_reg:].float().cpu()


# ── Crop and patch helpers ────────────────────────────────────────────────────

def mask_bbox(mask_arr, mask_idx, padding):
    ys, xs = np.where(mask_arr == mask_idx)
    if len(xs) == 0:
        return None
    h, w = mask_arr.shape
    return (max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(w, int(xs.max()) + padding),
            min(h, int(ys.max()) + padding))


def all_objects_bbox(mask_arr, padding):
    """Union bounding box of all non-zero mask pixels, expanded by padding."""
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    h, w = mask_arr.shape
    return (max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(w, int(xs.max()) + padding),
            min(h, int(ys.max()) + padding))


def make_crop_with_context(rgb, mask_arr, bbox, model_size):
    """
    Crop to bbox and upscale to model_size×model_size.
    Background (mask==0) is zeroed out; other objects stay for context.
    Returns: pil_rgb, upscaled_mask (np uint16, nearest-neighbour).
    """
    x0, y0, x1, y1 = bbox
    rgb_crop  = rgb[y0:y1, x0:x1].copy()
    mask_crop = mask_arr[y0:y1, x0:x1]
    rgb_crop[mask_crop == 0] = 0          # black out pure background only
    pil_rgb  = Image.fromarray(rgb_crop).resize((model_size, model_size), Image.BILINEAR)
    up_mask  = np.array(
        Image.fromarray(mask_crop.astype(np.uint16)).resize(
            (model_size, model_size), Image.NEAREST))
    return pil_rgb, up_mask


def get_mask_patch_indices(up_mask, mask_idx, pps, patch_size):
    """
    Returns patch indices (0..pps*pps-1) whose centre pixel == mask_idx
    in the model_size upscaled mask.
    """
    valid = []
    for pi in range(pps * pps):
        py, px = divmod(pi, pps)
        cx = px * patch_size + patch_size // 2
        cy = py * patch_size + patch_size // 2
        if up_mask[cy, cx] == mask_idx:
            valid.append(pi)
    return valid


def patch_to_crop_px(patch_idx, pps, patch_size):
    """Patch centre in model_size crop pixels."""
    py, px = divmod(patch_idx, pps)
    return px * patch_size + patch_size // 2, py * patch_size + patch_size // 2


def patch_crop_to_orig(patch_idx, pps, patch_size, bbox):
    """Map a patch in the cropped model_size image back to original image coords."""
    x0, y0, x1, y1 = bbox
    cw, ch = x1 - x0, y1 - y0
    model_size = pps * patch_size
    py, px = divmod(patch_idx, pps)
    cx = x0 + int((px * patch_size + patch_size / 2) * cw / model_size)
    cy = y0 + int((py * patch_size + patch_size / 2) * ch / model_size)
    return cx, cy


def patch_full_to_orig(patch_idx, pps, patch_size, orig_hw):
    """Map a patch in the full model_size image back to original image coords."""
    h, w = orig_hw
    model_size = pps * patch_size
    py, px = divmod(patch_idx, pps)
    cx = int((px * patch_size + patch_size / 2) * w / model_size)
    cy = int((py * patch_size + patch_size / 2) * h / model_size)
    return cx, cy


# ── Correspondence: ref crop → full target ────────────────────────────────────

def compute_correspondences(ref_rgb, tgt_rgb,
                            ref_mask, tgt_mask,
                            mask_indices,
                            processor, model, model_size, n_reg, patch_size,
                            padding, max_kp_per_mask):
    """
    For each ref mask:
      - Crop ref to that mask's bbox, upscale → dense query features (in-mask only).
      - Compare against features from the FULL target image (one pass, shared).
      - Record where each query patch matched in the target, and which target
        mask it landed on.

    Returns:
      results  {mask_idx: {confidence, correctness, ref_pts, tgt_pts, sims,
                           ref_crop_pts, tgt_full_pts, match_correct}}
      crop_vis {mask_idx: (ref_crop_pil, tgt_full_pil)}
    """
    pps = model_size // patch_size
    rng = np.random.default_rng(42)

    # ── Target: crop to union bbox of all objects, black out background ──
    tgt_union_bbox = all_objects_bbox(tgt_mask, padding)
    if tgt_union_bbox is None:
        return {idx: dict(confidence=0.0, correctness=0.0, ref_pts=[], tgt_pts=[],
                          sims=[], ref_crop_pts=[], tgt_full_pts=[], match_correct=[])
                for idx in mask_indices}, {}

    tx0, ty0, tx1, ty1 = tgt_union_bbox
    tgt_rgb_crop  = tgt_rgb[ty0:ty1, tx0:tx1].copy()
    tgt_mask_crop = tgt_mask[ty0:ty1, tx0:tx1]
    tgt_rgb_crop[tgt_mask_crop == 0] = 0   # black out background

    tgt_pil     = Image.fromarray(tgt_rgb_crop).resize((model_size, model_size), Image.BILINEAR)
    tgt_mask_up = np.array(
        Image.fromarray(tgt_mask_crop.astype(np.uint16)).resize(
            (model_size, model_size), Image.NEAREST))
    tgt_feats_all = extract_features(tgt_pil, processor, model, model_size, n_reg)  # (pps², D)
    tgt_norm_all  = F.normalize(tgt_feats_all, dim=-1)

    results  = {}
    crop_vis = {}

    for idx in mask_indices:
        ref_bbox = mask_bbox(ref_mask, idx, padding)
        if ref_bbox is None:
            results[idx] = dict(confidence=0.0, correctness=0.0,
                                ref_pts=[], tgt_pts=[], sims=[],
                                ref_crop_pts=[], tgt_full_pts=[], match_correct=[])
            continue

        # Ref: crop, upscale, filter to in-mask patches
        ref_pil, ref_up_mask = make_crop_with_context(ref_rgb, ref_mask, ref_bbox, model_size)
        crop_vis[idx] = (ref_pil, tgt_pil)

        ref_valid = get_mask_patch_indices(ref_up_mask, idx, pps, patch_size)
        if not ref_valid:
            results[idx] = dict(confidence=0.0, correctness=0.0,
                                ref_pts=[], tgt_pts=[], sims=[],
                                ref_crop_pts=[], tgt_full_pts=[], match_correct=[])
            continue

        ref_feats = extract_features(ref_pil, processor, model, model_size, n_reg)[ref_valid]
        ref_norm  = F.normalize(ref_feats, dim=-1)

        sim            = torch.matmul(ref_norm, tgt_norm_all.T)  # (n_valid, pps²)
        best_tgt_patch = sim.argmax(dim=1).tolist()              # absolute patch idx in full tgt
        best_sims      = sim.max(dim=1).values

        # Which target mask did each match land on?
        matched_mask_val = []
        for tp in best_tgt_patch:
            py, px = divmod(tp, pps)
            cy = min(py * patch_size + patch_size // 2, model_size - 1)
            cx = min(px * patch_size + patch_size // 2, model_size - 1)
            matched_mask_val.append(int(tgt_mask_up[cy, cx]))

        n_ref      = len(ref_valid)
        confidence = float(best_sims.mean())

        # Vote: which target mask got the most queries?
        votes = {}
        for mv in matched_mask_val:
            if mv > 0:
                votes[mv] = votes.get(mv, 0) + 1
        assignment    = max(votes, key=votes.get) if votes else None
        assign_frac   = (votes[assignment] / n_ref) if assignment else 0.0

        sel = rng.choice(n_ref, min(max_kp_per_mask, n_ref), replace=False)

        ref_pts           = [patch_crop_to_orig(ref_valid[i],       pps, patch_size, ref_bbox)       for i in sel]
        tgt_pts           = [patch_crop_to_orig(best_tgt_patch[i],  pps, patch_size, tgt_union_bbox) for i in sel]
        sims              = [float(best_sims[i])                                                      for i in sel]
        ref_crop_pts      = [patch_to_crop_px(ref_valid[i],         pps, patch_size)                 for i in sel]
        tgt_full_pts      = [patch_to_crop_px(best_tgt_patch[i],    pps, patch_size)                 for i in sel]
        sel_matched_vals  = [matched_mask_val[i]                                                      for i in sel]
        match_correct     = [mv == assignment for mv in sel_matched_vals]

        results[idx] = dict(
            confidence        = round(confidence, 4),
            assign_frac       = round(assign_frac, 4),
            assignment        = assignment,
            votes             = votes,
            ref_pts           = ref_pts,
            tgt_pts           = tgt_pts,
            sims              = sims,
            ref_crop_pts      = ref_crop_pts,
            tgt_full_pts      = tgt_full_pts,
            match_correct     = match_correct,
            sel_matched_vals  = sel_matched_vals,   # raw per-point votes, for post-resolve fixup
        )

    return results, crop_vis


# ── Assignment resolution ─────────────────────────────────────────────────────

def _greedy_resolve(ref_indices, votes_per_ref):
    """
    Greedy conflict-free assignment.
    votes_per_ref: {ref_idx: {tgt_mask: weight}}
    Returns {ref_idx: assigned_tgt_mask or None}.
    """
    total_weight = {r: max(votes_per_ref.get(r, {}).values(), default=0.0)
                    for r in ref_indices}
    sorted_refs = sorted(ref_indices, key=lambda r: total_weight[r], reverse=True)
    taken, result = set(), {}
    for ref_idx in sorted_refs:
        votes = votes_per_ref.get(ref_idx, {})
        best = None
        for tgt in sorted(votes, key=votes.get, reverse=True):
            if tgt not in taken:
                best = tgt
                break
        result[ref_idx] = best
        if best is not None:
            taken.add(best)
    return result


def resolve_assignments(corr):
    """
    Per-frame conflict-free greedy assignment (in-place).
    Ensures no two ref masks claim the same target mask for a single frame.
    """
    votes_per_ref = {ref_idx: d.get("votes", {}) for ref_idx, d in corr.items()}
    result = _greedy_resolve(list(corr.keys()), votes_per_ref)

    for ref_idx, new_asgn in result.items():
        d = corr[ref_idx]
        votes = d.get("votes", {})
        d["assignment"] = new_asgn
        total = sum(votes.values()) or 1
        d["assign_frac"] = round(votes.get(new_asgn, 0) / total, 4) if new_asgn else 0.0
        d["match_correct"] = [mv == new_asgn for mv in d.get("sel_matched_vals", [])]


def compute_canonical_assignments(corr_dir, ref_mask_indices):
    """
    Read all per-frame correspondence JSONs, aggregate assign_frac-weighted votes
    for each ref_idx across the whole phase, then resolve conflicts globally.
    Returns {ref_idx: canonical_tgt_mask_idx} — one consistent assignment per object
    for the entire phase.
    """
    from collections import defaultdict
    raw = defaultdict(lambda: defaultdict(float))  # ref_idx → {tgt: total_weight}

    for json_path in sorted(corr_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for ref_str, entry in data.items():
            asgn = entry.get("assignment")
            frac = entry.get("assign_frac", 0.0)
            if asgn is not None:
                raw[int(ref_str)][asgn] += frac

    votes_per_ref = {r: dict(raw[r]) for r in ref_mask_indices}
    return _greedy_resolve(ref_mask_indices, votes_per_ref)


# ── Visualisation ─────────────────────────────────────────────────────────────

_OBJ_COLORS_BGR = [
    (  0,   0, 255),  # 1 – red
    (  0, 200,   0),  # 2 – green
    (255,   0,   0),  # 3 – blue
    (  0, 220, 220),  # 4 – yellow
    (220,   0, 220),  # 5 – magenta
    (255, 200,   0),  # 6 – cyan
    (128,   0, 255),  # 7 – purple
    (  0, 128, 255),  # 8 – orange
]


def draw_keypoint_vis(ref_rgb_masked, tgt_rgb_masked, corr, ref_object_map, threshold,
                      ref_label=None):
    """Full-frame side-by-side: ref (masked) | target (masked), with correspondence lines."""
    h1, w1 = ref_rgb_masked.shape[:2]
    h2, w2 = tgt_rgb_masked.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = ref_rgb_masked
    canvas[:h2, w1:] = tgt_rgb_masked

    for mask_idx, data in corr.items():
        if not data["ref_pts"]:
            continue
        obj_color = _OBJ_COLORS_BGR[(mask_idx - 1) % len(_OBJ_COLORS_BGR)]
        max_s = max(data["sims"]) if data["sims"] else 1.0

        for (x1, y1), (x2, y2), s, ok in zip(
                data["ref_pts"], data["tgt_pts"], data["sims"], data["match_correct"]):
            alpha      = 0.25 + 0.75 * (s / max(max_s, 1e-6))
            line_color = tuple(int(c * alpha) for c in obj_color)
            dot_color  = obj_color if ok else (0, 0, 128)   # dim red if wrong mask

            cv2.circle(canvas, (x1, y1),      5, obj_color,  -1)
            cv2.circle(canvas, (x1, y1),      5, (255,255,255), 1)
            cv2.circle(canvas, (x2 + w1, y2), 5, dot_color,  -1)
            cv2.circle(canvas, (x2 + w1, y2), 5, (255,255,255), 1)
            cv2.line(canvas, (x1, y1), (x2 + w1, y2), line_color, 1, lineType=cv2.LINE_AA)

    y_leg = 40
    for mask_idx in sorted(corr.keys()):
        obj_color = _OBJ_COLORS_BGR[(mask_idx - 1) % len(_OBJ_COLORS_BGR)]
        name  = ref_object_map.get(mask_idx, {}).get("name", f"mask{mask_idx}")
        data  = corr[mask_idx]
        asgn  = data.get("assignment", "?")
        label = (f"ref{mask_idx}:{name} → tgt{asgn}  "
                 f"cos={data['confidence']:.0%}  vote={data['assign_frac']:.0%}")
        cv2.putText(canvas, label, (8, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 3)
        cv2.putText(canvas, label, (8, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.40, obj_color, 1)
        y_leg += 18

    ref_title = f"REFERENCE [{ref_label}]" if ref_label else "REFERENCE"
    cv2.putText(canvas, ref_title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(canvas, "TARGET  (bright = voted for assigned mask)",
                (w1 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)
    return canvas


def draw_mask_crop_vis(ref_pil, tgt_pil, data, obj_color_bgr, mask_name):
    """
    Left: ref crop (zoomed in on the mask), with query dots.
    Right: full target image (model_size), with matched dots.
    Bright dot = landed on expected target mask. Dark red = wrong mask.
    """
    ref_np = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
    tgt_np = cv2.cvtColor(np.array(tgt_pil), cv2.COLOR_RGB2BGR)
    h, w   = ref_np.shape[:2]
    gap    = 4
    canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]      = ref_np
    canvas[:, w+gap:]  = tgt_np

    ref_crop_pts = data.get("ref_crop_pts", [])
    tgt_full_pts = data.get("tgt_full_pts", [])
    sims         = data.get("sims", [])
    match_correct= data.get("match_correct", [])

    max_s = max(sims) if sims else 1.0
    for (x1, y1), (x2, y2), s, ok in zip(ref_crop_pts, tgt_full_pts, sims, match_correct):
        alpha      = 0.25 + 0.75 * (s / max(max_s, 1e-6))
        line_color = tuple(int(c * alpha) for c in obj_color_bgr)
        dot_color  = obj_color_bgr if ok else (0, 0, 128)

        cv2.circle(canvas, (x1, y1),           5, obj_color_bgr, -1)
        cv2.circle(canvas, (x1, y1),           5, (255,255,255), 1)
        cv2.circle(canvas, (x2 + w + gap, y2), 5, dot_color,     -1)
        cv2.circle(canvas, (x2 + w + gap, y2), 5, (255,255,255), 1)
        cv2.line(canvas, (x1, y1), (x2 + w + gap, y2), line_color, 1, lineType=cv2.LINE_AA)

    conf  = data.get("confidence", 0.0)
    afrac = data.get("assign_frac", 0.0)
    asgn  = data.get("assignment", "?")
    cv2.putText(canvas, f"REF crop [{mask_name}]",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
    cv2.putText(canvas, f"REF crop [{mask_name}]",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(canvas, f"TGT union  → mask{asgn}  cos={conf:.0%}  vote={afrac:.0%}",
                (w + gap + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
    cv2.putText(canvas, f"TGT union  → mask{asgn}  cos={conf:.0%}  vote={afrac:.0%}",
                (w + gap + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return canvas


def load_frame_hidden_masks(sam2_dir):
    """
    Parse verified_masks.txt → {frame_stem: set_of_hidden_pixel_values}.
    IDs in the file are 0-indexed; pixel values = ID + 1.
    """
    vmt = sam2_dir / "verified_masks.txt"
    result = {}
    if not vmt.exists():
        return result
    with open(vmt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "| hidden:" in line:
                parts = line.split("| hidden:")
                stem   = parts[0].strip()
                hidden = set(int(x.strip()) + 1
                             for x in parts[1].strip().split(",")
                             if x.strip().isdigit())
            else:
                stem   = line.strip()
                hidden = set()
            result[stem] = hidden
    return result


def _build_tgt_to_ref(corr):
    """Build target_mask_idx → (ref_mask_idx, confidence, assign_frac) from corr dict."""
    tgt_to_ref = {}
    for ref_idx, data in corr.items():
        asgn = data.get("assignment")
        if asgn is not None:
            tgt_to_ref[asgn] = (ref_idx, data.get("confidence", 0.0), data.get("assign_frac", 0.0))
    return tgt_to_ref


def draw_palette_labeled(mask_arr, corr, ref_object_map, threshold, hidden_pixel_vals=None):
    """
    Color each target mask region using the ASSIGNED ref object's palette index.
    hidden_pixel_vals: set of mask pixel values to leave black and unlabeled.
    """
    hidden_pixel_vals = hidden_pixel_vals or set()
    pal       = get_palette()
    tgt_to_ref = _build_tgt_to_ref(corr)

    h, w = mask_arr.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for tgt_idx in np.unique(mask_arr):
        if tgt_idx == 0 or tgt_idx in hidden_pixel_vals:
            continue
        if tgt_idx in tgt_to_ref:
            ref_idx, _, _ = tgt_to_ref[tgt_idx]
            color = pal[int(ref_idx) % len(pal)]
        else:
            color = pal[int(tgt_idx) % len(pal)]
        out[mask_arr == int(tgt_idx)] = color

    img  = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for tgt_idx in np.unique(mask_arr):
        if tgt_idx == 0 or tgt_idx in hidden_pixel_vals:
            continue
        ys, xs = np.where(mask_arr == int(tgt_idx))
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        if tgt_idx in tgt_to_ref:
            ref_idx, conf, afrac = tgt_to_ref[tgt_idx]
            obj   = ref_object_map.get(ref_idx, {"name": f"mask{ref_idx}"})
            label = f"{obj['name']}\ncos={conf:.0%} / vote={afrac:.0%}"
            color = "yellow" if conf >= threshold and afrac >= threshold else "orange"
        else:
            label = f"tgt{tgt_idx}\n(unassigned)"
            color = "red"
        for dx, dy in [(-1, -1), (1, 1)]:
            draw.text((cx + dx, cy + dy), label, font=font, fill="black", anchor="mm")
        draw.text((cx, cy), label, font=font, fill=color, anchor="mm")

    return img



def draw_palette_chain(mask_arr, chain_assign, ref_object_map, hidden_pixel_vals=None,
                       canonical_fracs=None, threshold=0.5):
    """
    Render palette image using chain_assign: {ref_idx: assigned_tgt_mask_idx}.
    Regions not covered by any assignment keep their original palette colour.
    hidden_pixel_vals: left black.
    canonical_fracs: {ref_idx: float} — fraction of frames agreeing with canonical.
      Labels are yellow if >= threshold, orange otherwise.
    """
    hidden_pixel_vals = hidden_pixel_vals or set()
    canonical_fracs   = canonical_fracs or {}
    pal = get_palette()

    # Invert: tgt_mask_idx → ref_idx
    tgt_to_ref = {tgt: ref for ref, tgt in chain_assign.items() if tgt is not None}

    h, w = mask_arr.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for tgt_idx in np.unique(mask_arr):
        if tgt_idx == 0 or tgt_idx in hidden_pixel_vals:
            continue
        ref_idx = tgt_to_ref.get(int(tgt_idx))
        color   = pal[int(ref_idx) % len(pal)] if ref_idx is not None \
                  else pal[int(tgt_idx) % len(pal)]
        out[mask_arr == int(tgt_idx)] = color

    img  = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for tgt_idx in np.unique(mask_arr):
        if tgt_idx == 0 or tgt_idx in hidden_pixel_vals:
            continue
        ys, xs = np.where(mask_arr == int(tgt_idx))
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        ref_idx = tgt_to_ref.get(int(tgt_idx))
        if ref_idx is not None:
            obj   = ref_object_map.get(ref_idx, {"name": f"mask{ref_idx}"})
            cfrac = canonical_fracs.get(ref_idx, 1.0)
            label = f"{obj['name']}\ncanon={cfrac:.0%}"
            color = "yellow" if cfrac >= threshold else "orange"
        else:
            label = f"tgt{tgt_idx}\n(unassigned)"
            color = "red"
        for dx, dy in [(-1, -1), (1, 1)]:
            draw.text((cx + dx, cy + dy), label, font=font, fill="black", anchor="mm")
        draw.text((cx, cy), label, font=font, fill=color, anchor="mm")

    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def run(scene_phase_dir, model_id, threshold, max_kp_per_mask, crop_padding, hf_token=None, ref_stem=None, use_verified_ref=False):
    scene_phase_dir = Path(scene_phase_dir)
    sam2_dir  = scene_phase_dir / "sam2"
    masks_dir = sam2_dir / "masks"

    out_root        = sam2_dir / "dino_output"
    masked_dir      = out_root / "masked_rgb"
    corr_dir        = out_root / "correspondence"
    pal_dir         = out_root / "palette_labeled"
    pal_hidden_dir  = out_root / "palette_labeled_hidden"
    pal_chain_dir   = out_root / "palette_chain"
    kp_dir          = out_root / "keypoints"
    crop_dir        = out_root / "crop_masks"
    final_dir       = out_root / "final_assignment"
    for d in [masked_dir, corr_dir, pal_dir, pal_hidden_dir, pal_chain_dir, kp_dir, crop_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(sam2_dir / "mask_to_object.json") as f:
        m2o = json.load(f)
    n_objects      = len(m2o)
    ref_object_map = {int(k): v["object"] for k, v in m2o.items()}
    print(f"Scene  : {scene_phase_dir}")
    print(f"Objects: {[v['name'] for v in ref_object_map.values()]}")

    phase = scene_phase_dir.name   # "0", "1", or "2"

    # Object indices come from mask_to_object.json — always complete
    ref_mask_indices = sorted(ref_object_map.keys())

    if ref_stem is not None:
        ref_mask_path = masks_dir / f"{ref_stem}.png"
        if not ref_mask_path.exists():
            raise FileNotFoundError(f"--ref_stem {ref_stem}: {ref_mask_path} not found")
        print(f"  [ref frame] {ref_stem}  (forced via --ref_stem)")
    elif use_verified_ref:
        # Masks are in ref-state — pick a verified frame that has all objects visible
        vmt = sam2_dir / "verified_masks.txt"
        verified_stems = []
        if vmt.exists():
            for line in vmt.read_text().splitlines():
                s = line.strip()
                if s:
                    verified_stems.append(s.split(" | ")[0].strip())
        good = [masks_dir / f"{s}.png" for s in verified_stems
                if (masks_dir / f"{s}.png").exists()
                and len(set(np.unique(load_mask(masks_dir / f"{s}.png")).tolist()) - {0}) == n_objects]
        if not good:
            good = [masks_dir / f"{s}.png" for s in verified_stems if (masks_dir / f"{s}.png").exists()]
        ref_mask_path = good[-1] if phase == "1" else good[0]
        print(f"  [ref frame] {ref_mask_path.stem}  (verified ref-state frame)")
    else:
        ref_mask_path = find_reference_frame(masks_dir, sam2_dir, n_objects, phase)
    ref_stem       = ref_mask_path.stem
    ref_mask       = load_mask(ref_mask_path)
    ref_rgb_path   = find_rgb_path(scene_phase_dir, ref_stem)
    ref_rgb        = load_rgb(ref_rgb_path) if ref_rgb_path else \
                     np.zeros((*ref_mask.shape, 3), dtype=np.uint8)
    ref_rgb_masked = black_bg(ref_rgb, ref_mask)

    processor, model, model_size, n_reg, patch_size = load_dino(model_id, hf_token)
    get_palette()

    # Only run correspondence on partitioned frames (others are already clean)
    partitioned_stems = set(load_partitioned_frames(sam2_dir))
    all_frames = [p for p in sorted(masks_dir.glob("*.png"))
                  if p.stem in partitioned_stems]
    total_frames = len(list(masks_dir.glob("*.png")))
    print(f"  Partitioned frames to check: {len(all_frames)} / {total_frames}")

    # Hidden mask info for palette_labeled_hidden output
    frame_hidden = load_frame_hidden_masks(sam2_dir)

    flagged = []

    for mask_path in tqdm(all_frames, desc="Frames"):
        stem         = mask_path.stem
        tgt_mask     = load_mask(mask_path)
        tgt_rgb_path = find_rgb_path(scene_phase_dir, stem)
        tgt_rgb      = load_rgb(tgt_rgb_path) if tgt_rgb_path else \
                       np.zeros((*tgt_mask.shape, 3), dtype=np.uint8)
        tgt_rgb_masked = black_bg(tgt_rgb, tgt_mask)

        Image.fromarray(tgt_rgb_masked).save(masked_dir / f"{stem}.png")

        if not (set(np.unique(tgt_mask).tolist()) - {0}):
            continue

        corr, crop_vis = compute_correspondences(
            ref_rgb, tgt_rgb,
            ref_mask, tgt_mask,
            ref_mask_indices,
            processor, model, model_size, n_reg, patch_size,
            crop_padding, max_kp_per_mask,
        )
        resolve_assignments(corr)   # conflict-free greedy reassignment (in-place)

        any_flagged = any(
            d["confidence"] < threshold or d["assign_frac"] < threshold
            for d in corr.values()
        )
        if any_flagged:
            flagged.append(stem)

        json_out = {
            idx: {
                "object":      ref_object_map.get(idx, {"id":"?","name":f"mask{idx}"}),
                "confidence":  d["confidence"],   # mean cosine similarity
                "assign_frac": d["assign_frac"],  # fraction of queries voting for winner
                "assignment":  d["assignment"],   # which target mask this object was found in
                "votes":       {str(k): v for k, v in d.get("votes", {}).items()},
                "flagged":     d["confidence"] < threshold or d["assign_frac"] < threshold,
            }
            for idx, d in corr.items()
        }
        with open(corr_dir / f"{stem}.json", "w") as f:
            json.dump(json_out, f, indent=2)

        hidden_vals = frame_hidden.get(stem, set())

        draw_palette_labeled(tgt_mask, corr, ref_object_map, threshold).save(
            pal_dir / f"{stem}.png")

        draw_palette_labeled(tgt_mask, corr, ref_object_map, threshold,
                             hidden_pixel_vals=hidden_vals).save(
            pal_hidden_dir / f"{stem}.png")

        vis = draw_keypoint_vis(ref_rgb_masked, tgt_rgb_masked, corr, ref_object_map, threshold,
                                ref_label=ref_stem)
        cv2.imwrite(str(kp_dir / f"{stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        for mask_idx, (ref_pil, tgt_pil) in crop_vis.items():
            data      = corr.get(mask_idx, {})
            obj_color = _OBJ_COLORS_BGR[(mask_idx - 1) % len(_OBJ_COLORS_BGR)]
            name      = ref_object_map.get(mask_idx, {}).get("name", f"mask{mask_idx}")
            crop_canvas = draw_mask_crop_vis(ref_pil, tgt_pil, data, obj_color, name)
            cv2.imwrite(str(crop_dir / f"{stem}_mask{mask_idx}.png"), crop_canvas)

    print(f"\n=== Done (correspondence pass) ===")
    print(f"  Frames   : {len(all_frames)}")
    print(f"  Flagged (<{threshold:.0%}): {len(flagged)}")
    if flagged:
        print(f"  First    : {flagged[:5]}")
    print(f"  Output → {out_root}")

    # ── Global canonical chain: majority-vote across all frames ───────────────
    print("\nComputing global canonical assignments...")
    canonical = compute_canonical_assignments(corr_dir, ref_mask_indices)
    canon_readable = {ref_object_map.get(r, {}).get("name", f"ref{r}"): t
                      for r, t in canonical.items()}
    print(f"  Canonical mapping (ref object → target mask): {canon_readable}")

    # ── Update JSONs with canonical_assign_frac and canonical_match ───────────
    print("Updating JSONs with canonical metrics...")
    for json_path in sorted(corr_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for ref_str, entry in data.items():
            ref_idx   = int(ref_str)
            canon_tgt = canonical.get(ref_idx)
            votes_raw = entry.get("votes", {})
            votes_int = {int(k): v for k, v in votes_raw.items()}
            total     = sum(votes_int.values()) or 1
            canon_frac = round(votes_int.get(canon_tgt, 0) / total, 4) \
                         if canon_tgt is not None else 0.0
            entry["canonical_assignment"]  = canon_tgt
            entry["canonical_assign_frac"] = canon_frac
            entry["canonical_match"]       = (entry.get("assignment") == canon_tgt)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Re-render palette_chain/ with per-frame canonical_fracs ───────────────
    for mask_path in tqdm(all_frames, desc="Chain render"):
        stem      = mask_path.stem
        tgt_mask  = load_mask(mask_path)
        hidden_vals = frame_hidden.get(stem, set())

        canonical_fracs = {}
        jpath = corr_dir / f"{stem}.json"
        if jpath.exists():
            with open(jpath) as f:
                fd = json.load(f)
            for ref_str, entry in fd.items():
                canonical_fracs[int(ref_str)] = entry.get("canonical_assign_frac", 1.0)

        draw_palette_chain(tgt_mask, canonical, ref_object_map,
                           hidden_pixel_vals=hidden_vals,
                           canonical_fracs=canonical_fracs,
                           threshold=threshold).save(
            pal_chain_dir / f"{stem}.png")

    # ── final_assignment/: per-frame DINO assignment for ALL frames ───────────
    # Every frame now has a correspondence JSON so all get per-frame labels.
    # Canonical fallback is kept only as a safety net for missing JSONs.
    print("\nRendering final_assignment/ for all frames...")
    all_mask_files = sorted(masks_dir.glob("*.png"))
    for mask_path in tqdm(all_mask_files, desc="Final"):
        stem     = mask_path.stem
        tgt_mask = load_mask(mask_path)

        jpath = corr_dir / f"{stem}.json"
        if jpath.exists():
            # Partitioned frame: reconstruct corr dict from JSON for per-frame draw
            with open(jpath) as f:
                fd = json.load(f)
            # Build a minimal corr-like dict that draw_palette_labeled understands
            frame_corr = {}
            for ref_str, entry in fd.items():
                frame_corr[int(ref_str)] = {
                    "assignment":  entry.get("assignment"),
                    "confidence":  entry.get("confidence", 0.0),
                    "assign_frac": entry.get("assign_frac", 0.0),
                }
            draw_palette_labeled(tgt_mask, frame_corr, ref_object_map,
                                 threshold=threshold,
                                 hidden_pixel_vals=set()).save(
                final_dir / f"{stem}.png")
        else:
            # Non-partitioned frame: mask indices ARE the correct assignment
            # per mask_to_object.json — render directly with identity mapping.
            identity_corr = {
                ref_idx: {"assignment": ref_idx, "confidence": 1.0, "assign_frac": 1.0}
                for ref_idx in ref_mask_indices
            }
            draw_palette_labeled(tgt_mask, identity_corr, ref_object_map,
                                 threshold=threshold,
                                 hidden_pixel_vals=set()).save(
                final_dir / f"{stem}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",       required=True)
    p.add_argument("--model",           default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--threshold",       type=float, default=0.5)
    p.add_argument("--max_kp_per_mask", type=int,   default=20)
    p.add_argument("--crop_padding",    type=int,   default=20)
    p.add_argument("--hf_token",           default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--ref_stem",           default=None,
                   help="Force a specific frame stem as the DINO reference frame")
    p.add_argument("--use_verified_ref",   action="store_true",
                   help="Pick a verified (ref-state) frame as reference instead of a non-verified one")
    args = p.parse_args()
    run(args.scene_dir, args.model, args.threshold, args.max_kp_per_mask,
        args.crop_padding, args.hf_token,
        ref_stem=args.ref_stem, use_verified_ref=args.use_verified_ref)

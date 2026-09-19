"""Attribute VQA ground truth: "which object is X" style questions built
from each visible object's real questionnaire attributes (name/synonyms,
material, function, color).

Every type below only fires when its answer is provably unique for that
frame -- if a value is shared by more than one visible object, the question
is dropped rather than kept with an ambiguous answer. That guarantee is the
whole point of this generator; weakening it anywhere is a correctness bug,
not a style choice.

Types generated:
  attr_single_color     "Which object is [color]?"
  attr_single_material  "Which object is made of [material]?"
  attr_single_function  "Which object is used for [function]?"
  attr_composition      "What is the object made of [material] that is used
                          for [function]?" -- only when that exact
                          (material, function) pair belongs to one object.
  attr_odd_one_out      "Which object is NOT made of [material]?" -- only
                          when exactly one visible object lacks a material
                          every other visible object shares.
  attr_count            "How many objects are made of [material]?" -- not an
                          identity question, so uniqueness doesn't apply the
                          same way; the count itself is unambiguous by
                          construction (it's just len() of a real set).
  attr_synonym          "Is there a [name] in the scene?" -- yes/no, with a
                          genuine "no" counterpart (see below). This
                          replaces an earlier version whose ground truth was
                          always "yes", which made it unfalsifiable: a model
                          that reflexively answers yes scored 100% whether
                          or not it recognized anything.
  attr_absent            "Is there a [color] object in the scene?" (no) --
                          real distractor value from the wider catalog,
                          genuinely absent from this frame.

Every fact above is a property of the object itself, not of the camera
viewpoint -- confirmed: within one phase (a camera sweep around one static
arrangement) the same fact stays true across every frame. That makes
independent per-frame sampling wasteful (measured up to 96.7% repeat rate on
attr_count in one phase) -- see collect_candidates() and dedup.py for the
phase-wide fix. This is NOT true of the spatial tasks (gt_spatial.py):
left/right, closest, and farthest genuinely change with camera viewpoint
(confirmed: 90% of object pairs flipped spatial_lr_binary's answer at least
once across a single phase), so those are deliberately left un-deduped.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from lib import object_attrs
from generate_spatial_gt import MIN_MASK_AREA_PX

ALL_TYPES = (
    "attr_single_color", "attr_single_material", "attr_single_function",
    "attr_composition", "attr_odd_one_out", "attr_count",
    "attr_synonym", "attr_absent",
)

# attr_synonym's yes/no candidates are collected under these two SEPARATE
# bucket names (not "attr_synonym") so the phase distributor caps each
# independently at 1 -- a shared bucket could otherwise place e.g. 2 "yes"
# and 0 "no" in some frame, purely from whichever key had the lighter load.
# Every emitted item's own "type" field still says "attr_synonym"; the
# split only affects internal bookkeeping. See generate_gt.py.
CANDIDATE_BUCKETS = ALL_TYPES[:-2] + ("attr_synonym_yes", "attr_synonym_no", "attr_absent")


def bbox_of(mask: np.ndarray, oid: int):
    ys, xs = np.where(mask == oid)
    assert len(xs) > 0, (
        f"object id {oid} has no mask pixels in this frame -- a question is about "
        "to reference an object that isn't actually visible here. This should be "
        "unreachable: every oid a question can answer with is drawn from this "
        "exact frame's present_ids, computed from this exact mask array."
    )
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _all_names(a: dict) -> set:
    """Every recorded name/synonym for one object, lowercased, for
    ambiguity checks against other visible objects."""
    return {n.lower() for n in a.get("name", [])}


def build_folder_to_fewsol(lookup: dict) -> dict:
    """lookup is fewsol_id -> folder path; build the reverse map once per
    run (not per frame -- this used to be an O(catalog size) linear scan
    repeated inside every frame's candidate collection)."""
    out = {}
    for fewsol_id, folder in lookup.items():
        out[folder.rsplit("/", 1)[-1]] = fewsol_id
    return out


def collect_candidates(scene_name, kind, phase, fid, mask_path, mapping,
                        lookup, sos_catalog_names, folder_to_fewsol, rng,
                        distractor_scan=40):
    """All valid, unique-answer candidates for this ONE frame -- uncapped,
    unchosen. Returns dict[type_name -> list[(key, item)]]; `key` identifies
    the underlying fact (shared across frames when the same fact is valid in
    more than one), for dedup.py's phase-wide distribution to consume.
    attr_synonym's yes/no candidates are tagged with distinguishable keys
    but share the "attr_synonym" type bucket -- the caller is responsible
    for capping yes and no separately (1 each) so the type doesn't end up
    all-yes or all-no in some frame; see generate_gt.py.

    mapping: {mask_id: {"name": str, "oid": fewsol_id str}}. lookup:
    fewsol_id -> SOS folder path. sos_catalog_names / folder_to_fewsol:
    the wider catalog, for attr_synonym's "no" case and attr_absent's
    distractor pool."""
    mask = np.array(Image.open(mask_path))
    # MIN_MASK_AREA_PX excludes 1-2 pixel mask fragments (segmentation noise,
    # not a meaningfully visible object) -- same floor gt_spatial.py's
    # compute_instances() applies, kept in sync via the shared constant.
    present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping
                   and int((mask == v).sum()) >= MIN_MASK_AREA_PX]
    out = {t: [] for t in CANDIDATE_BUCKETS}
    if len(present_ids) < 2:
        return out

    attrs = {}
    for oid in present_ids:
        a = object_attrs(mapping[oid]["oid"], lookup)
        if a:
            attrs[oid] = a
    if len(attrs) < 2:
        return out

    img_h, img_w = mask.shape[:2]
    base = {"scene_id": scene_name, "kind": kind, "phase": phase, "frame": fid, "mask_path": mask_path,
            "img_w": int(img_w), "img_h": int(img_h)}

    def bbox(oid):
        return bbox_of(mask, oid)

    def obj_name(oid):
        return attrs[oid]["name"][0] if attrs[oid]["name"] else mapping[oid]["name"]

    # ---- single-attribute grounding: color, material, function.
    # Guard: only kept when exactly one visible object has this value.
    for field, verb in [("color", "is"), ("material", "made of"), ("function", "used for")]:
        owners = {}
        for oid, a in attrs.items():
            for val in a[field]:
                owners.setdefault(val, []).append(oid)
        type_name = f"attr_single_{field}"
        for val, ov in owners.items():
            if len(ov) != 1:
                continue
            oid = ov[0]
            q = (f"Which object is {val}?" if field == "color"
                 else f"Which object is {verb} {val}?")
            item = {**base, "type": type_name, "question": q,
                    "answer": obj_name(oid), "answer_bbox": bbox(oid),
                    "attr_value": val, "target_oid": mapping[oid]["oid"]}
            out[type_name].append((val, item))

    # ---- composition (material + function). Guard: the (material,
    # function) pair must belong to exactly one visible object.
    pair_owners = {}
    for oid, a in attrs.items():
        for m in a["material"]:
            for f in a["function"]:
                pair_owners.setdefault((m, f), []).append(oid)
    for (m, f), ov in pair_owners.items():
        if len(ov) != 1:
            continue
        oid = ov[0]
        item = {**base, "type": "attr_composition",
                "question": f"What is the object made of {m} that is used for {f}?",
                "answer": obj_name(oid), "answer_bbox": bbox(oid),
                "target_oid": mapping[oid]["oid"]}
        out["attr_composition"].append(((m, f), item))

    # ---- synonym recognition, "yes" case: a real, less-common synonym of a
    # visible object, kept only if that synonym string doesn't also appear
    # in any OTHER visible object's own name list (would make the yes/no
    # answer correct but the underlying reference ambiguous).
    present_name_sets = {oid: _all_names(a) for oid, a in attrs.items()}
    for oid, a in attrs.items():
        names = a["name"]
        if len(names) < 2:
            continue
        for alt in names[1:]:
            alt_l = alt.lower()
            if any(alt_l in present_name_sets[other] for other in attrs if other != oid):
                continue
            item = {**base, "type": "attr_synonym",
                    "question": f"Is there a {alt} in the scene?",
                    "answer": "yes", "answer_bbox": bbox(oid),
                    "canonical_name": obj_name(oid), "target_oid": mapping[oid]["oid"]}
            out["attr_synonym_yes"].append(((oid, alt), item))

    # ---- synonym "no" case + absence probe: scan a shuffled slice of the
    # wider catalog per frame for real values genuinely absent here -- same
    # "genuinely absent, not a made-up trap" standard either way.
    all_present_names = set().union(*present_name_sets.values()) if present_name_sets else set()
    present_colors = {v for a in attrs.values() for v in a["color"]}
    seen_names_this_frame = set()
    seen_colors_this_frame = set()
    scan_pool = list(sos_catalog_names)
    rng.shuffle(scan_pool)
    for cand_folder in scan_pool[:distractor_scan]:
        cand_fewsol = folder_to_fewsol.get(cand_folder)
        if cand_fewsol is None:
            continue
        cand_attrs = object_attrs(cand_fewsol, lookup)
        if not cand_attrs:
            continue
        # Dedup key must match what the QUESTION actually depends on, not
        # which catalog object it came from -- two different catalog
        # objects sharing the same primary name or color produce the exact
        # same question text, and treating them as different "facts" let
        # both get placed into the same frame (measured: 637 literal
        # duplicate rows from this before the fix). seen_this_frame also
        # skips a second candidate for an already-seen value within this
        # SAME frame, so redundant candidates aren't even collected.
        if cand_attrs["name"]:
            cand_name = cand_attrs["name"][0]
            key = cand_name.lower()
            if key not in all_present_names and key not in seen_names_this_frame:
                seen_names_this_frame.add(key)
                item = {**base, "type": "attr_synonym",
                        "question": f"Is there a {cand_name} in the scene?",
                        "answer": "no", "answer_bbox": None,
                        "distractor_source": cand_folder}
                out["attr_synonym_no"].append((key, item))
        if cand_attrs["color"]:
            cand_color = cand_attrs["color"][0]
            if cand_color not in present_colors and cand_color not in seen_colors_this_frame:
                seen_colors_this_frame.add(cand_color)
                item = {**base, "type": "attr_absent",
                        "question": f"Is there a {cand_color} object in the scene?",
                        "answer": "no", "answer_bbox": None,
                        "distractor_source": cand_folder}
                out["attr_absent"].append((cand_color, item))

    # ---- counting by shared material.
    mat_owners = {}
    for oid, a in attrs.items():
        for m in a["material"]:
            mat_owners.setdefault(m, set()).add(oid)
    for m, ov in mat_owners.items():
        if len(ov) < 2:
            continue
        item = {**base, "type": "attr_count",
                "question": f"How many objects in the scene are made of {m}?",
                "answer": len(ov), "answer_oids": [mapping[o]["oid"] for o in ov]}
        out["attr_count"].append((m, item))

    # ---- odd-one-out. Guard: exactly one visible object must lack a
    # material shared by every other visible object.
    for m, ov in mat_owners.items():
        missing = [oid for oid in attrs if oid not in ov]
        if len(missing) == 1 and len(ov) >= 2:
            oid = missing[0]
            item = {**base, "type": "attr_odd_one_out",
                    "question": f"Which object is NOT made of {m}?",
                    "answer": obj_name(oid), "answer_bbox": bbox(oid),
                    "target_oid": mapping[oid]["oid"]}
            out["attr_odd_one_out"].append((m, item))

    return out

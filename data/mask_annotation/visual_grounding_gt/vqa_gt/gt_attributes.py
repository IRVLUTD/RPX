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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lib import object_attrs

ALL_TYPES = (
    "attr_single_color", "attr_single_material", "attr_single_function",
    "attr_composition", "attr_odd_one_out", "attr_count",
    "attr_synonym", "attr_absent",
)


def bbox_of(mask: np.ndarray, oid: int):
    ys, xs = np.where(mask == oid)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _all_names(a: dict) -> set:
    """Every recorded name/synonym for one object, lowercased, for
    ambiguity checks against other visible objects."""
    return {n.lower() for n in a.get("name", [])}


def gen_attribute_questions(scene_name, kind, phase, fid, mask_path, mapping,
                             lookup, sos_catalog_names, rng):
    """mapping: {mask_id: {"name": str, "oid": fewsol_id str}}. lookup:
    fewsol_id -> SOS folder path. sos_catalog_names: full list of catalog
    object folder names, for attr_absent's distractor pool."""
    mask = np.array(Image.open(mask_path))
    present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping]
    if len(present_ids) < 2:
        return []

    attrs = {}
    for oid in present_ids:
        a = object_attrs(mapping[oid]["oid"], lookup)
        if a:
            attrs[oid] = a
    if len(attrs) < 2:
        return []

    base = {"scene_id": scene_name, "kind": kind, "phase": phase, "frame": fid, "mask_path": mask_path}
    items = []

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
        for val, ov in owners.items():
            if len(ov) != 1:
                continue
            oid = ov[0]
            q = (f"Which object is {val}?" if field == "color"
                 else f"Which object is {verb} {val}?")
            items.append({**base, "type": f"attr_single_{field}", "question": q,
                          "answer": obj_name(oid), "answer_bbox": bbox(oid),
                          "attr_value": val, "target_oid": mapping[oid]["oid"]})

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
        items.append({**base, "type": "attr_composition",
                      "question": f"What is the object made of {m} that is used for {f}?",
                      "answer": obj_name(oid), "answer_bbox": bbox(oid),
                      "target_oid": mapping[oid]["oid"]})

    # ---- synonym recognition. "Yes" case: a real, less-common synonym of a
    # visible object, kept only if that synonym string doesn't also appear
    # in any OTHER visible object's own name list (would make the yes/no
    # answer correct but the underlying reference ambiguous). "No" case:
    # a real synonym belonging to some catalog object, kept only if it
    # matches NO visible object's name list at all -- same "genuinely
    # absent, not a made-up trap" standard attr_absent already uses.
    present_name_sets = {oid: _all_names(a) for oid, a in attrs.items()}
    yes_candidates = []
    for oid, a in attrs.items():
        names = a["name"]
        if len(names) < 2:
            continue
        for alt in names[1:]:
            alt_l = alt.lower()
            if any(alt_l in present_name_sets[other] for other in attrs if other != oid):
                continue
            yes_candidates.append((oid, alt))
    if yes_candidates:
        oid, alt = rng.choice(yes_candidates)
        items.append({**base, "type": "attr_synonym",
                      "question": f"Is there a {alt} in the scene?",
                      "answer": "yes", "answer_bbox": bbox(oid),
                      "canonical_name": obj_name(oid), "target_oid": mapping[oid]["oid"]})

    all_present_names = set().union(*present_name_sets.values()) if present_name_sets else set()
    no_candidates = list(sos_catalog_names)
    rng.shuffle(no_candidates)
    for cand_folder in no_candidates[:15]:
        cand_fewsol = None
        for fid_, folder in lookup.items():
            if folder.endswith("/" + cand_folder) or folder == cand_folder:
                cand_fewsol = fid_
                break
        if cand_fewsol is None:
            continue
        cand_attrs = object_attrs(cand_fewsol, lookup)
        if not cand_attrs or not cand_attrs["name"]:
            continue
        cand_name = cand_attrs["name"][0]
        if cand_name.lower() in all_present_names:
            continue
        items.append({**base, "type": "attr_synonym",
                      "question": f"Is there a {cand_name} in the scene?",
                      "answer": "no", "answer_bbox": None,
                      "distractor_source": cand_folder})
        break

    # ---- absence probe: a real attribute value from the wider catalog
    # that's genuinely absent from THIS frame's actual objects.
    present_colors = {v for a in attrs.values() for v in a["color"]}
    distractor_pool = list(sos_catalog_names)
    rng.shuffle(distractor_pool)
    for cand_folder in distractor_pool[:15]:
        cand_fewsol = None
        for fid_, folder in lookup.items():
            if folder.endswith("/" + cand_folder) or folder == cand_folder:
                cand_fewsol = fid_
                break
        if cand_fewsol is None:
            continue
        cand_attrs = object_attrs(cand_fewsol, lookup)
        if not cand_attrs or not cand_attrs["color"]:
            continue
        cand_color = cand_attrs["color"][0]
        if cand_color not in present_colors:
            items.append({**base, "type": "attr_absent",
                          "question": f"Is there a {cand_color} object in the scene?",
                          "answer": "no", "answer_bbox": None,
                          "distractor_source": cand_folder})
            break

    # ---- counting by shared material.
    mat_owners = {}
    for oid, a in attrs.items():
        for m in a["material"]:
            mat_owners.setdefault(m, set()).add(oid)
    for m, ov in mat_owners.items():
        if len(ov) < 2:
            continue
        items.append({**base, "type": "attr_count",
                      "question": f"How many objects in the scene are made of {m}?",
                      "answer": len(ov), "answer_oids": [mapping[o]["oid"] for o in ov]})

    # ---- odd-one-out. Guard: exactly one visible object must lack a
    # material shared by every other visible object.
    for m, ov in mat_owners.items():
        missing = [oid for oid in attrs if oid not in ov]
        if len(missing) == 1 and len(ov) >= 2:
            oid = missing[0]
            items.append({**base, "type": "attr_odd_one_out",
                          "question": f"Which object is NOT made of {m}?",
                          "answer": obj_name(oid), "answer_bbox": bbox(oid),
                          "target_oid": mapping[oid]["oid"]})

    return items

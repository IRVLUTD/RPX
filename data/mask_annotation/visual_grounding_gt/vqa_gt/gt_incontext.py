"""In-context (two-image) VQA ground truth: Image 1 is a deterministic SOS
reference crop (sos_reference.py), Image 2 is the existing MOS/Ego target
frame. The answer is always exactly one bbox in Image 2.

Three task families, per the frozen spec:
  1. inctx_attr_single_color / _material / _function / _composition /
     _odd_one_out  -- "attribute transfer" (Option B): Image 1 shows a
     DIFFERENT SOS object that shares the intended attribute with the real
     answer object in Image 2. See the module docstring section below on
     the multi-attribute ambiguity gate for why "shares the intended
     attribute" is not, by itself, a sufficient condition.
  2. inctx_spatial_farthest -- Image 1 shows the EXACT SOS identity of the
     existing spatial_farthest anchor object (Option A is intentional and
     correct here -- see vqa_gt/README.md's in-context section).

Source-fact eligibility is intentionally NOT recomputed independently here:
the five general families reuse the exact same "exactly one owner"/"exactly
one minority" conditions gt_attributes.py's collect_candidates() already
enforces (duplicated below in _present_attrs/_owners/_pair_owners/
_mat_owners rather than importing gt_attributes' internals, so this module
can never accidentally change collect_candidates()'s own published output --
see the report's implementation-structure section for why). The spatial
family reuses gt_spatial.compute_farthest_candidates() directly (no
duplication -- that logic has a non-trivial 2D/3D agreement check worth
sharing verbatim).

---- The multi-attribute ambiguity gate (why matching the intended value
alone is not enough) ----

A questionnaire object can have multiple colors/materials/functions. The
in-context question never names the specific value ("same color as Image 1",
not "same red as Image 1") -- so if the chosen reference has a SECOND value
in that field that some OTHER visible Image-2 object also happens to own,
the question now has two textually-valid answers even though only one was
intended. This module rejects any reference candidate for which this can
happen: for every one of the candidate's own values in the relevant field,
every scene object owning that value (in Image 2, this exact frame) must be
the intended target and no one else. Composition applies this over the
candidate's full material x function cross product; odd-one-out applies it
by requiring every one of the candidate's materials that produces a
well-defined odd-one-out fact in this frame to agree on the same answer.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from generate_spatial_gt import MIN_MASK_AREA_PX, compute_instances
from gt_attributes import bbox_of
from gt_spatial import compute_farthest_candidates, _add_position_proxies, _bbox_of_instance
from lib import object_attrs
from sos_catalog import Identity, join_scene_mask, normalize_value

SCHEMA_VERSION = "incontext_v1"
GENERATOR_VERSION = "gt_incontext.py@v1"

GENERAL_FIELD_TYPES = {
    "color": "inctx_attr_single_color",
    "material": "inctx_attr_single_material",
    "function": "inctx_attr_single_function",
}

TEMPLATES = {
    "inctx_attr_single_color": "Which object in Image 2 has the same color as the object shown in "
                                "Image 1? What is its bounding box in Image 2?",
    "inctx_attr_single_material": "Which object in Image 2 is made of the same material as the object "
                                   "shown in Image 1? What is its bounding box in Image 2?",
    "inctx_attr_single_function": "Which object in Image 2 is used for the same purpose as the object "
                                   "shown in Image 1? What is its bounding box in Image 2?",
    "inctx_attr_composition": "Which object in Image 2 has the same material and purpose as the object "
                               "shown in Image 1? What is its bounding box in Image 2?",
    "inctx_attr_odd_one_out": "Which object in Image 2 is NOT made of the same material as the object "
                              "shown in Image 1? What is its bounding box in Image 2?",
    "inctx_spatial_farthest": "Which object in Image 2 is farthest from the object shown in Image 1? "
                              "What is its bounding box in Image 2?",
}

SOURCE_QUESTION = {
    "inctx_attr_single_color": lambda v: f"Which object is {v}?",
    "inctx_attr_single_material": lambda v: f"Which object is made of {v}?",
    "inctx_attr_single_function": lambda v: f"Which object is used for {v}?",
    "inctx_attr_composition": lambda mf: f"What is the object made of {mf[0]} that is used for {mf[1]}?",
    "inctx_attr_odd_one_out": lambda v: f"Which object is NOT made of {v}?",
    "inctx_spatial_farthest": lambda v: f"Which object is farthest from the {v}?",
}


class DropReason:
    NO_ELIGIBLE_FACT = "no_eligible_fact"                  # matches gt_attributes.py's own eligibility gate
    NO_CANDIDATE_REFERENCE = "no_candidate_reference"       # no catalog object carries the value at all
    AMBIGUITY_GATE_FAILED = "ambiguity_gate_failed"         # every candidate created a second valid answer
    TARGET_BBOX_INVALID = "target_bbox_invalid"
    NO_CATALOG_JOIN_FOR_ANCHOR = "no_catalog_join_for_anchor"  # spatial only: anchor not in the 70-object catalog
    AMBIGUOUS_ANCHOR_MAPPING = "ambiguous_anchor_mapping"


@dataclass
class Drop:
    type: str
    reason: str
    detail: str = ""


# ---------------------------------------------------------------- frame context

def _present_attrs(mask: np.ndarray, mapping: dict, lookup: dict) -> dict:
    """oid (local mask id) -> raw attrs dict, for every object visible in
    this frame at or above MIN_MASK_AREA_PX with a resolvable questionnaire.
    Deliberately duplicated from gt_attributes.collect_candidates()'s own
    opening lines (not imported) -- see module docstring."""
    present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping
                   and int((mask == v).sum()) >= MIN_MASK_AREA_PX]
    attrs = {}
    for oid in present_ids:
        a = object_attrs(mapping[oid]["oid"], lookup)
        if a:
            attrs[oid] = a
    return attrs


def _owners(attrs: dict, field: str) -> dict:
    o = {}
    for oid, a in attrs.items():
        for v in a[field]:
            o.setdefault(v, []).append(oid)
    return o


def _norm_owners(attrs: dict, field: str) -> dict:
    o = {}
    for oid, a in attrs.items():
        for v in a[field]:
            o.setdefault(normalize_value(v), set()).add(oid)
    return o


def _pair_owners(attrs: dict) -> dict:
    o = {}
    for oid, a in attrs.items():
        for m in a["material"]:
            for f in a["function"]:
                o.setdefault((m, f), []).append(oid)
    return o


def _norm_pair_owners(attrs: dict) -> dict:
    o = {}
    for oid, a in attrs.items():
        for m in a["material"]:
            for f in a["function"]:
                o.setdefault((normalize_value(m), normalize_value(f)), set()).add(oid)
    return o


def _mat_owners(attrs: dict) -> dict:
    o = {}
    for oid, a in attrs.items():
        for m in a["material"]:
            o.setdefault(m, set()).add(oid)
    return o


# ---------------------------------------------------------------- identity / selection

def _identity_ok(candidate, target_identity: Identity, target_local_name: str, require_diff_category: bool) -> bool:
    if target_identity.global_object_id is not None and candidate.global_object_id == target_identity.global_object_id:
        return False
    if candidate.source_catalog_id == target_identity.source_catalog_id:
        return False  # redundant safety net if global_object_id somehow didn't catch it
    if require_diff_category:
        if normalize_value(candidate.object_name) == normalize_value(target_local_name):
            return False
        if normalize_value(candidate.class_name) == normalize_value(target_local_name):
            return False
    return True


def _selection_hash(seed: int, scene_id: str, kind: str, phase, target_key: str, fact_key: str, ref_global_id: int) -> str:
    key = f"{seed}|{scene_id}|{kind}|{phase}|{target_key}|{fact_key}|{ref_global_id}"
    return hashlib.sha256(key.encode()).hexdigest()


def _pick_reference(candidates: list, seed: int, scene_id: str, kind: str, phase, target_key: str, fact_key: str):
    if not candidates:
        return None
    return min(candidates, key=lambda c: _selection_hash(seed, scene_id, kind, phase, target_key, fact_key, c.global_object_id))


def _sample_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


# ---------------------------------------------------------------- general (attribute-transfer) families

def _find_single_field_references(field: str, value: str, target_local_id: int, attrs: dict,
                                   catalog: dict, target_identity: Identity, target_local_name: str):
    """Returns (candidates_identity_only, candidates_diff_category) -- both
    already passed the cross-modal ambiguity gate (§5 of the spec)."""
    norm_val = normalize_value(value)
    norm_owners = _norm_owners(attrs, field)
    identity_only, diff_category = [], []
    for obj in catalog.values():
        if norm_val not in obj.attrs_norm[field]:
            continue
        if not _identity_ok(obj, target_identity, target_local_name, require_diff_category=False):
            continue
        # ambiguity gate: EVERY value this candidate owns in this field must,
        # in this exact frame, be owned only by the target (or by no one).
        ambiguous = False
        for v in obj.attrs_norm[field]:
            owners_of_v = norm_owners.get(v, set())
            if owners_of_v - {target_local_id}:
                ambiguous = True
                break
        if ambiguous:
            continue
        identity_only.append(obj)
        if _identity_ok(obj, target_identity, target_local_name, require_diff_category=True):
            diff_category.append(obj)
    return identity_only, diff_category


def _find_composition_references(m: str, f: str, target_local_id: int, attrs: dict,
                                  catalog: dict, target_identity: Identity, target_local_name: str):
    norm_pair = (normalize_value(m), normalize_value(f))
    norm_pair_owners = _norm_pair_owners(attrs)
    identity_only, diff_category = [], []
    for obj in catalog.values():
        cand_pairs = {(normalize_value(mm), normalize_value(ff))
                      for mm in obj.attrs_raw["material"] for ff in obj.attrs_raw["function"]}
        if norm_pair not in cand_pairs:
            continue
        if not _identity_ok(obj, target_identity, target_local_name, require_diff_category=False):
            continue
        ambiguous = False
        for pair in cand_pairs:
            owners_of_pair = norm_pair_owners.get(pair, set())
            if owners_of_pair - {target_local_id}:
                ambiguous = True
                break
        if ambiguous:
            continue
        identity_only.append(obj)
        if _identity_ok(obj, target_identity, target_local_name, require_diff_category=True):
            diff_category.append(obj)
    return identity_only, diff_category


def _find_odd_one_out_references(majority_material: str, target_local_id: int, attrs: dict,
                                  mat_owners: dict, catalog: dict, target_identity: Identity, target_local_name: str):
    """target_local_id here is the MINORITY (answer) object -- the reference
    must possess majority_material and must never be the answer object."""
    norm_owners = _norm_owners(attrs, "material")
    norm_majority = normalize_value(majority_material)
    identity_only, diff_category = [], []
    for obj in catalog.values():
        if norm_majority not in obj.attrs_norm["material"]:
            continue
        if not _identity_ok(obj, target_identity, target_local_name, require_diff_category=False):
            continue
        # every applicable material of this candidate must yield the SAME
        # odd-one-out answer (or no well-defined odd-one-out at all) -- see
        # module docstring.
        ambiguous = False
        for cand_m_norm in obj.attrs_norm["material"]:
            # find the raw material string(s) in THIS frame matching this
            # normalized value, to look up mat_owners (raw-keyed, matching
            # gt_attributes.py's own eligibility exactly)
            raw_ms = [m for m in mat_owners if normalize_value(m) == cand_m_norm]
            for raw_m in raw_ms:
                ov = mat_owners[raw_m]
                missing = [oid for oid in attrs if oid not in ov]
                if len(missing) == 1 and len(ov) >= 2 and missing[0] != target_local_id:
                    ambiguous = True
                    break
            if ambiguous:
                break
        if ambiguous:
            continue
        identity_only.append(obj)
        if _identity_ok(obj, target_identity, target_local_name, require_diff_category=True):
            diff_category.append(obj)
    return identity_only, diff_category


def gen_incontext_general_for_frame(scene_id, kind, phase, fid, mask, mapping, lookup, catalog,
                                     catalog_by_scid, ref_crops, seed, revision, require_diff_category=True):
    """Returns (items, drops). items: list of output rows (schema in
    generate_gt.py's wiring / the report). drops: list of Drop, for the
    validator's drop-reason accounting. require_diff_category selects which
    policy's candidate pool is actually USED to build rows (identity-only
    candidates are still computed and counted for the coverage audit even
    when the stricter policy is what gets used -- see gen_for_*_incontext's
    caller for how both counts get reported)."""
    attrs = _present_attrs(mask, mapping, lookup)
    items, drops = [], []
    if len(attrs) < 2:
        return items, drops

    img_h, img_w = mask.shape[:2]

    def emit(type_, target_local_id, fact_key, source_q, identity_only, diff_category,
              attribute_kind, attribute_value=None, attribute_material=None, attribute_function=None):
        pool = diff_category if require_diff_category else identity_only
        if not pool:
            drops.append(Drop(type_, DropReason.AMBIGUITY_GATE_FAILED if identity_only
                               else DropReason.NO_CANDIDATE_REFERENCE, fact_key))
            return
        ref_obj = _pick_reference(pool, seed, scene_id, kind, phase, str(target_local_id), fact_key)
        crop = ref_crops.get(ref_obj.object_id)
        if crop is None:
            drops.append(Drop(type_, "missing_reference_crop", ref_obj.object_id))
            return
        bbox = bbox_of(mask, target_local_id)
        if not (0 <= bbox[0] <= bbox[2] < img_w and 0 <= bbox[1] <= bbox[3] < img_h):
            drops.append(Drop(type_, DropReason.TARGET_BBOX_INVALID, str(bbox)))
            return
        target_identity = join_scene_mask(mapping, target_local_id, catalog_by_scid)
        cx = ((bbox[0] + bbox[2]) / 2.0) / img_w
        cy = ((bbox[1] + bbox[3]) / 2.0) / img_h
        items.append(_build_row(
            scene_id=scene_id, kind=kind, phase=phase, frame=fid, img_w=img_w, img_h=img_h,
            type_=type_, question=TEMPLATES[type_], answer=mapping[target_local_id]["name"],
            answer_bbox=bbox, evidence=None, revision=revision,
            source_type=type_.replace("inctx_", "", 1), source_question=source_q,
            target_identity=target_identity, ref_obj=ref_obj, ref_crop=crop,
            attribute_kind=attribute_kind, attribute_value=attribute_value,
            attribute_material=attribute_material, attribute_function=attribute_function,
            center_x_norm=cx, center_y_norm=cy,
            different_category=(ref_obj in diff_category),
            target_local_mask_id=target_local_id, target_frame_locator_scene=scene_id,
            fact_key=fact_key,
        ))

    for field, type_ in GENERAL_FIELD_TYPES.items():
        for value, ov in _owners(attrs, field).items():
            if len(ov) != 1:
                continue
            target_local_id = ov[0]
            target_identity_for_check = join_scene_mask(mapping, target_local_id, catalog_by_scid)
            identity_only, diff_category = _find_single_field_references(
                field, value, target_local_id, attrs, catalog,
                target_identity_for_check, mapping[target_local_id]["name"])
            emit(type_, target_local_id, f"{field}:{normalize_value(value)}",
                 SOURCE_QUESTION[type_](value), identity_only, diff_category,
                 attribute_kind=field, attribute_value=value)

    for (m, f), ov in _pair_owners(attrs).items():
        if len(ov) != 1:
            continue
        target_local_id = ov[0]
        target_identity_for_check = join_scene_mask(mapping, target_local_id, catalog_by_scid)
        identity_only, diff_category = _find_composition_references(
            m, f, target_local_id, attrs, catalog, target_identity_for_check, mapping[target_local_id]["name"])
        emit("inctx_attr_composition", target_local_id, f"composition:{normalize_value(m)}|{normalize_value(f)}",
             SOURCE_QUESTION["inctx_attr_composition"]((m, f)), identity_only, diff_category,
             attribute_kind="composition", attribute_material=m, attribute_function=f)

    mat_owners = _mat_owners(attrs)
    for m, ov in mat_owners.items():
        missing = [oid for oid in attrs if oid not in ov]
        if not (len(missing) == 1 and len(ov) >= 2):
            continue
        target_local_id = missing[0]  # the minority/answer object
        target_identity_for_check = join_scene_mask(mapping, target_local_id, catalog_by_scid)
        identity_only, diff_category = _find_odd_one_out_references(
            m, target_local_id, attrs, mat_owners, catalog, target_identity_for_check, mapping[target_local_id]["name"])
        emit("inctx_attr_odd_one_out", target_local_id, f"odd_one_out:{normalize_value(m)}",
             SOURCE_QUESTION["inctx_attr_odd_one_out"](m), identity_only, diff_category,
             attribute_kind="material", attribute_material=m)

    return items, drops


# ---------------------------------------------------------------- spatial (Option A)

def gen_incontext_spatial_for_frame(scene_id, phase, fid, mask, depth, mapping, catalog_by_scid,
                                     ref_crops, revision):
    items, drops = [], []
    instances = compute_instances(mask, mapping, depth)
    inst_list = list(instances.values())
    if len(inst_list) < 2:
        return items, drops
    img_h, img_w = mask.shape[:2]
    _add_position_proxies(instances, img_w, img_h)
    depth_vals = [(i["median_depth"], i) for i in inst_list if i["median_depth"] is not None]
    farthest_candidates = compute_farthest_candidates(inst_list, depth_vals)

    for ref_inst, top, d2d, others in farthest_candidates:
        anchor_identity = join_scene_mask(mapping, ref_inst["mask_id"], catalog_by_scid)
        if anchor_identity.object_id is None:
            drops.append(Drop("inctx_spatial_farthest", DropReason.NO_CATALOG_JOIN_FOR_ANCHOR, ref_inst["name"]))
            continue
        crop = ref_crops.get(anchor_identity.object_id)
        if crop is None:
            drops.append(Drop("inctx_spatial_farthest", "missing_reference_crop", anchor_identity.object_id))
            continue
        answer_bbox = _bbox_of_instance(top)
        if not (0 <= answer_bbox[0] <= answer_bbox[2] < img_w and 0 <= answer_bbox[1] <= answer_bbox[3] < img_h):
            drops.append(Drop("inctx_spatial_farthest", DropReason.TARGET_BBOX_INVALID, str(answer_bbox)))
            continue
        assert top["mask_id"] != ref_inst["mask_id"], "farthest answer must never be the anchor itself"
        target_identity = join_scene_mask(mapping, top["mask_id"], catalog_by_scid)
        cx = ((answer_bbox[0] + answer_bbox[2]) / 2.0) / img_w
        cy = ((answer_bbox[1] + answer_bbox[3]) / 2.0) / img_h
        ref_cat_obj = _CatalogObjLike(anchor_identity, crop)
        items.append(_build_row(
            scene_id=scene_id, kind="mos", phase=phase, frame=fid, img_w=img_w, img_h=img_h,
            type_="inctx_spatial_farthest", question=TEMPLATES["inctx_spatial_farthest"],
            answer=top["name"], answer_bbox=answer_bbox,
            evidence={"reference": ref_inst["name"],
                      "distances_2d_px": {o["name"]: round(d2d[o["mask_id"]], 1) for o in others}},
            revision=revision, source_type="spatial_farthest",
            source_question=SOURCE_QUESTION["inctx_spatial_farthest"](ref_inst["name"]),
            target_identity=target_identity, ref_obj=ref_cat_obj, ref_crop=crop,
            attribute_kind=None, attribute_value=None, attribute_material=None, attribute_function=None,
            center_x_norm=cx, center_y_norm=cy, different_category=None,
            target_local_mask_id=top["mask_id"], target_frame_locator_scene=scene_id,
            fact_key=f"spatial_farthest:{anchor_identity.source_catalog_id}",
        ))
    return items, drops


class _CatalogObjLike:
    """Adapter so _build_row's ref_obj.* access works uniformly for the
    spatial family, whose "reference" is an Identity (resolved from the
    scene's own anchor) rather than a fresh catalog search result."""
    def __init__(self, identity: Identity, crop):
        self.object_id = identity.object_id
        self.global_object_id = identity.global_object_id
        self.source_catalog_id = identity.source_catalog_id


# ---------------------------------------------------------------- row assembly

def _build_row(*, scene_id, kind, phase, frame, img_w, img_h, type_, question, answer, answer_bbox,
                evidence, revision, source_type, source_question, target_identity: Identity,
                ref_obj, ref_crop, attribute_kind, attribute_value, attribute_material, attribute_function,
                center_x_norm, center_y_norm, different_category, target_local_mask_id,
                target_frame_locator_scene, fact_key):
    ref_key = ref_obj.global_object_id
    sample_id = _sample_id(type_, scene_id, kind, phase, frame, target_identity.source_catalog_id,
                            ref_obj.source_catalog_id, fact_key)
    rgb_ext = "webp"
    if kind == "mos":
        target_shard = f"scenes/{scene_id}/{phase}/rgb.tar"
    else:
        target_shard = f"scenes/{scene_id}/ego/rgb.tar"
    return {
        "scene_id": scene_id, "kind": kind, "phase": phase, "frame": frame,
        "img_w": int(img_w), "img_h": int(img_h),
        "type": type_, "question": question, "answer": answer, "answer_bbox": answer_bbox,
        "evidence": evidence,
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "dataset_revision": revision, "sample_id": sample_id,
        "source_type": source_type, "source_question": source_question,
        "target_image": {"repo_id": "IRVLUTD/RPX", "revision": revision,
                          "shard": target_shard, "member": f"rgb/{frame}.{rgb_ext}"},
        "reference_image": {"repo_id": "IRVLUTD/RPX", "revision": revision,
                             "shard": getattr(ref_crop, "rgb_shard", None) or "",
                             "member": getattr(ref_crop, "rgb_member", None) or "",
                             "crop_bbox": list(ref_crop.crop_bbox)},
        "target_local_mask_id": int(target_local_mask_id),
        "target_source_catalog_id": target_identity.source_catalog_id,
        "target_object_id": target_identity.object_id,
        "target_global_object_id": target_identity.global_object_id,
        "reference_source_catalog_id": ref_obj.source_catalog_id,
        "reference_object_id": ref_obj.object_id,
        "reference_global_object_id": ref_obj.global_object_id,
        "reference_frame": ref_crop.frame_id,
        "reference_mask_bbox": list(ref_crop.mask_bbox),
        "reference_crop_bbox": list(ref_crop.crop_bbox),
        "reference_crop_sha256": ref_crop.crop_sha256,
        "attribute_kind": attribute_kind, "attribute_value": attribute_value,
        "attribute_material": attribute_material, "attribute_function": attribute_function,
        "answer_center_x_norm": center_x_norm, "answer_center_y_norm": center_y_norm,
        "answer_is_centered": bool(0.25 <= center_x_norm <= 0.75 and 0.25 <= center_y_norm <= 0.75),
        "different_category": different_category,
        "_dedup_key": f"{type_}|{fact_key}|{target_identity.source_catalog_id}|{ref_obj.source_catalog_id}",
    }


# ---------------------------------------------------------------- phase-wide dedup (general family only)

def gen_incontext_general_for_phase(scene_id, kind, phase, selected_fids, mask_dir, mapping, lookup,
                                     catalog, catalog_by_scid, ref_crops, seed, revision, rng,
                                     max_per_type=None, require_diff_category=True):
    """Phase-wide dedup, mirroring generate_gt.py's _gen_attributes_for_phase
    exactly for the same reason: a fact ("this lion is the unique blue
    object, and hammer.2 is a valid blue reference for it") is a property of
    the static object arrangement, not of which frame happens to be
    rendering it -- repeating it independently in every frame of the phase
    would reproduce the same >90% repeat-rate problem the original
    generator's dedup.py was built to fix (see vqa_gt/README.md). The
    reference is already deterministic per (scene,kind,phase,target,fact) --
    see _pick_reference -- so the same fact/reference pairing recurs
    verbatim across every frame it's valid in; only answer_bbox/frame/
    center_*_norm actually vary per frame. dedup.distribute() places each
    distinct (type, fact, target, reference) key once per frame it's valid
    in, spread by the same least-loaded-first rule as the original
    generator, before anything repeats.

    Returns (items, drops_by_type_reason: dict[(type,reason) -> count],
    frame_coverage: dict[frame_id -> int] for the report)."""
    import numpy as np
    from PIL import Image
    import dedup as dedup_mod

    frame_candidates = {}
    all_drops = []
    for fid in selected_fids:
        mask = np.array(Image.open(f"{mask_dir}/{fid}.png"))
        items, drops = gen_incontext_general_for_frame(
            scene_id, kind, phase, fid, mask, mapping, lookup, catalog, catalog_by_scid,
            ref_crops, seed, revision, require_diff_category=require_diff_category)
        all_drops.extend(drops)
        bucket = {}
        for it in items:
            bucket.setdefault(it["type"], []).append((it["_dedup_key"], it))
        frame_candidates[fid] = bucket

    assigned = dedup_mod.distribute(frame_candidates, max_per_type, rng)
    out_items = []
    for fid in selected_fids:
        for it in assigned[fid]:
            it = dict(it)
            it.pop("_dedup_key", None)
            out_items.append(it)
    return out_items, all_drops

"""Q3: build a TRUE row-paired in-context general bank -- one row per
published attribute.parquet row (of the 5 retained types), not an
independently-redistributed fact bank. scene/kind/phase/frame/target/bbox
are taken VERBATIM from the published row; only a valid Option-B reference
is attached (or the row is dropped if none exists). No independent
dedup.distribute() redistribution -- this is a pure 1:1 augmentation of the
existing published set.

Restages ONLY sam2/masks+meta tars (no rgb/depth/fisheye), same ~360MB
footprint as exhaustive_validate_general.py, scene-condition by
scene-condition, deleted after each cell's rows are processed.
"""
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))
from generate_spatial_gt import load_mapping, imread_mask, MIN_MASK_AREA_PX  # noqa: E402
from lib import object_attrs, load_fewsol_lookup  # noqa: E402
import sos_catalog as sc  # noqa: E402
import sos_reference as sr  # noqa: E402
import gt_incontext as gi  # noqa: E402

SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
STAGE_TMP = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged_masks_only")
REF_MANIFEST = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops/reference_crops_v1.parquet")
PUBLISHED = "out/vqa_parquet/attribute.parquet"
OUT_PATH = Path("out/incontext_paired/gt_incontext_general_rowpaired.jsonl")
DROPS_PATH = Path("out/incontext_paired/rowpaired_drops.jsonl")
SEED = 42
REVISION = "main"
RETAINED_TYPES = {
    "attr_single_color": ("inctx_attr_single_color", "color"),
    "attr_single_material": ("inctx_attr_single_material", "material"),
    "attr_single_function": ("inctx_attr_single_function", "function"),
    "attr_composition": ("inctx_attr_composition", "composition"),
    "attr_odd_one_out": ("inctx_attr_odd_one_out", "odd_one_out"),
}


def stage_masks_only(scene, phase_dir):
    from huggingface_hub import hf_hub_download
    prefix = f"scenes/{scene}/{'ego' if phase_dir == 'ego' else phase_dir}"
    dest = STAGE_TMP / scene / phase_dir
    if (dest / "sam2" / "mask_to_object.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=f"{prefix}/labels/masks/v1.tar", revision=REVISION)
    with tarfile.open(p) as tf:
        tf.extractall(dest / "sam2")
    if (dest / "sam2" / "sam2" / "masks").exists():
        (dest / "sam2" / "sam2" / "masks").rename(dest / "sam2" / "masks")
        shutil.rmtree(dest / "sam2" / "sam2", ignore_errors=True)
    real = os.path.realpath(p)
    os.remove(p)
    if os.path.exists(real):
        os.remove(real)
    p2 = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=f"{prefix}/labels/sam2_meta/v1.tar", revision=REVISION)
    with tarfile.open(p2) as tf:
        tf.extractall(dest / "_meta")
    (dest / "_meta" / "sam2" / "mask_to_object.json").rename(dest / "sam2" / "mask_to_object.json")
    shutil.rmtree(dest / "_meta")
    real2 = os.path.realpath(p2)
    os.remove(p2)
    if os.path.exists(real2):
        os.remove(real2)
    return dest


def find_target_local_id(mapping, target_oid):
    for mid, v in mapping.items():
        if v["oid"] == target_oid:
            return mid
    return None


def _native(x):
    """pandas/pyarrow row values arrive as numpy scalars (int64, etc.), which
    json.dumps rejects -- cast to plain Python types once, here, rather than
    scattering int()/str() calls through every call site below."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_native(v) for v in x.tolist()]
    if isinstance(x, list):
        return [_native(v) for v in x]
    return x


def process_row(pub_row, mapping, attrs_cache, mask_dir, catalog, catalog_by_scid, ref_crops):
    scene, kind, phase, frame = (_native(pub_row["scene_id"]), _native(pub_row["kind"]),
                                   _native(pub_row["phase"]), _native(pub_row["frame"]))
    pub_type = pub_row["type"]
    inctx_type, field = RETAINED_TYPES[pub_type]
    target_oid = _native(pub_row["target_oid"])

    if frame not in attrs_cache:
        mask_path = mask_dir / f"{frame}.png"
        if not mask_path.exists():
            attrs_cache[frame] = (None, None)
        else:
            mask = imread_mask(str(mask_path))
            present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping
                           and int((mask == v).sum()) >= MIN_MASK_AREA_PX]
            attrs = {}
            for oid in present_ids:
                a = object_attrs(mapping[oid]["oid"], gi_lookup)
                if a:
                    attrs[oid] = a
            attrs_cache[frame] = (mask, attrs)
    mask, attrs = attrs_cache[frame]
    if mask is None:
        return None, gi.Drop(inctx_type, "target_frame_mask_missing", frame)

    target_local_id = find_target_local_id(mapping, target_oid)
    if target_local_id is None or target_local_id not in attrs:
        return None, gi.Drop(inctx_type, "target_not_resolvable_in_frame", str(target_oid))

    target_identity = sc.join_scene_mask(mapping, target_local_id, catalog_by_scid)
    target_name = mapping[target_local_id]["name"]

    if field in ("color", "material", "function"):
        value = _native(pub_row["attr_value"])
        identity_only, diff_category = gi._find_single_field_references(
            field, value, target_local_id, attrs, catalog, target_identity, target_name)
        pool = diff_category
        fact_key = f"{field}:{sc.normalize_value(value)}"
        attribute_kind, attribute_value, attribute_material, attribute_function = field, value, None, None
    elif field == "composition":
        # BUG FIX (found during Q11 canonical validation -- see report): a
        # target can legitimately own MORE THAN ONE unique (material,
        # function) pair in the same frame (e.g. 2 functions sharing a
        # material). The published attribute.parquet has ONE row per such
        # pair (each with its own question text); picking "any" eligible
        # pair via next() ignored that and collapsed every published row
        # for this target into the SAME reconstructed pair, producing
        # duplicate output rows (confirmed: 43,486 mos + 15,754 ego
        # duplicate composition rows, 100% of all duplicates found).
        # Fix: parse the ACTUAL (material, function) pair out of THIS
        # published row's own question text -- the only place the exact
        # pair is recorded in the published schema -- then verify it's
        # still eligible in this frame, rather than guessing.
        q = pub_row["question"]
        try:
            m = q.split("made of ", 1)[1].split(" that is used for ")[0]
            f = q.split(" that is used for ", 1)[1].rstrip("?")
        except IndexError:
            return None, gi.Drop(inctx_type, "composition_question_unparseable", q)
        pair_owners = gi._pair_owners(attrs)
        owners = pair_owners.get((m, f), [])
        if len(owners) != 1 or owners[0] != target_local_id:
            return None, gi.Drop(inctx_type, "published_composition_fact_not_reproducible", f"{m}|{f}")
        identity_only, diff_category = gi._find_composition_references(
            m, f, target_local_id, attrs, catalog, target_identity, target_name)
        pool = diff_category
        fact_key = f"composition:{sc.normalize_value(m)}|{sc.normalize_value(f)}"
        attribute_kind, attribute_value, attribute_material, attribute_function = "composition", None, m, f
    else:  # odd_one_out
        mat_owners = gi._mat_owners(attrs)
        my_mat = next((m for m, ov in mat_owners.items()
                       if len(ov) >= 2 and len([o for o in attrs if o not in ov]) == 1
                       and [o for o in attrs if o not in ov][0] == target_local_id), None)
        if my_mat is None:
            return None, gi.Drop(inctx_type, "published_odd_one_out_fact_not_reproducible", str(target_oid))
        identity_only, diff_category = gi._find_odd_one_out_references(
            my_mat, target_local_id, attrs, mat_owners, catalog, target_identity, target_name)
        pool = diff_category
        fact_key = f"odd_one_out:{sc.normalize_value(my_mat)}"
        attribute_kind, attribute_value, attribute_material, attribute_function = "material", None, my_mat, None

    if not pool:
        return None, gi.Drop(inctx_type, gi.DropReason.NO_CANDIDATE_REFERENCE if not identity_only
                              else gi.DropReason.AMBIGUITY_GATE_FAILED, fact_key)

    ref_obj = gi._pick_reference(pool, SEED, scene, kind, phase, str(target_local_id), fact_key)
    crop = ref_crops.get(ref_obj.object_id)
    if crop is None:
        return None, gi.Drop(inctx_type, "missing_reference_crop", ref_obj.object_id)

    bbox = _native(pub_row["answer_bbox"])
    img_w, img_h = _native(pub_row["img_w"]), _native(pub_row["img_h"])
    cx = ((bbox[0] + bbox[2]) / 2.0) / img_w
    cy = ((bbox[1] + bbox[3]) / 2.0) / img_h

    row = gi._build_row(
        scene_id=scene, kind=kind, phase=phase, frame=frame, img_w=img_w, img_h=img_h,
        type_=inctx_type, question=gi.TEMPLATES[inctx_type], answer=_native(pub_row["answer"]), answer_bbox=list(bbox),
        evidence=None, revision=REVISION, source_type=pub_type, source_question=_native(pub_row["question"]),
        target_identity=target_identity, ref_obj=ref_obj, ref_crop=crop,
        attribute_kind=attribute_kind, attribute_value=attribute_value,
        attribute_material=attribute_material, attribute_function=attribute_function,
        center_x_norm=cx, center_y_norm=cy, different_category=True,
        target_local_mask_id=target_local_id, target_frame_locator_scene=scene, fact_key=fact_key,
    )
    row.pop("_dedup_key", None)
    row["source_sample_id"] = gi._sample_id("published", scene, kind, phase, frame, pub_type, target_oid, fact_key)
    return row, None


def main():
    global gi_lookup
    gi_lookup = load_fewsol_lookup(SOS_ROOT)
    catalog = sc.load_catalog()
    catalog_by_scid = sc.by_source_catalog_id(catalog)
    ref_crops = {oid: sr.ReferenceCrop(**r) for oid, r in sr.load_reference_manifest(REF_MANIFEST).items()}

    print("loading published attribute.parquet ...", flush=True)
    pub = pq.read_table(PUBLISHED).to_pandas()
    pub = pub[pub["type"].isin(RETAINED_TYPES.keys())].copy()
    pub["phase_dir"] = pub.apply(lambda r: "ego" if r["kind"] == "ego" else str(int(r["phase"])), axis=1)
    print(f"{len(pub)} published rows across the 5 retained types", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_out, n_drop = 0, 0
    groups = pub.groupby(["scene_id", "phase_dir"])
    scene_conditions = list(groups.groups.keys())
    print(f"{len(scene_conditions)} scene-conditions to process", flush=True)

    with open(OUT_PATH, "w") as out_f, open(DROPS_PATH, "w") as drop_f:
        for i, (scene, phase_dir) in enumerate(scene_conditions):
            sub = groups.get_group((scene, phase_dir))
            dest = stage_masks_only(scene, phase_dir)
            mapping = load_mapping(str(dest / "sam2" / "mask_to_object.json"))
            mask_dir = dest / "sam2" / "masks"
            attrs_cache = {}
            for _, pub_row in sub.iterrows():
                row, drop = process_row(pub_row, mapping, attrs_cache, mask_dir, catalog, catalog_by_scid, ref_crops)
                if row is not None:
                    out_f.write(json.dumps(row) + "\n")
                    n_out += 1
                else:
                    drop_f.write(json.dumps({"scene_id": scene, "phase_dir": phase_dir, "type": drop.type,
                                              "reason": drop.reason, "detail": drop.detail}) + "\n")
                    n_drop += 1
            shutil.rmtree(STAGE_TMP / scene, ignore_errors=True)
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(scene_conditions)} scene-conditions, {n_out} kept, {n_drop} dropped", flush=True)

    print(f"\n{len(pub)} published rows -> {n_out} row-paired rows, {n_drop} dropped")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()

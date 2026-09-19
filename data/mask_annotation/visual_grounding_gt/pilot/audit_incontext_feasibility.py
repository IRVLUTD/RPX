"""Phase-B feasibility audit (spec section 14.B): for every eligible general
attribute-transfer fact across the 7 already-staged scenes (mos all 3 phases
+ ego), compute whether AT LEAST ONE valid reference exists under (1)
identity-only and (2) different-category policies, and tally drop reasons.
Also tallies spatial_farthest anchor-catalog-join coverage. Reuses the exact
matching helpers gt_incontext.py uses for real generation -- this is a
measurement pass, not a separate reimplementation, so there is no risk of
the audit and the generator disagreeing about what counts as valid.

Untracked (pilot/ convention). Does not write any output rows -- tallies
only, printed as a report.
"""
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))

from generate_spatial_gt import load_mapping, imread_mask, compute_instances, imread_depth  # noqa: E402
from lib import load_fewsol_lookup  # noqa: E402
import sos_catalog as sc  # noqa: E402
import sos_reference as sr  # noqa: E402
import gt_incontext as gi  # noqa: E402
from gt_spatial import compute_farthest_candidates, _add_position_proxies  # noqa: E402

SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
STAGE_ROOT = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged")
REF_MANIFEST = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops/reference_crops_v1.parquet")

SCENES = ["scene001", "scene004", "scene010", "scene016", "scene046", "scene069", "scene098"]


def load_ref_crops():
    import json
    manifest = sr.load_reference_manifest(REF_MANIFEST)
    out = {}
    for oid, r in manifest.items():
        r = dict(r)
        r["mask_bbox"] = json.loads(r["mask_bbox"]) if isinstance(r["mask_bbox"], str) else r["mask_bbox"]
        r["crop_bbox"] = json.loads(r["crop_bbox"]) if isinstance(r["crop_bbox"], str) else r["crop_bbox"]
        out[oid] = sr.ReferenceCrop(**r)
    return out


def audit_general_frame(mask, mapping, lookup, catalog, catalog_by_scid, tally, drop_tally):
    attrs = gi._present_attrs(mask, mapping, lookup)
    if len(attrs) < 2:
        return

    for field, type_ in gi.GENERAL_FIELD_TYPES.items():
        for value, ov in gi._owners(attrs, field).items():
            if len(ov) != 1:
                continue
            target_local_id = ov[0]
            tid = sc.join_scene_mask(mapping, target_local_id, catalog_by_scid)
            identity_only, diff_category = gi._find_single_field_references(
                field, value, target_local_id, attrs, catalog, tid, mapping[target_local_id]["name"])
            tally[type_]["facts"] += 1
            if identity_only:
                tally[type_]["identity_only_ok"] += 1
            else:
                drop_tally[(type_, "no_candidate_reference")] += 1
            if diff_category:
                tally[type_]["diff_category_ok"] += 1
            elif identity_only:
                drop_tally[(type_, "ambiguity_or_category_gate_failed")] += 1

    for (m, f), ov in gi._pair_owners(attrs).items():
        if len(ov) != 1:
            continue
        target_local_id = ov[0]
        tid = sc.join_scene_mask(mapping, target_local_id, catalog_by_scid)
        identity_only, diff_category = gi._find_composition_references(
            m, f, target_local_id, attrs, catalog, tid, mapping[target_local_id]["name"])
        tally["inctx_attr_composition"]["facts"] += 1
        if identity_only:
            tally["inctx_attr_composition"]["identity_only_ok"] += 1
        else:
            drop_tally[("inctx_attr_composition", "no_candidate_reference")] += 1
        if diff_category:
            tally["inctx_attr_composition"]["diff_category_ok"] += 1
        elif identity_only:
            drop_tally[("inctx_attr_composition", "ambiguity_or_category_gate_failed")] += 1

    mat_owners = gi._mat_owners(attrs)
    for m, ov in mat_owners.items():
        missing = [oid for oid in attrs if oid not in ov]
        if not (len(missing) == 1 and len(ov) >= 2):
            continue
        target_local_id = missing[0]
        tid = sc.join_scene_mask(mapping, target_local_id, catalog_by_scid)
        identity_only, diff_category = gi._find_odd_one_out_references(
            m, target_local_id, attrs, mat_owners, catalog, tid, mapping[target_local_id]["name"])
        tally["inctx_attr_odd_one_out"]["facts"] += 1
        if identity_only:
            tally["inctx_attr_odd_one_out"]["identity_only_ok"] += 1
        else:
            drop_tally[("inctx_attr_odd_one_out", "no_candidate_reference")] += 1
        if diff_category:
            tally["inctx_attr_odd_one_out"]["diff_category_ok"] += 1
        elif identity_only:
            drop_tally[("inctx_attr_odd_one_out", "ambiguity_or_category_gate_failed")] += 1


def audit_spatial_frame(mask, depth, mapping, catalog_by_scid, ref_crops, spatial_tally):
    instances = compute_instances(mask, mapping, depth)
    inst_list = list(instances.values())
    if len(inst_list) < 2:
        return
    H, W = mask.shape[:2]
    _add_position_proxies(instances, W, H)
    depth_vals = [(i["median_depth"], i) for i in inst_list if i["median_depth"] is not None]
    for ref_inst, top, d2d, others in compute_farthest_candidates(inst_list, depth_vals):
        spatial_tally["facts"] += 1
        anchor_identity = sc.join_scene_mask(mapping, ref_inst["mask_id"], catalog_by_scid)
        if anchor_identity.object_id is None:
            spatial_tally["anchor_not_in_catalog"] += 1
        elif anchor_identity.object_id not in ref_crops:
            spatial_tally["anchor_no_crop"] += 1
        else:
            spatial_tally["ok"] += 1


def main():
    lookup = load_fewsol_lookup(SOS_ROOT)
    catalog = sc.load_catalog()
    catalog_by_scid = sc.by_source_catalog_id(catalog)
    ref_crops = load_ref_crops()

    tally = {t: Counter() for t in
             ["inctx_attr_single_color", "inctx_attr_single_material", "inctx_attr_single_function",
              "inctx_attr_composition", "inctx_attr_odd_one_out"]}
    drop_tally = Counter()
    spatial_tally = Counter()
    n_frames = {"mos": 0, "ego": 0}

    for scene in SCENES:
        for phase in (0, 1, 2):
            root = STAGE_ROOT / scene / str(phase)
            mapping = load_mapping(str(root / "sam2/mask_to_object.json"))
            fids = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob(str(root / "sam2/masks/*.png")))
            for fid in fids:
                mask = imread_mask(str(root / "sam2/masks" / f"{fid}.png"))
                audit_general_frame(mask, mapping, lookup, catalog, catalog_by_scid, tally, drop_tally)
                depth = imread_depth(str(root / "depth" / f"{fid}.png"))
                audit_spatial_frame(mask, depth, mapping, catalog_by_scid, ref_crops, spatial_tally)
                n_frames["mos"] += 1
        ego_root = STAGE_ROOT / scene / "ego"
        mapping = load_mapping(str(ego_root / "sam2/mask_to_object.json"))
        fids = sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(str(ego_root / "sam2/masks/*.png")))
        for fid in fids:
            mask = imread_mask(str(ego_root / "sam2/masks" / f"{fid}.png"))
            audit_general_frame(mask, mapping, lookup, catalog, catalog_by_scid, tally, drop_tally)
            n_frames["ego"] += 1
        print(f"{scene} done", flush=True)

    print(f"\n{n_frames['mos']} mos frames, {n_frames['ego']} ego frames audited across {len(SCENES)} scenes\n")
    print(f"{'type':30s} {'facts':>8s} {'identity_only_ok':>18s} {'diff_category_ok':>18s} {'id_only_%':>10s} {'diff_cat_%':>10s}")
    for t, c in tally.items():
        facts = c["facts"] or 1
        print(f"{t:30s} {c['facts']:8d} {c['identity_only_ok']:18d} {c['diff_category_ok']:18d} "
              f"{100*c['identity_only_ok']/facts:9.1f}% {100*c['diff_category_ok']/facts:9.1f}%")

    print("\ndrop reasons:")
    for (t, reason), n in sorted(drop_tally.items()):
        print(f"  {t:30s} {reason:35s} {n:6d}")

    print("\nspatial_farthest anchor-catalog-join coverage:")
    for k, v in spatial_tally.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

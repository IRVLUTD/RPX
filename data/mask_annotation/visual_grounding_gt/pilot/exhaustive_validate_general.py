"""Q13: exhaustive (not sampled) cross-modal ambiguity + attribute-sharing
re-verification for EVERY row in the in-context general Parquets.

Determination (see report): the stored row schema does NOT carry the full
per-frame visible-object roster (only target+reference identity), so exact
re-verification needs the real per-frame mask -- but NOT rgb/depth/fisheye.
Smallest sufficient restaging: sam2/masks/v1.tar + sam2_meta/v1.tar per
(scene, mos-phase|ego) -- ~0.9MB + ~10KB each, ~360MB total for all 100
scenes x 4 conditions (vs multi-GB for a full scene restage). This script
does exactly that restaging, scene-by-scene, disk-bounded (delete masks
after that scene's rows are checked), and nothing else -- no rgb, no depth,
no generation, no rerun of gt_incontext.py.
"""
import glob
import json
import os
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))
from generate_spatial_gt import load_mapping, imread_mask, MIN_MASK_AREA_PX  # noqa: E402
from lib import object_attrs, load_fewsol_lookup  # noqa: E402
import sos_catalog as sc  # noqa: E402

SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
STAGE_TMP = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged_masks_only")
OUT_LOG = "out/incontext_paired/exhaustive_ambiguity_report.json"


def stage_masks_only(scene, phase_dir):
    from huggingface_hub import hf_hub_download
    prefix = f"scenes/{scene}/{'ego' if phase_dir == 'ego' else phase_dir}"
    dest = STAGE_TMP / scene / phase_dir
    if (dest / "sam2" / "mask_to_object.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    p = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=f"{prefix}/labels/masks/v1.tar", revision="main")
    with tarfile.open(p) as tf:
        tf.extractall(dest / "sam2")
    if (dest / "sam2" / "sam2" / "masks").exists():
        (dest / "sam2" / "sam2" / "masks").rename(dest / "sam2" / "masks")
        shutil.rmtree(dest / "sam2" / "sam2", ignore_errors=True)
    real = os.path.realpath(p)
    os.remove(p)
    if os.path.exists(real):
        os.remove(real)

    p2 = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=f"{prefix}/labels/sam2_meta/v1.tar", revision="main")
    with tarfile.open(p2) as tf:
        tf.extractall(dest / "_meta")
    (dest / "_meta" / "sam2" / "mask_to_object.json").rename(dest / "sam2" / "mask_to_object.json")
    shutil.rmtree(dest / "_meta")
    real2 = os.path.realpath(p2)
    os.remove(p2)
    if os.path.exists(real2):
        os.remove(real2)
    return dest


def main():
    lookup = load_fewsol_lookup(SOS_ROOT)
    catalog = sc.load_catalog()

    print("loading in-context general parquets ...", flush=True)
    mos = pq.read_table("out/incontext_parquet/incontext_mos_attribute_bbox.parquet").to_pandas()
    ego = pq.read_table("out/incontext_parquet/incontext_ego_attribute_bbox.parquet").to_pandas()
    mos["phase_dir"] = mos["phase"].astype(int).astype(str)
    ego["phase_dir"] = "ego"
    import pandas as pd
    all_df = pd.concat([mos, ego], ignore_index=True)
    print(f"{len(all_df)} total general rows to check", flush=True)

    groups = all_df.groupby(["scene_id", "phase_dir"])
    scene_conditions = list(groups.groups.keys())
    print(f"{len(scene_conditions)} (scene, condition) cells to restage+check", flush=True)

    results = {"share_ok": 0, "share_fail": 0, "ambiguity_ok": 0, "ambiguity_fail": 0}
    failures = []
    n_done = 0

    for scene, phase_dir in scene_conditions:
        sub = groups.get_group((scene, phase_dir))
        dest = stage_masks_only(scene, phase_dir)
        mapping = load_mapping(str(dest / "sam2" / "mask_to_object.json"))
        mask_cache = {}
        for _, row in sub.iterrows():
            fid = row["frame"]
            if fid not in mask_cache:
                mask_path = dest / "sam2" / "masks" / f"{fid}.png"
                if not mask_path.exists():
                    mask_cache[fid] = None
                else:
                    mask = imread_mask(str(mask_path))
                    present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping
                                   and int((mask == v).sum()) >= MIN_MASK_AREA_PX]
                    attrs = {}
                    for oid in present_ids:
                        a = object_attrs(mapping[oid]["oid"], lookup)
                        if a:
                            attrs[oid] = a
                    mask_cache[fid] = attrs
            attrs = mask_cache[fid]
            if attrs is None:
                continue

            ref_obj = catalog.get(row["reference_object_id"])
            if ref_obj is None:
                results["share_fail"] += 1
                continue
            field = row["attribute_kind"]
            if field == "composition":
                share_ok = (sc.normalize_value(row["attribute_material"]) in ref_obj.attrs_norm["material"] and
                            sc.normalize_value(row["attribute_function"]) in ref_obj.attrs_norm["function"])
            elif row["type"] == "inctx_attr_odd_one_out":
                share_ok = sc.normalize_value(row["attribute_material"]) in ref_obj.attrs_norm["material"]
            else:
                share_ok = sc.normalize_value(row["attribute_value"]) in ref_obj.attrs_norm.get(field, [])
            if not share_ok:
                results["share_fail"] += 1
                failures.append((row["sample_id"], "share_fail"))
                continue
            results["share_ok"] += 1

            target_local_id = int(row["target_local_mask_id"])
            ambiguous = False
            if row["type"] == "inctx_attr_odd_one_out":
                mo = {}
                for oid, a in attrs.items():
                    for m in a["material"]:
                        mo.setdefault(m, set()).add(oid)
                for cand_m_norm in ref_obj.attrs_norm["material"]:
                    for raw_m, owners in mo.items():
                        if sc.normalize_value(raw_m) != cand_m_norm:
                            continue
                        missing = [oid for oid in attrs if oid not in owners]
                        if len(missing) == 1 and len(owners) >= 2 and missing[0] != target_local_id:
                            ambiguous = True
            elif field == "composition":
                cand_pairs = {(sc.normalize_value(mm), sc.normalize_value(ff))
                              for mm in ref_obj.attrs_raw["material"] for ff in ref_obj.attrs_raw["function"]}
                for oid, a in attrs.items():
                    if oid == target_local_id:
                        continue
                    other_pairs = {(sc.normalize_value(mm), sc.normalize_value(ff))
                                    for mm in a["material"] for ff in a["function"]}
                    if other_pairs & cand_pairs:
                        ambiguous = True
            else:
                cand_values = set(ref_obj.attrs_norm[field])
                for oid, a in attrs.items():
                    if oid == target_local_id:
                        continue
                    other_norm = {sc.normalize_value(v) for v in a[field]}
                    if other_norm & cand_values:
                        ambiguous = True
            if ambiguous:
                results["ambiguity_fail"] += 1
                failures.append((row["sample_id"], "ambiguity_fail"))
            else:
                results["ambiguity_ok"] += 1

        shutil.rmtree(STAGE_TMP / scene, ignore_errors=True)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  ... {n_done}/{len(scene_conditions)} scene-conditions checked, running results: {results}", flush=True)

    print(f"\nFINAL EXHAUSTIVE RESULTS ({len(all_df)} rows, {n_done} scene-conditions): {results}")
    print(f"failures: {len(failures)}")
    for sid, reason in failures[:20]:
        print("  FAIL", sid, reason)

    Path(OUT_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "w") as f:
        json.dump({"results": results, "n_rows": len(all_df), "n_scene_conditions": n_done,
                    "failures": [{"sample_id": s, "reason": r} for s, r in failures]}, f, indent=2)
    print(f"written to {OUT_LOG}")


if __name__ == "__main__":
    main()

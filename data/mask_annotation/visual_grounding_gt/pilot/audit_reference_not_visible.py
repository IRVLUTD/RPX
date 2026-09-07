"""Q9: verify that an Option-B reference object is not ITSELF also visible
in Image 2 (the target frame) as some OTHER object -- i.e. the reference's
catalog identity must not match any non-target visible object in that exact
frame. This is a distinct check from the existing "reference != target"
identity check: it additionally rules out a reference that happens to be a
different-instance duplicate of an object that's separately, legitimately
present elsewhere in the very same frame the model is shown.

Restricted to the 7 KEEP_STAGED scenes still locally cached (no restage) --
disclosed as a sample, same methodology as the other 7-scene checks in this
audit; a full-dataset version would need the same ~360MB mask-only restage
already used elsewhere in this audit.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))
from generate_spatial_gt import load_mapping, imread_mask, MIN_MASK_AREA_PX  # noqa: E402
import sos_catalog as sc  # noqa: E402

STAGE_ROOT = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged"
KEEP_STAGED = ["scene001", "scene004", "scene010", "scene016", "scene046", "scene069", "scene098"]


def main():
    catalog = sc.load_catalog()
    catalog_by_scid = sc.by_source_catalog_id(catalog)

    mos = pq.read_table("out/incontext_parquet/incontext_mos_attribute_bbox.parquet").to_pandas()
    ego = pq.read_table("out/incontext_parquet/incontext_ego_attribute_bbox.parquet").to_pandas()
    mos["phase_dir"] = mos["phase"].astype(int).astype(str)
    ego["phase_dir"] = "ego"
    import pandas as pd
    df = pd.concat([mos, ego], ignore_index=True)
    sub = df[df.scene_id.isin(KEEP_STAGED)]
    print(f"checking {len(sub)} rows from the {len(KEEP_STAGED)} KEEP_STAGED scenes", flush=True)

    n_checked, n_violations = 0, 0
    violations = []
    mask_cache = {}
    for _, row in sub.iterrows():
        root = f"{STAGE_ROOT}/{row.scene_id}/{row.phase_dir}"
        mapping_path = f"{root}/sam2/mask_to_object.json"
        if not os.path.exists(mapping_path):
            continue
        cache_key = (row.scene_id, row.phase_dir)
        if cache_key not in mask_cache:
            mask_cache[cache_key] = load_mapping(mapping_path)
        mapping = mask_cache[cache_key]

        mask_path = f"{root}/sam2/masks/{row.frame}.png"
        frame_cache_key = (row.scene_id, row.phase_dir, row.frame)
        if frame_cache_key not in mask_cache:
            if not os.path.exists(mask_path):
                mask_cache[frame_cache_key] = None
            else:
                m = imread_mask(mask_path)
                present_ids = [int(v) for v in np.unique(m) if v != 0 and int(v) in mapping
                               and int((m == v).sum()) >= MIN_MASK_AREA_PX]
                mask_cache[frame_cache_key] = present_ids
        present_ids = mask_cache[frame_cache_key]
        if present_ids is None:
            continue
        n_checked += 1

        ref_scid = row.reference_source_catalog_id
        target_local_id = int(row.target_local_mask_id)
        for oid in present_ids:
            if oid == target_local_id:
                continue
            other_identity = sc.join_scene_mask(mapping, oid, catalog_by_scid)
            if other_identity.source_catalog_id == ref_scid:
                n_violations += 1
                violations.append({
                    "sample_id": row.sample_id, "scene": row.scene_id, "frame": row.frame,
                    "reference_object_id": row.reference_object_id,
                    "also_visible_as_local_mask_id": oid,
                })
                break

    print(f"\nchecked: {n_checked} rows with resolvable masks")
    print(f"VIOLATIONS (reference also independently visible in Image 2): {n_violations}")
    for v in violations[:10]:
        print(" ", v)


if __name__ == "__main__":
    main()

"""Targeted correction for the 25-group / 50-row odd_one_out duplicate bug
found in the v2 canonical validation pass (see build_row_paired_general.py's
odd_one_out fix). Only 4 scenes (all ego) are affected. Restages just those
4 scenes' ego masks, reprocesses every published attr_odd_one_out row for
them with the FIXED question-text-parsing logic, and replaces the
duplicate/wrong rows in gt_incontext_general_rowpaired_v2.jsonl in place
(atomic write via a temp file + rename)."""
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, ".")
from build_row_paired_general import (SOS_ROOT, REF_MANIFEST, PUBLISHED, REVISION, gi,  # noqa: E402
                                       process_row, stage_masks_only, load_mapping)
import shutil

STAGE_TMP = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged_masks_only")
JSONL_PATH = Path("out/incontext_paired/gt_incontext_general_rowpaired_v2.jsonl")
AFFECTED_SCENES = {"scene026", "scene033", "scene071", "scene100"}


def main():
    from lib import load_fewsol_lookup
    import sos_catalog as sc
    import sos_reference as sr
    import build_row_paired_general as brp

    brp.gi_lookup = load_fewsol_lookup(SOS_ROOT)
    catalog = sc.load_catalog()
    catalog_by_scid = sc.by_source_catalog_id(catalog)
    ref_crops = {oid: sr.ReferenceCrop(**r) for oid, r in sr.load_reference_manifest(REF_MANIFEST).items()}

    # Remove EVERY existing odd_one_out row for the 4 affected scenes (not
    # just the ones flagged as sample_id-duplicates) -- reprocessing ALL of
    # the affected scenes' published odd_one_out rows below will recreate
    # the correct ones alongside the fixed ones; keeping the old, still-
    # correct rows around too would create a NEW duplicate between the old
    # (still valid) row and its freshly-reprocessed twin.
    rows = []
    with open(JSONL_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    kept_rows = [r for r in rows
                 if not (r["type"] == "inctx_attr_odd_one_out" and r["kind"] == "ego" and r["scene_id"] in AFFECTED_SCENES)]
    print(f"{len(rows)} -> {len(kept_rows)} rows after removing ALL odd_one_out/ego rows for the 4 affected scenes")

    pub = pq.read_table(PUBLISHED).to_pandas()
    pub = pub[(pub["type"] == "attr_odd_one_out") & (pub["kind"] == "ego") & (pub["scene_id"].isin(AFFECTED_SCENES))]
    print(f"{len(pub)} published attr_odd_one_out ego rows in the affected scenes to reprocess")

    new_rows = []
    for scene in AFFECTED_SCENES:
        dest = stage_masks_only(scene, "ego")
        mapping = load_mapping(str(dest / "sam2" / "mask_to_object.json"))
        mask_dir = dest / "sam2" / "masks"
        attrs_cache = {}
        sub = pub[pub.scene_id == scene]
        for _, pub_row in sub.iterrows():
            row, drop = process_row(pub_row, mapping, attrs_cache, mask_dir, catalog, catalog_by_scid, ref_crops)
            if row is not None:
                new_rows.append(row)
        shutil.rmtree(STAGE_TMP / scene, ignore_errors=True)
        print(f"  {scene}: reprocessed, {len(new_rows)} cumulative new rows")

    final_rows = kept_rows + new_rows
    final_ids = [r["sample_id"] for r in final_rows]
    assert len(final_ids) == len(set(final_ids)), "still has duplicates after the targeted fix!"

    tmp_path = JSONL_PATH.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for r in final_rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, JSONL_PATH)
    print(f"\n{len(rows)} -> {len(final_rows)} final rows (removed {len(rows)-len(kept_rows)} duplicates, "
          f"added {len(new_rows)} correctly-reprocessed rows)")
    print(f"0 duplicate sample_ids: {len(final_ids) == len(set(final_ids))}")


if __name__ == "__main__":
    main()

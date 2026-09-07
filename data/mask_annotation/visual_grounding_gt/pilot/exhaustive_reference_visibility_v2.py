"""Exhaustive (not sampled) reference-not-visible audit over EVERY row in
release_candidate_v2's general Parquets (315,472 mos + 112,420 ego).
Requires the true full per-frame visible-object roster for all 100 scenes --
restages ONLY sam2/masks+meta tars (~360MB total, no rgb/depth/fisheye),
scene-by-scene, deleted after each cell's rows are checked. Does NOT apply
to inctx_spatial_farthest (Image 1 is intentionally the anchor, which IS
present in Image 2 by design).
"""
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))
from generate_spatial_gt import load_mapping, imread_mask, MIN_MASK_AREA_PX  # noqa: E402
import sos_catalog as sc  # noqa: E402

CANON_DIR = "out/release_candidate_v2"
STAGE_TMP = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged_masks_only")
REVISION = "main"
OUT_LOG = "out/release_candidate_v2/exhaustive_reference_visibility.json"


def stage_masks_only(scene, phase_dir):
    from huggingface_hub import hf_hub_download
    prefix = f"scenes/{scene}/{'ego' if phase_dir == 'ego' else phase_dir}"
    dest = STAGE_TMP / scene / phase_dir
    if (dest / "sam2" / "mask_to_object.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            p = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=f"{prefix}/labels/masks/v1.tar", revision=REVISION)
            break
        except Exception as e:
            if "429" not in str(e) or attempt == 5:
                raise
            import time
            time.sleep(30 * (2 ** attempt))
    with tarfile.open(p) as tf:
        tf.extractall(dest / "sam2")
    if (dest / "sam2" / "sam2" / "masks").exists():
        (dest / "sam2" / "sam2" / "masks").rename(dest / "sam2" / "masks")
        shutil.rmtree(dest / "sam2" / "sam2", ignore_errors=True)
    real = os.path.realpath(p)
    os.remove(p)
    if os.path.exists(real):
        os.remove(real)
    for attempt in range(6):
        try:
            p2 = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=f"{prefix}/labels/sam2_meta/v1.tar", revision=REVISION)
            break
        except Exception as e:
            if "429" not in str(e) or attempt == 5:
                raise
            import time
            time.sleep(30 * (2 ** attempt))
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
    catalog = sc.load_catalog()
    catalog_by_scid = sc.by_source_catalog_id(catalog)

    mos = pq.read_table(f"{CANON_DIR}/incontext_mos_attribute_bbox.parquet",
                         columns=["sample_id", "scene_id", "phase", "frame", "reference_source_catalog_id",
                                  "target_local_mask_id"]).to_pandas()
    ego = pq.read_table(f"{CANON_DIR}/incontext_ego_attribute_bbox.parquet",
                         columns=["sample_id", "scene_id", "phase", "frame", "reference_source_catalog_id",
                                  "target_local_mask_id"]).to_pandas()
    mos["phase_dir"] = mos["phase"].astype(int).astype(str)
    mos["kind"] = "mos"
    ego["phase_dir"] = "ego"
    ego["kind"] = "ego"
    print(f"MOS rows to check: {len(mos)}  Ego rows to check: {len(ego)}", flush=True)

    results = {"mos": {"checked": 0, "violations": 0}, "ego": {"checked": 0, "violations": 0}}
    violations = {"mos": [], "ego": []}

    for kind, df in [("mos", mos), ("ego", ego)]:
        groups = df.groupby(["scene_id", "phase_dir"])
        scene_conditions = list(groups.groups.keys())
        print(f"[{kind}] {len(scene_conditions)} scene-conditions", flush=True)
        for i, (scene, phase_dir) in enumerate(scene_conditions):
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
                        m = imread_mask(str(mask_path))
                        present_ids = [int(v) for v in np.unique(m) if v != 0 and int(v) in mapping
                                       and int((m == v).sum()) >= MIN_MASK_AREA_PX]
                        mask_cache[fid] = present_ids
                present_ids = mask_cache[fid]
                if present_ids is None:
                    continue
                results[kind]["checked"] += 1
                ref_scid = row["reference_source_catalog_id"]
                target_local_id = int(row["target_local_mask_id"])
                for oid in present_ids:
                    if oid == target_local_id:
                        continue
                    if mapping[oid]["oid"] == ref_scid:
                        results[kind]["violations"] += 1
                        violations[kind].append({"sample_id": row["sample_id"], "scene": scene,
                                                  "frame": fid, "also_visible_as": oid})
                        break
            shutil.rmtree(STAGE_TMP / scene, ignore_errors=True)
            if (i + 1) % 20 == 0:
                print(f"  [{kind}] ... {i+1}/{len(scene_conditions)} scene-conditions, "
                      f"checked={results[kind]['checked']} violations={results[kind]['violations']}", flush=True)

    print(f"\n=== FINAL EXHAUSTIVE REFERENCE-NOT-VISIBLE RESULTS ===")
    print(f"MOS: {results['mos']['checked']} checked, {results['mos']['violations']} violations")
    print(f"Ego: {results['ego']['checked']} checked, {results['ego']['violations']} violations")

    with open(OUT_LOG, "w") as f:
        json.dump({"results": results, "violations": violations}, f, indent=2)
    print(f"written to {OUT_LOG}")


if __name__ == "__main__":
    main()

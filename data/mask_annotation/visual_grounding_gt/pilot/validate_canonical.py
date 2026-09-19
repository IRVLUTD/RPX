"""Q11/Q12: independent validation of ONLY the 3 canonical release-candidate
files, including zero-row frames in per-frame minima (the true universe of
frames per scene-condition is taken from the PUBLISHED attribute.parquet /
spatial_bbox.parquet's own distinct-frame counts, since that's the ground
truth of "how many frames exist" for that scene-condition -- a frame this
canonical file has zero rows for must still count as 0, not be omitted).

Records the full SHA-256 of every input file it validated (Q12).
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
import sos_catalog as sc  # noqa: E402

CANON_DIR = "out/release_candidate_v1"
REF_MANIFEST = "out/reference_crops/reference_crops_v1.parquet"
PUBLISHED_ATTR = "out/vqa_parquet/attribute.parquet"
PUBLISHED_SPATIAL = "out/vqa_parquet/spatial_bbox.parquet"
OUT_REPORT = "out/release_candidate_v1/VALIDATION_REPORT.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def universe_frames(kind, phase, published_attr):
    sub = published_attr[(published_attr.kind == kind)]
    if kind == "mos":
        sub = sub[sub.phase == phase]
    g = sub.drop_duplicates(subset=["scene_id", "frame"])
    return g.groupby("scene_id")["frame"].apply(set).to_dict()


def main():
    report = {"input_file_hashes": {}, "checks": {}}

    files = {
        "incontext_mos_attribute_bbox": f"{CANON_DIR}/incontext_mos_attribute_bbox.parquet",
        "incontext_ego_attribute_bbox": f"{CANON_DIR}/incontext_ego_attribute_bbox.parquet",
        "incontext_mos_spatial_bbox": f"{CANON_DIR}/incontext_mos_spatial_bbox.parquet",
        "reference_crops_v1.parquet": REF_MANIFEST,
        "attribute.parquet (published, for frame-universe ground truth)": PUBLISHED_ATTR,
        "spatial_bbox.parquet (published, for frame-universe ground truth)": PUBLISHED_SPATIAL,
    }
    print("=== Q12: SHA-256 of every input this validator reads ===")
    for label, path in files.items():
        h = sha256_of(path)
        report["input_file_hashes"][label] = {"path": os.path.abspath(path), "sha256": h}
        print(f"  {h}  {os.path.abspath(path)}")

    catalog = sc.load_catalog()
    ref_df = pq.read_table(REF_MANIFEST).to_pandas()
    ref_sha_by_obj = dict(zip(ref_df["object_id"], ref_df["crop_sha256"]))
    published_attr = pq.read_table(PUBLISHED_ATTR, columns=["scene_id", "kind", "phase", "frame"]).to_pandas()
    published_spatial = pq.read_table(PUBLISHED_SPATIAL, columns=["scene_id", "kind", "phase", "frame", "type"]).to_pandas()
    published_spatial = published_spatial[published_spatial.type == "spatial_farthest"]

    for name in ["incontext_mos_attribute_bbox", "incontext_ego_attribute_bbox", "incontext_mos_spatial_bbox"]:
        path = files[name]
        df = pq.read_table(path).to_pandas()
        print(f"\n{'='*80}\n{name} ({len(df)} rows)\n{'='*80}")
        r = {}
        r["rows"] = len(df)
        r["by_type_kind_phase"] = {str(k): int(v) for k, v in df.groupby(["type", "kind", "phase"], dropna=False).size().items()}

        # bbox
        bad_bbox = []
        for i, row in df.iterrows():
            x0, y0, x1, y1 = row["answer_bbox"]
            if not (0 <= x0 <= x1 < row["img_w"] and 0 <= y0 <= y1 < row["img_h"]):
                bad_bbox.append(row["sample_id"])
        r["bbox_violations"] = len(bad_bbox)
        print(f"bbox violations: {len(bad_bbox)}")

        # identity
        same_id = df[df["reference_global_object_id"].notna() & df["target_global_object_id"].notna() &
                      (df["reference_global_object_id"] == df["target_global_object_id"])]
        same_scid = df[df["reference_source_catalog_id"] == df["target_source_catalog_id"]]
        r["identity_violations_goid"] = len(same_id)
        r["identity_violations_scid"] = len(same_scid)
        print(f"identity violations: goid={len(same_id)} scid={len(same_scid)}")

        # dup sample id / semantic
        r["duplicate_sample_ids"] = int(df["sample_id"].duplicated().sum())
        key_cols = ["scene_id", "kind", "phase", "frame", "type", "target_source_catalog_id",
                    "reference_source_catalog_id", "attribute_value", "attribute_material", "attribute_function"]
        key_cols = [c for c in key_cols if c in df.columns]
        r["duplicate_semantic_rows_within_frame"] = int(df.duplicated(subset=key_cols, keep=False).sum())
        print(f"duplicate sample_ids: {r['duplicate_sample_ids']}  duplicate semantic rows: {r['duplicate_semantic_rows_within_frame']}")

        # assets
        missing_assets = df[~df["reference_object_id"].map(lambda o: ref_sha_by_obj.get(o) is not None if isinstance(o, str) else False)]
        sha_mismatch = df[df.apply(lambda row: isinstance(row["reference_object_id"], str)
                                    and ref_sha_by_obj.get(row["reference_object_id"]) != row["reference_crop_sha256"], axis=1)]
        r["missing_reference_assets"] = len(missing_assets)
        r["reference_sha256_mismatches"] = len(sha_mismatch)
        print(f"missing reference assets: {len(missing_assets)}  sha256 mismatches: {len(sha_mismatch)}")

        # per-frame counts INCLUDING zero-row frames from the published universe
        kind_for_universe = "mos" if "mos" in name else "ego"
        if kind_for_universe == "mos" and "spatial" not in name:
            phases = [0, 1, 2]
        elif "spatial" in name:
            phases = [0, 1, 2]
        else:
            phases = [None]

        per_frame_counts = []
        zero_cells = []
        for phase in phases:
            universe = universe_frames(kind_for_universe, phase, published_attr if "spatial" not in name else published_spatial)
            if kind_for_universe == "mos":
                sub = df[df.phase == phase] if "spatial" not in name or True else df
            else:
                sub = df
            actual_counts = sub.groupby(["scene_id", "frame"]).size().to_dict()
            for scene, frames in universe.items():
                for fr in frames:
                    n = actual_counts.get((scene, fr), 0)
                    per_frame_counts.append(n)
                    if n == 0:
                        zero_cells.append((scene, kind_for_universe, phase, fr))
        per_frame_counts = np.array(per_frame_counts) if per_frame_counts else np.array([0])
        r["per_frame_min_including_zero"] = int(per_frame_counts.min())
        r["per_frame_median_including_zero"] = float(np.median(per_frame_counts))
        r["per_frame_max"] = int(per_frame_counts.max())
        r["zero_row_frame_count"] = len(zero_cells)
        r["total_universe_frames"] = len(per_frame_counts)
        print(f"per-frame (INCLUDING zero-row frames): min={r['per_frame_min_including_zero']} "
              f"median={r['per_frame_median_including_zero']:.1f} max={r['per_frame_max']}  "
              f"zero-row frames: {len(zero_cells)}/{len(per_frame_counts)}")

        # centered coverage / 30-frame check
        centered = df[df["answer_is_centered"] == True]  # noqa: E712
        ccov = centered.drop_duplicates(subset=["scene_id", "kind", "phase", "frame"] if "phase" in df.columns else ["scene_id", "kind", "frame"])
        if "spatial" in name or kind_for_universe == "mos":
            group_cols = ["scene_id", "phase"]
        else:
            group_cols = ["scene_id"]
        ccounts = ccov.groupby(group_cols).size()
        deficient = ccounts[ccounts < 30]
        all_scenes = sorted(df.scene_id.unique())
        expected_cells = len(all_scenes) * (3 if "phase" in group_cols else 1)
        present_cells = len(ccounts)
        r["centered_ge30_cells"] = int((ccounts >= 30).sum())
        r["centered_lt30_cells"] = int(len(deficient)) + (expected_cells - present_cells)
        r["expected_cells"] = expected_cells
        print(f"cells with >=30 centered frames: {r['centered_ge30_cells']}/{expected_cells}  "
              f"deficient: {r['centered_lt30_cells']}")

        report["checks"][name] = r

    os.makedirs(CANON_DIR, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfull report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()

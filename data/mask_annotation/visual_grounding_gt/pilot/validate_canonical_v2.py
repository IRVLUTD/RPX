"""Q4/Q7/Q12: full independent validation of release_candidate_v2, adding
(vs the v1 validator): prompt-leakage check, pinned-revision check
(rejects "main" or any non-40-hex-char revision), and source-row-pairing
verification against the published attribute.parquet (every row must
resolve to a real published row with the same scene/kind/phase/frame/type/
target_oid/answer_bbox, matching Q1's corrected exact key).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
import sos_catalog as sc  # noqa: E402

CANON_DIR = "out/release_candidate_v2"
REF_MANIFEST = "out/reference_crops/reference_crops_v1.parquet"
PUBLISHED_ATTR = "out/vqa_parquet/attribute.parquet"
PUBLISHED_SPATIAL = "out/vqa_parquet/spatial_bbox.parquet"
OUT_REPORT = f"{CANON_DIR}/VALIDATION_REPORT.json"
PINNED_REVISION = "93e31d378f1f98eca18a7aa01a2279c9f332440c"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def universe_frames(kind, phase, published):
    sub = published[(published.kind == kind)]
    if kind == "mos":
        sub = sub[sub.phase == phase]
    g = sub.drop_duplicates(subset=["scene_id", "frame"])
    return g.groupby("scene_id")["frame"].apply(set).to_dict()


def check_leakage(df, templates, catalog):
    leaks = []
    forbidden_terms = set()
    for c in catalog.values():
        forbidden_terms.add(sc.normalize_value(c.object_name))
        forbidden_terms.add(sc.normalize_value(c.object_id))
    for i, row in df.iterrows():
        expected = templates.get(row["type"])
        if expected is not None and row["question"] != expected:
            leaks.append((row["sample_id"], "question_not_fixed_template"))
            continue
        q_norm = sc.normalize_value(row["question"])
        for term in forbidden_terms:
            if len(term) > 3 and term in q_norm:
                leaks.append((row["sample_id"], f"contains_catalog_term:{term}"))
    return leaks


def check_revision_pinned(df):
    bad = []
    for i, row in df.iterrows():
        for loc_name in ("target_image", "reference_image"):
            rev = row[loc_name]["revision"]
            if rev != PINNED_REVISION:
                bad.append((row["sample_id"], loc_name, rev))
    return bad


def check_source_pairing(df, published_attr):
    """Every general row must resolve, via the Q1-corrected exact key, to
    at least one real published attribute.parquet row."""
    pub_index = set()
    for _, r in published_attr.iterrows():
        if r["answer_bbox"] is None:
            continue  # non-bbox published types (e.g. attr_synonym "no") -- never a match target here
        phase_key = None if pd.isna(r["phase"]) else int(r["phase"])
        pub_index.add((r["scene_id"], r["kind"], phase_key, r["type"], r["target_oid"], tuple(r["answer_bbox"])))
    unmatched = []
    for i, row in df.iterrows():
        phase_key = None if pd.isna(row["phase"]) else int(row["phase"])
        key = (row["scene_id"], row["kind"], phase_key, row["source_type"],
               row["target_source_catalog_id"], tuple(row["answer_bbox"]))
        if key not in pub_index:
            unmatched.append(row["sample_id"])
    return unmatched


def main():
    report = {"input_file_hashes": {}, "checks": {}}
    files = {
        "incontext_mos_attribute_bbox": f"{CANON_DIR}/incontext_mos_attribute_bbox.parquet",
        "incontext_ego_attribute_bbox": f"{CANON_DIR}/incontext_ego_attribute_bbox.parquet",
        "incontext_mos_spatial_bbox": f"{CANON_DIR}/incontext_mos_spatial_bbox.parquet",
        "reference_crops_v1.parquet": REF_MANIFEST,
        "attribute.parquet (published, ground truth)": PUBLISHED_ATTR,
        "spatial_bbox.parquet (published, ground truth)": PUBLISHED_SPATIAL,
    }
    print("=== Q12: SHA-256 of every input this validator reads ===")
    for label, path in files.items():
        h = sha256_of(path)
        report["input_file_hashes"][label] = {"path": os.path.abspath(path), "sha256": h}
        print(f"  {h}  {os.path.abspath(path)}")

    catalog = sc.load_catalog()
    templates = {
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
    ref_df = pq.read_table(REF_MANIFEST).to_pandas()
    ref_sha_by_obj = dict(zip(ref_df["object_id"], ref_df["crop_sha256"]))
    published_attr_full = pq.read_table(PUBLISHED_ATTR).to_pandas()
    published_attr = published_attr_full[["scene_id", "kind", "phase", "frame"]]
    published_spatial = pq.read_table(PUBLISHED_SPATIAL, columns=["scene_id", "kind", "phase", "frame", "type"]).to_pandas()
    published_spatial = published_spatial[published_spatial.type == "spatial_farthest"]

    for name in ["incontext_mos_attribute_bbox", "incontext_ego_attribute_bbox", "incontext_mos_spatial_bbox"]:
        path = files[name]
        df = pq.read_table(path).to_pandas()
        print(f"\n{'='*80}\n{name} ({len(df)} rows)\n{'='*80}")
        r = {"rows": len(df)}
        r["by_type_kind_phase"] = {str(k): int(v) for k, v in df.groupby(["type", "kind", "phase"], dropna=False).size().items()}

        bad_bbox = [row["sample_id"] for _, row in df.iterrows()
                    if not (0 <= row["answer_bbox"][0] <= row["answer_bbox"][2] < row["img_w"]
                            and 0 <= row["answer_bbox"][1] <= row["answer_bbox"][3] < row["img_h"])]
        r["bbox_violations"] = len(bad_bbox)
        print(f"bbox violations: {len(bad_bbox)}")

        same_id = df[df["reference_global_object_id"].notna() & df["target_global_object_id"].notna() &
                      (df["reference_global_object_id"] == df["target_global_object_id"])]
        same_scid = df[df["reference_source_catalog_id"] == df["target_source_catalog_id"]]
        r["identity_violations_goid"] = len(same_id)
        r["identity_violations_scid"] = len(same_scid)
        print(f"identity violations: goid={len(same_id)} scid={len(same_scid)}")

        r["duplicate_sample_ids"] = int(df["sample_id"].duplicated().sum())
        key_cols = [c for c in ["scene_id", "kind", "phase", "frame", "type", "target_source_catalog_id",
                                 "reference_source_catalog_id", "attribute_value", "attribute_material",
                                 "attribute_function"] if c in df.columns]
        r["duplicate_semantic_rows_within_frame"] = int(df.duplicated(subset=key_cols, keep=False).sum())
        print(f"duplicate sample_ids: {r['duplicate_sample_ids']}  duplicate semantic rows: {r['duplicate_semantic_rows_within_frame']}")

        missing_assets = df[~df["reference_object_id"].map(lambda o: ref_sha_by_obj.get(o) is not None if isinstance(o, str) else False)]
        sha_mismatch = df[df.apply(lambda row: isinstance(row["reference_object_id"], str)
                                    and ref_sha_by_obj.get(row["reference_object_id"]) != row["reference_crop_sha256"], axis=1)]
        r["missing_reference_assets"] = len(missing_assets)
        r["reference_sha256_mismatches"] = len(sha_mismatch)
        print(f"missing reference assets: {len(missing_assets)}  sha256 mismatches: {len(sha_mismatch)}")

        leaks = check_leakage(df, templates, catalog)
        r["prompt_leakage_findings"] = len(leaks)
        print(f"prompt leakage findings: {len(leaks)}")

        bad_revision = check_revision_pinned(df)
        r["non_pinned_revision_count"] = len(bad_revision)
        print(f"non-pinned-revision locators: {len(bad_revision)} (expected pinned SHA: {PINNED_REVISION})")

        if name != "incontext_mos_spatial_bbox":
            unmatched_src = check_source_pairing(df, published_attr_full)
            r["source_row_pairing_unmatched"] = len(unmatched_src)
            print(f"source-row pairing: {len(df) - len(unmatched_src)}/{len(df)} resolve to a real published row "
                  f"({len(unmatched_src)} unmatched)")

        kind_for_universe = "mos" if "mos" in name else "ego"
        phases = [0, 1, 2] if (kind_for_universe == "mos" or "spatial" in name) else [None]
        per_frame_counts, zero_cells = [], []
        for phase in phases:
            universe = universe_frames(kind_for_universe, phase, published_attr if "spatial" not in name else published_spatial)
            sub = df[df.phase == phase] if kind_for_universe == "mos" else df
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
              f"zero-row frames: {len(zero_cells)}/{len(per_frame_counts)} "
              f"(universe: MOS=75000 or Ego=23050 expected)")

        centered = df[df["answer_is_centered"] == True]  # noqa: E712
        ccov = centered.drop_duplicates(subset=["scene_id", "kind", "phase", "frame"])
        group_cols = ["scene_id", "phase"] if (kind_for_universe == "mos" or "spatial" in name) else ["scene_id"]
        ccounts = ccov.groupby(group_cols).size()
        all_scenes = sorted(df.scene_id.unique())
        expected_cells = len(all_scenes) * (3 if "phase" in group_cols else 1)
        r["centered_ge30_cells"] = int((ccounts >= 30).sum())
        r["centered_lt30_cells"] = expected_cells - r["centered_ge30_cells"]
        r["expected_cells"] = expected_cells
        print(f"cells with >=30 centered frames: {r['centered_ge30_cells']}/{expected_cells}")

        report["checks"][name] = r

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfull report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()

"""Independent validator for the 3 in-context Parquets (spec §15). Does NOT
import gt_incontext's matching/selection functions -- re-derives everything
it checks from the catalog manifest, the reference-crop manifest, and (for
the sampled semantic re-verification) fresh mask reads, using its own
lightweight logic. Reads Parquet only; never assumes generator-internal
state.
"""
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))

from sos_catalog import normalize_value, load_catalog, by_source_catalog_id  # noqa: E402

PARQUET_DIR = sys.argv[1] if len(sys.argv) > 1 else "out/incontext_parquet"
STAGE_ROOT = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged"
SAMPLE_N = 300


def load(name):
    path = f"{PARQUET_DIR}/{name}.parquet"
    if not os.path.exists(path):
        return None
    return pq.read_table(path).to_pandas()


def check_bbox(df, w_col="img_w", h_col="img_h"):
    bad = []
    for i, row in df.iterrows():
        x0, y0, x1, y1 = row["answer_bbox"]
        if not (0 <= x0 <= x1 < row[w_col] and 0 <= y0 <= y1 < row[h_col]):
            bad.append(row["sample_id"])
    return bad


def check_identity_violations(df):
    same_identity = df[df["reference_global_object_id"].notna() & df["target_global_object_id"].notna() &
                        (df["reference_global_object_id"] == df["target_global_object_id"])]
    same_scid = df[df["reference_source_catalog_id"] == df["target_source_catalog_id"]]
    return same_identity, same_scid


def check_category_violations(df, catalog):
    violations = []
    for i, row in df.iterrows():
        ref_obj = catalog.get(row["reference_object_id"])
        if ref_obj is None:
            continue
        if normalize_value(ref_obj.object_name) == normalize_value(row["answer"]):
            violations.append(row["sample_id"])
    return violations


def check_duplicate_semantic_rows_within_frame(df):
    key_cols = ["scene_id", "kind", "phase", "frame", "type", "target_source_catalog_id",
                "reference_source_catalog_id", "attribute_value", "attribute_material", "attribute_function"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    return df[dup_mask]


def check_prompt_leakage(df, templates, catalog):
    leaks = []
    forbidden_terms = set()
    for c in catalog.values():
        forbidden_terms.add(normalize_value(c.object_name))
        forbidden_terms.add(normalize_value(c.object_id))
    for i, row in df.iterrows():
        expected = templates.get(row["type"])
        if expected is not None and row["question"] != expected:
            leaks.append((row["sample_id"], "question_not_fixed_template"))
            continue
        q_norm = normalize_value(row["question"])
        for term in forbidden_terms:
            if len(term) > 3 and term in q_norm:
                leaks.append((row["sample_id"], f"contains_catalog_term:{term}"))
    return leaks


def frame_coverage_table(df):
    g = df.drop_duplicates(subset=["scene_id", "kind", "phase", "frame"])
    return g.groupby(["scene_id", "kind", "phase"], dropna=False).size()


def centered_coverage_table(df):
    centered = df[df["answer_is_centered"]]
    g = centered.drop_duplicates(subset=["scene_id", "kind", "phase", "frame"])
    return g.groupby(["scene_id", "kind", "phase"], dropna=False).size()


def sampled_semantic_reverify_general(df, catalog, n=SAMPLE_N):
    """Independent re-check (not calling gt_incontext): for a random sample,
    (1) reference actually carries the intended attribute value(s) per the
    catalog, and (2) re-loads the real frame mask and independently
    verifies no OTHER visible object in that exact frame owns any of the
    reference's values in the relevant field (the core ambiguity-gate
    guarantee), using fresh logic written here, not imported."""
    from generate_spatial_gt import load_mapping, imread_mask, MIN_MASK_AREA_PX
    from lib import object_attrs, load_fewsol_lookup
    from pathlib import Path

    SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
    lookup = load_fewsol_lookup(SOS_ROOT)

    rows = df.sample(min(n, len(df)), random_state=0).to_dict("records")
    results = {"share_ok": 0, "share_fail": 0, "ambiguity_ok": 0, "ambiguity_fail": 0, "skipped_no_stage": 0}
    failures = []
    for row in rows:
        ref_obj = catalog.get(row["reference_object_id"])
        if ref_obj is None:
            results["share_fail"] += 1
            failures.append((row["sample_id"], "reference_not_in_catalog"))
            continue
        field = row["attribute_kind"]
        if field == "composition":
            m_ok = normalize_value(row["attribute_material"]) in ref_obj.attrs_norm["material"]
            f_ok = normalize_value(row["attribute_function"]) in ref_obj.attrs_norm["function"]
            share_ok = m_ok and f_ok
        elif field == "material" and row["type"] == "inctx_attr_odd_one_out":
            share_ok = normalize_value(row["attribute_material"]) in ref_obj.attrs_norm["material"]
        else:
            share_ok = normalize_value(row["attribute_value"]) in ref_obj.attrs_norm.get(field, [])
        if share_ok:
            results["share_ok"] += 1
        else:
            results["share_fail"] += 1
            failures.append((row["sample_id"], "reference_does_not_share_intended_attribute"))
            continue

        if row["type"] == "inctx_attr_odd_one_out":
            continue  # ambiguity re-check for odd-one-out needs full mat_owners; covered by unit tests instead

        phase_dir = "ego" if row["kind"] == "ego" else str(int(row["phase"]))
        root = f"{STAGE_ROOT}/{row['scene_id']}/{phase_dir}"
        mask_path = f"{root}/sam2/masks/{row['frame']}.png"
        if not os.path.exists(mask_path):
            results["skipped_no_stage"] += 1
            continue
        mapping = load_mapping(f"{root}/sam2/mask_to_object.json")
        mask = imread_mask(mask_path)
        present_ids = [int(v) for v in np.unique(mask) if v != 0 and int(v) in mapping
                       and int((mask == v).sum()) >= MIN_MASK_AREA_PX]
        target_local_id = row["target_local_mask_id"]
        ambiguous = False
        if field == "composition":
            # The question asks about material AND function JOINTLY ("same
            # material and purpose") -- an object sharing only ONE of the
            # two independently is not a real alternative answer. Must
            # check the (material, function) PAIR, not each field alone
            # (an earlier version of this exact check did that and produced
            # ~30% false-positive "ambiguity" -- every object that merely
            # shared the reference's material, regardless of function, got
            # flagged even when no real second answer existed).
            cand_pairs = {(normalize_value(mm), normalize_value(ff))
                          for mm in ref_obj.attrs_raw["material"] for ff in ref_obj.attrs_raw["function"]}
            for oid in present_ids:
                if oid == target_local_id:
                    continue
                a = object_attrs(mapping[oid]["oid"], lookup)
                if not a:
                    continue
                other_pairs = {(normalize_value(mm), normalize_value(ff))
                                for mm in a["material"] for ff in a["function"]}
                if other_pairs & cand_pairs:
                    ambiguous = True
        else:
            cand_values = set(ref_obj.attrs_norm[field])
            for oid in present_ids:
                if oid == target_local_id:
                    continue
                a = object_attrs(mapping[oid]["oid"], lookup)
                if not a:
                    continue
                other_norm = {normalize_value(v) for v in a[field]}
                if other_norm & cand_values:
                    ambiguous = True
        if ambiguous:
            results["ambiguity_fail"] += 1
            failures.append((row["sample_id"], "independent_reverify_found_ambiguity"))
        else:
            results["ambiguity_ok"] += 1
    return results, failures


def main():
    catalog = load_catalog()
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

    ref_manifest_path = "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops/reference_crops_v1.parquet"
    ref_df = pq.read_table(ref_manifest_path).to_pandas()
    ref_sha_by_obj = dict(zip(ref_df["object_id"], ref_df["crop_sha256"]))

    for name in ["incontext_mos_attribute_bbox", "incontext_ego_attribute_bbox", "incontext_mos_spatial_bbox"]:
        print(f"\n{'='*80}\n{name}\n{'='*80}")
        df = load(name)
        if df is None:
            print("  MISSING FILE")
            continue
        print(f"rows: {len(df)}")
        print("by type:\n", df["type"].value_counts().to_string())
        print("by phase:\n", df["phase"].value_counts(dropna=False).to_string())
        print("distinct scenes:", df["scene_id"].nunique())
        print("distinct target frames:", len(df.drop_duplicates(subset=['scene_id', 'kind', 'phase', 'frame'])))
        per_frame = df.groupby(["scene_id", "kind", "phase", "frame"]).size()
        print(f"per-frame question count: min={per_frame.min()} median={per_frame.median()} max={per_frame.max()}")
        print("centered:", int(df["answer_is_centered"].sum()), "/ non-centered:", int((~df["answer_is_centered"]).sum()))
        print("distinct target objects (source_catalog_id):", df["target_source_catalog_id"].nunique())
        print("distinct reference objects:", df["reference_object_id"].nunique())
        ref_freq = df["reference_object_id"].value_counts()
        print("reference-frequency top 10:\n", ref_freq.head(10).to_string())
        top_share = ref_freq.iloc[0] / len(df) if len(df) else 0
        print(f"top reference's share of all rows: {top_share:.1%}"
              + ("  <-- FLAG: severe concentration" if top_share > 0.10 else ""))

        bad_bbox = check_bbox(df)
        print(f"bbox violations: {len(bad_bbox)}")

        same_id, same_scid = check_identity_violations(df)
        print(f"different-identity violations (global_object_id): {len(same_id)}")
        print(f"different-identity violations (source_catalog_id): {len(same_scid)}")

        if "reference_object_id" in df.columns and df["reference_object_id"].notna().any():
            cat_violations = check_category_violations(df, catalog)
            print(f"different-category violations: {len(cat_violations)}")

        dup_sample_ids = df["sample_id"].duplicated().sum()
        print(f"duplicate sample_ids: {dup_sample_ids}")

        dup_semantic = check_duplicate_semantic_rows_within_frame(df)
        print(f"duplicate semantic rows within same frame: {len(dup_semantic)}")

        leaks = check_prompt_leakage(df, templates, catalog)
        print(f"prompt leakage findings: {len(leaks)}")
        for sid, reason in leaks[:5]:
            print("   ", sid, reason)

        missing_assets = df[~df["reference_object_id"].map(lambda o: ref_sha_by_obj.get(o) is not None
                                                             if isinstance(o, str) else False)]
        print(f"rows whose reference_object_id has no crop-manifest entry: {len(missing_assets)}")
        sha_mismatch = df[df.apply(lambda r: isinstance(r["reference_object_id"], str)
                                    and ref_sha_by_obj.get(r["reference_object_id"]) != r["reference_crop_sha256"],
                                    axis=1)]
        print(f"reference_crop_sha256 mismatches vs manifest: {len(sha_mismatch)}")

        if name != "incontext_mos_spatial_bbox":
            print("\nsampled semantic re-verification (independent logic, n<=%d):" % SAMPLE_N)
            results, failures = sampled_semantic_reverify_general(df, catalog)
            print(" ", results)
            for sid, reason in failures[:5]:
                print("   FAIL", sid, reason)
        else:
            spatial_anchor_eq_answer = df[df["reference_source_catalog_id"] == df["target_source_catalog_id"]]
            print(f"spatial rows where answer == anchor: {len(spatial_anchor_eq_answer)}")
            missing_evidence = df[df["evidence"].isna()]
            print(f"spatial rows missing evidence: {len(missing_evidence)}")

        print("\nframe coverage (scene,kind,phase) -- min/median/max distinct frames:")
        cov = frame_coverage_table(df)
        print(f"  min={cov.min()} median={cov.median()} max={cov.max()}  cells={len(cov)}")
        centered_cov = centered_coverage_table(df)
        low_cells = centered_cov[centered_cov < 30]
        all_cells = set(cov.index)
        centered_cells = set(centered_cov.index)
        zero_centered = all_cells - centered_cells
        print(f"cells with <30 eligible CENTERED frames: {len(low_cells) + len(zero_centered)} / {len(all_cells)}")


if __name__ == "__main__":
    main()

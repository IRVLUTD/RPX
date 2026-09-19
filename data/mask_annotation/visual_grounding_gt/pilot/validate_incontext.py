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
KEEP_STAGED = {"scene001", "scene004", "scene010", "scene016", "scene046", "scene069", "scene098"}


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

        if row["type"] == "inctx_attr_odd_one_out":
            # Odd-one-out: the intended fact is "material M is majority,
            # target lacks it". Ambiguous iff SOME OTHER material the
            # reference also owns has its own well-defined odd-one-out in
            # this frame (>=2 owners, exactly 1 non-owner) pointing at a
            # DIFFERENT object than the intended target.
            def mat_owners_local():
                o = {}
                for oid in present_ids:
                    a = object_attrs(mapping[oid]["oid"], lookup)
                    if not a:
                        continue
                    for m in a["material"]:
                        o.setdefault(m, set()).add(oid)
                return o
            mo = mat_owners_local()
            for cand_m_norm in ref_obj.attrs_norm["material"]:
                for raw_m, owners in mo.items():
                    if normalize_value(raw_m) != cand_m_norm:
                        continue
                    missing = [oid for oid in present_ids if oid not in owners]
                    if len(missing) == 1 and len(owners) >= 2 and missing[0] != target_local_id:
                        ambiguous = True
            if ambiguous:
                results["ambiguity_fail"] += 1
                failures.append((row["sample_id"], "independent_reverify_found_odd_one_out_ambiguity"))
            else:
                results["ambiguity_ok"] += 1
            continue

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


def sampled_spatial_geometric_reverify(df, n=SAMPLE_N):
    """Independent re-check of the 2D/3D farthest-agreement rule: reloads
    the real staged mask+depth and recomputes compute_farthest_candidates()
    from scratch (gt_spatial.py's actual geometry code, but this is
    re-deriving the SET of valid (anchor,answer) pairs independently of
    whatever gt_incontext.py happened to emit -- not trusting the row's own
    evidence dict). Restricted to the 7 KEEP_STAGED scenes still on local
    disk (the other 93 were deleted after generation per the disk-bounded
    design) -- disclosed explicitly, not silently narrowed."""
    from generate_spatial_gt import load_mapping, imread_mask, imread_depth, compute_instances
    from gt_spatial import compute_farthest_candidates, _add_position_proxies

    sub = df[df["scene_id"].isin(KEEP_STAGED)]
    rows = sub.sample(min(n, len(sub)), random_state=1).to_dict("records") if len(sub) else []
    results = {"agree": 0, "disagree": 0, "answer_eq_anchor": 0, "checked": len(rows), "available_pool": len(sub)}
    failures = []
    cache = {}
    for row in rows:
        key = (row["scene_id"], row["phase"], row["frame"])
        if key not in cache:
            root = f"{STAGE_ROOT}/{row['scene_id']}/{int(row['phase'])}"
            mapping = load_mapping(f"{root}/sam2/mask_to_object.json")
            mask = imread_mask(f"{root}/sam2/masks/{row['frame']}.png")
            depth = imread_depth(f"{root}/depth/{row['frame']}.png")
            instances = compute_instances(mask, mapping, depth)
            inst_list = list(instances.values())
            H, W = mask.shape[:2]
            _add_position_proxies(instances, W, H)
            depth_vals = [(i["median_depth"], i) for i in inst_list if i["median_depth"] is not None]
            pairs = {(ref["name"], top["name"]) for ref, top, _, _ in compute_farthest_candidates(inst_list, depth_vals)}
            cache[key] = pairs
        pairs = cache[key]
        anchor_name = row["evidence"]["reference"] if isinstance(row["evidence"], dict) else None
        if anchor_name == row["answer"]:
            results["answer_eq_anchor"] += 1
            failures.append((row["sample_id"], "answer_equals_anchor"))
            continue
        if (anchor_name, row["answer"]) in pairs:
            results["agree"] += 1
        else:
            results["disagree"] += 1
            failures.append((row["sample_id"], f"not_in_recomputed_2d3d_agreement_set: {anchor_name}->{row['answer']}"))
    return results, failures


def full_cell_coverage(df, expected_scenes, conditions, min_centered=30):
    """conditions: list of (kind, phase) pairs defining every expected
    scene-condition cell. Returns (deficient_cells, total_cells) where a
    deficient cell has <min_centered distinct centered-answer target frames
    -- including cells with ZERO rows at all, which a naive groupby would
    silently omit rather than flag."""
    import pandas as pd
    centered = df[df["answer_is_centered"] == True].copy()  # noqa: E712
    # normalize phase to a hashable, comparable key: -1 sentinel for ego
    # (never None/NaN -- verified a column mixing None with real ints across
    # rows gets silently upcast to float64 by pandas on column assignment,
    # turning None into NaN; NaN != NaN as a dict key, so a later
    # counts.get((scene,"ego",None), 0) lookup would never match and every
    # ego cell would read as 0/deficient even when fully covered).
    centered["_phase_key"] = centered["phase"].apply(lambda p: -1 if pd.isna(p) else int(p))
    g = centered.drop_duplicates(subset=["scene_id", "kind", "_phase_key", "frame"])
    counts = g.groupby(["scene_id", "kind", "_phase_key"]).size().to_dict()
    deficient = []
    for scene in expected_scenes:
        for kind, phase in conditions:
            n = counts.get((scene, kind, phase), 0)
            if n < min_centered:
                deficient.append((scene, kind, phase, n))
    return deficient, len(expected_scenes) * len(conditions)


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
        print("by type x kind x phase:\n", df.groupby(["type", "kind", "phase"], dropna=False).size().to_string())
        print("distinct scenes:", df["scene_id"].nunique(), "/ 100 expected")
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
            print(f"lexical same-name ('different-category' policy) violations: {len(cat_violations)}")

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

        # Q13: every row must carry two resolvable, well-formed image
        # locators. reference_image/target_image are struct columns
        # (repo_id/revision/shard/member[/crop_bbox]) -- verify structurally
        # for every row, then spot-check real HF tar membership for a
        # sample (avoids a full-dataset network re-verification, which
        # would mean re-downloading every scene's tar).
        def locator_ok(loc, need_crop_bbox):
            if not isinstance(loc, dict):
                return False
            if not (loc.get("repo_id") == "IRVLUTD/RPX" and loc.get("revision") and loc.get("shard") and loc.get("member")):
                return False
            if not loc["member"].endswith(".webp"):
                return False
            cb = loc.get("crop_bbox")
            if need_crop_bbox and (cb is None or len(cb) != 4):
                return False
            return True
        bad_target_loc = df[~df["target_image"].map(lambda l: locator_ok(l, False))]
        bad_ref_loc = df[~df["reference_image"].map(lambda l: locator_ok(l, True))]
        print(f"malformed target_image locators: {len(bad_target_loc)}")
        print(f"malformed reference_image locators: {len(bad_ref_loc)}")
        sample_scenes = sorted(df["scene_id"].unique())[:3]
        for scene in sample_scenes:
            shard = df[df.scene_id == scene]["target_image"].iloc[0]["shard"]
            print(f"  spot-check locator (not fetched): {scene} -> {shard}")

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
            print(f"\nsampled 2D/3D farthest-agreement re-verification (independent, restricted to "
                  f"{len(KEEP_STAGED)} still-staged scenes, n<={SAMPLE_N}):")
            sp_results, sp_failures = sampled_spatial_geometric_reverify(df)
            print(" ", sp_results)
            for sid, reason in sp_failures[:5]:
                print("   FAIL", sid, reason)

        print("\nframe coverage (scene,kind,phase) -- min/median/max distinct frames:")
        cov = frame_coverage_table(df)
        print(f"  min={cov.min()} median={cov.median()} max={cov.max()}  cells={len(cov)}")

    print(f"\n{'='*80}\nQ17: full scene-condition centered-coverage grid\n{'='*80}")
    general_mos = load("incontext_mos_attribute_bbox")
    general_ego = load("incontext_ego_attribute_bbox")
    spatial = load("incontext_mos_spatial_bbox")
    scenes = sorted(set((general_mos["scene_id"].unique() if general_mos is not None else [])) |
                     set((general_ego["scene_id"].unique() if general_ego is not None else [])) |
                     set((spatial["scene_id"].unique() if spatial is not None else [])))
    print(f"scenes present in output: {len(scenes)} / 100 expected")

    if general_mos is not None and general_ego is not None:
        import pandas as pd
        combined_general = pd.concat([general_mos, general_ego], ignore_index=True)
        conditions = [("mos", 0), ("mos", 1), ("mos", 2), ("ego", -1)]  # -1 = ego sentinel, see full_cell_coverage
        deficient, total = full_cell_coverage(combined_general, scenes, conditions, min_centered=30)
        print(f"\nGeneral in-context cells (100 scenes x 4 conditions = {total} expected): "
              f"{len(deficient)} deficient (<30 centered frames)")
        for scene, kind, phase, n in deficient:
            phase_label = "n/a (ego)" if phase == -1 else phase
            print(f"   DEFICIENT scene={scene} kind={kind} phase={phase_label}: {n} centered frames")

    if spatial is not None:
        conditions = [("mos", 0), ("mos", 1), ("mos", 2)]
        deficient, total = full_cell_coverage(spatial, scenes, conditions, min_centered=30)
        print(f"\nSpatial in-context cells (100 scenes x 3 mos phases = {total} expected): "
              f"{len(deficient)} deficient (<30 centered frames)")
        for scene, kind, phase, n in deficient:
            phase_label = "n/a (ego)" if phase == -1 else phase
            print(f"   DEFICIENT scene={scene} kind={kind} phase={phase_label}: {n} centered frames")


if __name__ == "__main__":
    main()

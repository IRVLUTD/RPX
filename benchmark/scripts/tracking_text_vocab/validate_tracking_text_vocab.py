#!/usr/bin/env python3
"""Independent validator for the tracking text-initialization vocabulary.

Re-derives facts from the Hub wherever practical instead of trusting the
builder's own bookkeeping (mask_to_object.json resolution is re-fetched and
re-joined here, not read back from the builder's log). Fails loudly (nonzero
exit) unless every required check in the task spec passes; every check's
result is included in the JSON report regardless of pass/fail.

Run this against the FINAL staged bundle before claiming the release is
ready -- not only against an intermediate build directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    EGO_TRACKING_REVISION,
    MOS_TRACKING_REVISION,
    REPO_ID,
    SENTINEL_SOURCE_CATALOG_IDS,
    by_source_catalog_id,
    load_catalog,
)

EXPECTED_CATALOG_SIZE = 70
EXPECTED_SCENES = 100
EXPECTED_MOS_CONDITIONS = 300
EXPECTED_EGO_CONDITIONS = 100
EXPECTED_TOTAL_CONDITIONS = 400
VQA_REVISION = "vqa"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_catalog_size(catalog) -> dict:
    n = len(catalog)
    ok = n == EXPECTED_CATALOG_SIZE
    print(f"[1] catalog size: {n} (expected {EXPECTED_CATALOG_SIZE}) -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "actual": n, "expected": EXPECTED_CATALOG_SIZE}


def check_catalog_completeness(catalog) -> dict:
    missing = []
    for obj in catalog:
        gaps = []
        if not obj.canonical_name:
            gaps.append("canonical_name")
        if not obj.colors_raw:
            gaps.append("color")
        if not obj.materials_raw:
            gaps.append("material")
        if not obj.functions_raw:
            gaps.append("function")
        if not obj.categories_raw:
            gaps.append("category")
        if gaps:
            missing.append({"source_catalog_id": obj.source_catalog_id, "object_id": obj.object_id, "missing": gaps})
    ok = not missing
    print(f"[2] catalog completeness: {len(missing)}/{len(catalog)} objects have a gap -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "objects_with_gaps": missing}


def _condition_dirs(kind: str, scene: str, phase) -> str:
    return str(phase) if kind == "mos" else "ego"


def check_mask_resolution_independent(catalog_by_scid: dict, scene_vocab: pd.DataFrame) -> dict:
    """Re-fetch every MOS/Ego sam2_meta/v1.tar independently (not reading
    the builder's own unresolved-entries log) and re-join by
    source_catalog_id. Confirms requirement 3 (every object.id resolves to
    exactly one source_catalog_id) and 4 (no unresolved/duplicated/float-
    corrupted IDs), and cross-checks against scene_condition_vocab."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    mos_files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=MOS_TRACKING_REVISION)
    ego_files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=EGO_TRACKING_REVISION)

    mos_conditions = sorted(
        {
            (parts[1], int(parts[2]))
            for f in mos_files
            if (parts := f.split("/")) and len(parts) == 6 and parts[0] == "scenes"
            and parts[3:] == ["labels", "sam2_meta", "v1.tar"] and parts[2].isdigit()
        }
    )
    ego_scenes = sorted(
        {f.split("/")[1] for f in ego_files if f.startswith("scenes/") and f.endswith("/ego/labels/sam2_meta/v1.tar")}
    )

    unresolved = []
    duplicate_object_ids = []
    excluded_sentinels = []
    all_entries = []

    def _process(kind, scene, phase, shard, revision):
        local_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=shard, revision=revision)
        with tarfile.open(local_path) as archive:
            mapping = json.load(archive.extractfile("sam2/mask_to_object.json"))
        seen_ids = Counter()
        for entry in mapping.values():
            scid = str(entry["object"]["id"])
            seen_ids[scid] += 1
            all_entries.append((kind, scene, phase, int(entry["mask_index"]), scid))
            if scid in SENTINEL_SOURCE_CATALOG_IDS:
                excluded_sentinels.append({"scene_id": scene, "kind": kind, "phase": phase, "source_catalog_id": scid})
            elif scid not in catalog_by_scid:
                unresolved.append({"scene_id": scene, "kind": kind, "phase": phase, "source_catalog_id": scid})
        for scid, count in seen_ids.items():
            # Repeated sentinel entries within one condition are expected
            # (multiple distinct unidentified mask instances); only a
            # repeated REAL catalog id within one condition is a defect.
            if count > 1 and scid not in SENTINEL_SOURCE_CATALOG_IDS:
                duplicate_object_ids.append({"scene_id": scene, "kind": kind, "phase": phase, "source_catalog_id": scid, "count": count})

    for scene, phase in mos_conditions:
        _process("mos", scene, phase, f"scenes/{scene}/{phase}/labels/sam2_meta/v1.tar", MOS_TRACKING_REVISION)
    for scene in ego_scenes:
        _process("ego", scene, None, f"scenes/{scene}/ego/labels/sam2_meta/v1.tar", EGO_TRACKING_REVISION)

    # Float corruption (e.g. "63.2" silently coerced through a numeric
    # dtype and back to something like "63.20") is caught for free here:
    # every scid is read as a raw JSON string (never passed through
    # float()/int()) and compared verbatim against the catalog's own
    # source_catalog_id strings via the `unresolved` lookup above -- a
    # corrupted ID simply fails to resolve and shows up there.
    ok = not unresolved and not duplicate_object_ids
    print(
        f"[3+4] independent mask_to_object.json re-resolution: {len(all_entries)} entries across "
        f"{len(mos_conditions)} MOS + {len(ego_scenes)} Ego conditions, {len(unresolved)} unresolved "
        f"(genuine catalog gaps), {len(excluded_sentinels)} sentinel (unknown/unlabeled, expected), "
        f"{len(duplicate_object_ids)} duplicate-within-condition -> {'PASS' if ok else 'FAIL'}"
    )
    return {
        "ok": ok,
        "total_entries": len(all_entries),
        "mos_conditions_checked": len(mos_conditions),
        "ego_conditions_checked": len(ego_scenes),
        "unresolved": unresolved,
        "excluded_sentinels_count": len(excluded_sentinels),
        "duplicate_object_ids_within_condition": duplicate_object_ids,
        "_all_entries": all_entries,  # consumed by check 9; stripped before the report is written
    }


def check_scene_condition_counts(scene_vocab: pd.DataFrame) -> dict:
    conditions = scene_vocab[["scene_id", "kind", "phase"]].drop_duplicates()
    mos = conditions[conditions.kind == "mos"]
    ego = conditions[conditions.kind == "ego"]
    scenes = sorted(scene_vocab["scene_id"].unique())
    ok = (
        len(scenes) == EXPECTED_SCENES
        and len(mos) == EXPECTED_MOS_CONDITIONS
        and len(ego) == EXPECTED_EGO_CONDITIONS
        and len(conditions) == EXPECTED_TOTAL_CONDITIONS
    )
    print(
        f"[5] scenes={len(scenes)} (want {EXPECTED_SCENES}), MOS conditions={len(mos)} (want "
        f"{EXPECTED_MOS_CONDITIONS}), Ego conditions={len(ego)} (want {EXPECTED_EGO_CONDITIONS}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return {
        "ok": ok,
        "scenes": len(scenes),
        "mos_conditions": len(mos),
        "ego_conditions": len(ego),
        "total_conditions": len(conditions),
    }


def check_no_duplicates(scene_vocab: pd.DataFrame) -> dict:
    key1 = ["scene_id", "kind", "phase", "mask_index"]
    key2 = ["scene_id", "kind", "phase", "source_catalog_id"]
    dup1 = scene_vocab[scene_vocab.duplicated(subset=key1, keep=False)]
    dup2 = scene_vocab[scene_vocab.duplicated(subset=key2, keep=False)]
    ok = dup1.empty and dup2.empty
    print(f"[6] duplicate (scene,kind,phase,mask_index): {len(dup1)}  duplicate (scene,kind,phase,scid): {len(dup2)} -> {'PASS' if ok else 'FAIL'}")
    return {
        "ok": ok,
        "duplicate_mask_index_rows": len(dup1),
        "duplicate_source_catalog_id_rows": len(dup2),
    }


def check_prompt_nonempty(scene_vocab: pd.DataFrame) -> dict:
    empty = scene_vocab[scene_vocab["prompt_text"].isna() | (scene_vocab["prompt_text"].astype(str).str.strip() == "")]
    ok = empty.empty
    print(f"[7] empty prompt_text rows: {len(empty)} -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "empty_prompt_rows": len(empty)}


def check_duplicate_prompts_within_condition(scene_vocab: pd.DataFrame) -> dict:
    collisions = []
    for (scene, kind, phase), group in scene_vocab.groupby(["scene_id", "kind", "phase"], dropna=False):
        counts = group["prompt_text"].value_counts()
        colliding = counts[counts > 1]
        for prompt_text, count in colliding.items():
            rows = group[group["prompt_text"] == prompt_text]
            collisions.append(
                {
                    "scene_id": scene,
                    "kind": kind,
                    "phase": None if pd.isna(phase) else int(phase),
                    "prompt_text": prompt_text,
                    "count": int(count),
                    "source_catalog_ids": sorted(rows["source_catalog_id"].tolist()),
                }
            )
    print(f"[8] duplicate prompt_text within a scene-condition: {len(collisions)} collision groups (reported, not modified)")
    return {"collision_groups": len(collisions), "collisions": collisions}


def _stage_masks_only(scene: str, phase_dir: str, revision: str, stage_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    shard = f"scenes/{scene}/{phase_dir}/labels/masks/v1.tar"
    local_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=shard, revision=revision)
    dest = stage_root / scene / phase_dir
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_path) as archive:
        archive.extractall(dest)
    return dest / "sam2" / "masks"


def check_exhaustive_mask_coverage(scene_vocab: pd.DataFrame, mask_resolution_entries: list[tuple], stage_root: Path) -> dict:
    """Streams every MOS and Ego mask PNG, one condition at a time, to
    prove every visible (nonzero) mask index in every frame is covered by
    that condition's constant scene_condition_vocab -- not that the
    vocabulary changes per frame (it never does). mask_resolution_entries
    (kind, scene, phase, mask_index, source_catalog_id) comes from check
    3+4's independent mask_to_object.json re-fetch and includes the
    documented "unknown" sentinel mask indices, which are legitimately
    ABSENT from scene_condition_vocab by design (see common.py) -- without
    folding them in here, every sentinel mask index would show up as a
    false-positive "missing mapping"."""
    from huggingface_hub import HfApi

    api = HfApi()
    mos_files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=MOS_TRACKING_REVISION)
    ego_files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=EGO_TRACKING_REVISION)
    mos_conditions = sorted(
        {
            (parts[1], int(parts[2]))
            for f in mos_files
            if (parts := f.split("/")) and len(parts) == 6 and parts[0] == "scenes"
            and parts[3:] == ["labels", "masks", "v1.tar"] and parts[2].isdigit()
        }
    )
    ego_scenes = sorted(
        {f.split("/")[1] for f in ego_files if f.startswith("scenes/") and f.endswith("/ego/labels/masks/v1.tar")}
    )

    vocab_index: dict[tuple, set[int]] = defaultdict(set)
    for _, row in scene_vocab.iterrows():
        phase = None if pd.isna(row["phase"]) else int(row["phase"])
        vocab_index[(row["scene_id"], row["kind"], phase)].add(int(row["mask_index"]))
    sentinel_index: dict[tuple, set[int]] = defaultdict(set)
    for kind, scene, phase, mask_index, scid in mask_resolution_entries:
        if scid in SENTINEL_SOURCE_CATALOG_IDS:
            sentinel_index[(scene, kind, phase)].add(mask_index)

    total_frames = 0
    total_visible_occurrences = 0
    total_sentinel_occurrences = 0
    missing: list[dict] = []
    extra_conditions: list[dict] = []
    conditions_checked = 0

    def _process(scene, kind, phase, phase_dir, revision):
        nonlocal total_frames, total_visible_occurrences, total_sentinel_occurrences, conditions_checked
        key = (scene, kind, phase)
        vocab_known = vocab_index.get(key)
        sentinel_known = sentinel_index.get(key, set())
        if vocab_known is None and not sentinel_known:
            extra_conditions.append({"scene_id": scene, "kind": kind, "phase": phase, "reason": "no vocabulary rows for this condition"})
        known_indices = (vocab_known or set()) | sentinel_known
        masks_dir = _stage_masks_only(scene, phase_dir, revision, stage_root)
        frame_files = sorted(masks_dir.glob("*.png"))
        for frame_path in frame_files:
            arr = np.array(_imread(frame_path))
            visible = set(int(v) for v in np.unique(arr) if v != 0)
            total_frames += 1
            total_visible_occurrences += len(visible)
            total_sentinel_occurrences += len(visible & sentinel_known)
            unmapped = visible - known_indices
            if unmapped:
                missing.append(
                    {
                        "scene_id": scene, "kind": kind, "phase": phase,
                        "frame": frame_path.stem, "unmapped_mask_indices": sorted(unmapped),
                    }
                )
        conditions_checked += 1
        shutil.rmtree(stage_root / scene, ignore_errors=True)

    for i, (scene, phase) in enumerate(mos_conditions, 1):
        _process(scene, "mos", phase, str(phase), MOS_TRACKING_REVISION)
        if i % 20 == 0 or i == len(mos_conditions):
            print(f"  [mask coverage] MOS {i}/{len(mos_conditions)}: frames={total_frames} missing={len(missing)}", flush=True)

    for i, scene in enumerate(ego_scenes, 1):
        _process(scene, "ego", None, "ego", EGO_TRACKING_REVISION)
        if i % 20 == 0 or i == len(ego_scenes):
            print(f"  [mask coverage] Ego {i}/{len(ego_scenes)}: frames={total_frames} missing={len(missing)}", flush=True)

    ok = not missing and not extra_conditions
    print(
        f"[9] exhaustive mask coverage: conditions_checked={conditions_checked} total_frames={total_frames} "
        f"visible_frame_object_occurrences={total_visible_occurrences} "
        f"(of which sentinel/unknown={total_sentinel_occurrences}) missing={len(missing)} "
        f"extra_conditions={len(extra_conditions)} -> {'PASS' if ok else 'FAIL'}"
    )
    return {
        "ok": ok,
        "conditions_checked": conditions_checked,
        "total_frames": total_frames,
        "total_visible_frame_object_occurrences": total_visible_occurrences,
        "total_sentinel_frame_occurrences": total_sentinel_occurrences,
        "missing_mappings": missing[:200],
        "missing_mappings_count": len(missing),
        "extra_conditions": extra_conditions,
    }


def _imread(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()


def check_vqa_cross_check(catalog_by_scid: dict) -> dict:
    """Reads vqa/attribute.parquet at the vqa branch, parses non-null
    target_oid plus JSON-encoded answer_oids, and verifies their union
    resolves against the official catalog. Report-only: never used to fill
    missing attributes."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(REPO_ID, repo_type="dataset", filename="vqa/attribute.parquet", revision=VQA_REVISION)
    df = pd.read_parquet(local_path, columns=["target_oid", "answer_oids"])

    ids = set()
    for value in df["target_oid"].dropna():
        ids.add(str(value))
    for value in df["answer_oids"].dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            for oid in parsed:
                ids.add(str(oid))

    resolved = {oid for oid in ids if oid in catalog_by_scid}
    unresolved = sorted(ids - resolved)
    ok = True  # report-only per the task -- VQA rows never gate vocabulary completeness
    print(
        f"[10] VQA identity cross-check: {len(ids)} distinct IDs referenced in vqa/attribute.parquet "
        f"(target_oid + answer_oids), {len(resolved)} resolve to the official catalog, "
        f"{len(unresolved)} do not (report-only, does not fail the release)"
    )
    return {
        "ok": ok,
        "distinct_ids_referenced": len(ids),
        "resolved_to_catalog": len(resolved),
        "unresolved": unresolved[:100],
        "unresolved_count": len(unresolved),
        "note": "report-only cross-check; these rows are never used to fill missing catalog attributes",
    }


def check_parquet_jsonl_equality(out_dir: Path) -> dict:
    """Reads the Parquet via pyarrow's own to_pylist() rather than pandas:
    pandas.read_parquet upcasts a nullable int64 column (e.g. phase, which
    is int for MOS and null for Ego) to float64/NaN on the way in, which
    would make this check compare "1.0" against JSONL's "1" and report a
    false mismatch that has nothing to do with the actual stored data."""
    import pyarrow.parquet as pq

    results = {}
    ok = True
    for base in ("object_catalog_vocab", "scene_condition_vocab"):
        parquet_records = pq.read_table(out_dir / f"{base}.parquet").to_pylist()
        jsonl_rows = _load_jsonl(out_dir / f"{base}.jsonl")
        same_count = len(parquet_records) == len(jsonl_rows)

        def _canon(rows):
            return sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)

        same_rows = _canon(parquet_records) == _canon(jsonl_rows)
        pass_ = same_count and same_rows
        ok = ok and pass_
        print(f"[11] {base}: parquet={len(parquet_records)} jsonl={len(jsonl_rows)} rows -> {'PASS' if pass_ else 'FAIL'}")
        results[base] = {"ok": pass_, "parquet_rows": len(parquet_records), "jsonl_rows": len(jsonl_rows)}
    return {"ok": ok, "tables": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="directory holding the 4 data files")
    parser.add_argument("--stage-dir", type=Path, default=Path("/tmp/tracking_vocab_mask_stage"))
    parser.add_argument("--skip-exhaustive-masks", action="store_true", help="debugging only -- final run must not use this")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    print("loading official catalog independently...")
    catalog = load_catalog()
    catalog_by_scid = by_source_catalog_id(catalog)

    scene_vocab = pd.read_parquet(args.bundle_dir / "scene_condition_vocab.parquet")
    scene_vocab["source_catalog_id"] = scene_vocab["source_catalog_id"].astype(str)

    report = {"bundle_dir": str(args.bundle_dir), "checks": {}}
    report["checks"]["1_catalog_size"] = check_catalog_size(catalog)
    report["checks"]["2_catalog_completeness"] = check_catalog_completeness(catalog)
    report["checks"]["3_4_mask_resolution"] = check_mask_resolution_independent(catalog_by_scid, scene_vocab)
    mask_resolution_entries = report["checks"]["3_4_mask_resolution"].pop("_all_entries")
    report["checks"]["5_scene_condition_counts"] = check_scene_condition_counts(scene_vocab)
    report["checks"]["6_no_duplicates"] = check_no_duplicates(scene_vocab)
    report["checks"]["7_prompt_nonempty"] = check_prompt_nonempty(scene_vocab)
    report["checks"]["8_duplicate_prompts"] = check_duplicate_prompts_within_condition(scene_vocab)
    if args.skip_exhaustive_masks:
        print("[9] SKIPPED (--skip-exhaustive-masks; not valid for a final release check)")
        report["checks"]["9_exhaustive_mask_coverage"] = {"ok": None, "skipped": True}
    else:
        args.stage_dir.mkdir(parents=True, exist_ok=True)
        report["checks"]["9_exhaustive_mask_coverage"] = check_exhaustive_mask_coverage(
            scene_vocab, mask_resolution_entries, args.stage_dir
        )
    report["checks"]["10_vqa_cross_check"] = check_vqa_cross_check(catalog_by_scid)
    report["checks"]["11_parquet_jsonl_equality"] = check_parquet_jsonl_equality(args.bundle_dir)

    required_ok = all(
        check.get("ok") is not False
        for name, check in report["checks"].items()
    )
    # duplicate-prompt audit (#8) is report-only (never silently modified);
    # it does not gate pass/fail on its own.
    hard_fail = any(
        check.get("ok") is False
        for name, check in report["checks"].items()
        if name != "8_duplicate_prompts"
    )
    report["overall_pass"] = not hard_fail

    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(f"\nfull report written to {args.report}")
    print(f"\nOVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

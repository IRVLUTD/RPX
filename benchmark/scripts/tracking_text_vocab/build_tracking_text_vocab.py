#!/usr/bin/env python3
"""Build the constant per-scene-condition text-initialization vocabulary
(primary color + canonical object name) for MOS and Ego tracking.

Two tables are produced:

  object_catalog_vocab  -- one row per officially published catalog object
                            (70), with its full authored attribute lists and
                            a deterministic primary_color/prompt_text.
  scene_condition_vocab -- one row per (scene, kind, phase) x object that is
                            ever tracked in that clip, joining each scene's
                            sam2/mask_to_object.json entries to the catalog
                            above by source_catalog_id.

Does NOT stream mask PNGs or derive anything from vqa/attribute.parquet --
see validate_tracking_text_vocab.py for the exhaustive per-frame mask
coverage audit and the VQA identity cross-check, which are read-only
verification passes over this builder's output, run separately.

Sentinel entries: some Ego mask_to_object.json entries carry the literal
{"id": "unknown", "name": "unlabeled"} placeholder for a tracked instance
that was never assigned a real object identity (confirmed empirically:
60/2,799 raw entries across all 400 conditions, 100% kind=ego, 100% exactly
this id/name pair -- never seen in MOS, never any other unresolved id).
These have no catalog identity to build a "primary color + canonical name"
prompt from, so they are excluded from scene_condition_vocab by design, not
silently dropped -- every excluded entry is written to
excluded_sentinel_entries.json and counted in manifest_summary.json. A
mask_to_object.json object.id that fails to resolve for any OTHER reason is
a genuine catalog gap and is written to unresolved_mask_entries.json as a
loud warning instead.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from common import (
    EGO_TRACKING_REVISION,
    MOS_TRACKING_REVISION,
    REPO_ID,
    SCHEMA_VERSION,
    SENTINEL_SOURCE_CATALOG_IDS,
    by_source_catalog_id,
    load_catalog,
    sort_key_scene_condition,
)


def list_mos_conditions(revision: str, repo_id: str = REPO_ID) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    conditions = []
    for filename in files:
        parts = filename.split("/")
        # scenes/<scene>/<phase>/labels/sam2_meta/v1.tar
        if len(parts) == 6 and parts[0] == "scenes" and parts[3:] == ["labels", "sam2_meta", "v1.tar"]:
            scene, phase_str = parts[1], parts[2]
            if phase_str.isdigit():
                conditions.append((scene, int(phase_str)))
    return sorted(set(conditions))


def list_ego_conditions(revision: str, repo_id: str = REPO_ID) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    scenes = set()
    for filename in files:
        if filename.startswith("scenes/") and filename.endswith("/ego/labels/sam2_meta/v1.tar"):
            scenes.add(filename.split("/")[1])
    return sorted(scenes)


def fetch_mask_to_object(scene: str, phase_dir: str, revision: str, repo_id: str = REPO_ID) -> dict:
    """Download only sam2_meta/v1.tar (small, ~10KB), extract
    sam2/mask_to_object.json, delete the local cache copy, return the
    parsed mapping plus the exact locator used (for provenance)."""
    from huggingface_hub import hf_hub_download

    shard = f"scenes/{scene}/{phase_dir}/labels/sam2_meta/v1.tar"
    local_path = hf_hub_download(repo_id, repo_type="dataset", filename=shard, revision=revision)
    with tarfile.open(local_path) as archive:
        member = archive.extractfile("sam2/mask_to_object.json")
        mapping = json.load(member)
    # hf_hub_download caches under the shared HF cache; do not delete the
    # cache itself (other processes may share it), but do not retain any
    # extra copy of our own.
    return {"shard": shard, "member": "sam2/mask_to_object.json", "mapping": mapping}


def build_scene_condition_rows(
    scene: str,
    kind: str,
    phase: int | None,
    phase_dir: str,
    revision: str,
    catalog_by_scid: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (rows, unresolved, excluded_sentinels). unresolved holds any
    mask entry whose object.id is neither a catalog member nor a known
    sentinel -- a genuine catalog gap the caller aggregates and the
    validator fails loudly on. excluded_sentinels holds the documented
    {"id": "unknown", "name": "unlabeled"} placeholder (see module
    docstring) -- reported, never a warning, never in scene_condition_vocab."""
    fetched = fetch_mask_to_object(scene, phase_dir, revision)
    mapping = fetched["mapping"]
    rows: list[dict] = []
    unresolved: list[dict] = []
    excluded_sentinels: list[dict] = []
    for _key, entry in mapping.items():
        mask_index = int(entry["mask_index"])
        object_ref = entry["object"]
        source_catalog_id = str(object_ref["id"])
        scene_mapping_name = object_ref["name"]
        catalog_obj = catalog_by_scid.get(source_catalog_id)
        if catalog_obj is None:
            record = {
                "scene_id": scene,
                "kind": kind,
                "phase": phase,
                "mask_index": mask_index,
                "source_catalog_id": source_catalog_id,
                "scene_mapping_name": scene_mapping_name,
            }
            if source_catalog_id in SENTINEL_SOURCE_CATALOG_IDS:
                excluded_sentinels.append(record)
            else:
                unresolved.append(record)
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scene_id": scene,
                "kind": kind,
                "phase": phase,
                "mask_index": mask_index,
                "source_catalog_id": source_catalog_id,
                "global_object_id": catalog_obj.global_object_id,
                "object_id": catalog_obj.object_id,
                "scene_mapping_name": scene_mapping_name,
                "canonical_name": catalog_obj.canonical_name,
                "display_name": catalog_obj.display_name,
                "primary_color": catalog_obj.primary_color,
                "colors_normalized": list(catalog_obj.colors_normalized),
                "prompt_text": catalog_obj.prompt_text,
                "source_revision": revision,
                "sam2_meta_shard": fetched["shard"],
                "sam2_meta_member": fetched["member"],
            }
        )
    return rows, unresolved, excluded_sentinels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit-scenes", type=int, help="debugging: cap the number of scenes processed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading official catalog...", flush=True)
    catalog = load_catalog()
    catalog_by_scid = by_source_catalog_id(catalog)
    print(f"catalog: {len(catalog)} objects", flush=True)

    mos_conditions = list_mos_conditions(MOS_TRACKING_REVISION)
    ego_scenes = list_ego_conditions(EGO_TRACKING_REVISION)
    if args.limit_scenes:
        mos_scenes_allowed = {c[0] for c in mos_conditions[: args.limit_scenes * 3]}
        mos_conditions = [c for c in mos_conditions if c[0] in mos_scenes_allowed][: args.limit_scenes * 3]
        ego_scenes = ego_scenes[: args.limit_scenes]
    print(f"MOS scene-phase conditions: {len(mos_conditions)}", flush=True)
    print(f"Ego scene conditions: {len(ego_scenes)}", flush=True)

    all_rows: list[dict] = []
    all_unresolved: list[dict] = []
    all_excluded_sentinels: list[dict] = []

    for i, (scene, phase) in enumerate(mos_conditions, 1):
        rows, unresolved, excluded = build_scene_condition_rows(
            scene, "mos", phase, str(phase), MOS_TRACKING_REVISION, catalog_by_scid
        )
        all_rows.extend(rows)
        all_unresolved.extend(unresolved)
        all_excluded_sentinels.extend(excluded)
        if i % 50 == 0 or i == len(mos_conditions):
            print(f"  MOS [{i}/{len(mos_conditions)}] {scene}/{phase}: {len(rows)} objects", flush=True)

    for i, scene in enumerate(ego_scenes, 1):
        rows, unresolved, excluded = build_scene_condition_rows(
            scene, "ego", None, "ego", EGO_TRACKING_REVISION, catalog_by_scid
        )
        all_rows.extend(rows)
        all_unresolved.extend(unresolved)
        all_excluded_sentinels.extend(excluded)
        if i % 25 == 0 or i == len(ego_scenes):
            print(f"  Ego [{i}/{len(ego_scenes)}] {scene}: {len(rows)} objects", flush=True)

    if all_excluded_sentinels:
        print(
            f"{len(all_excluded_sentinels)} sentinel (unknown/unlabeled) mask entries excluded "
            f"by design -- see excluded_sentinel_entries.json",
            flush=True,
        )
        sentinel_path = args.out_dir / "excluded_sentinel_entries.json"
        sentinel_path.write_text(json.dumps(all_excluded_sentinels, indent=2, sort_keys=True), encoding="utf-8")

    if all_unresolved:
        print(f"WARNING: {len(all_unresolved)} mask entries did not resolve to the official catalog", flush=True)
        unresolved_path = args.out_dir / "unresolved_mask_entries.json"
        unresolved_path.write_text(json.dumps(all_unresolved, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  written to {unresolved_path} (build continues; the validator fails loudly on this)", flush=True)

    all_rows.sort(key=sort_key_scene_condition)

    import pyarrow as pa
    import pyarrow.parquet as pq

    catalog_rows = [obj.to_row() for obj in sorted(catalog, key=lambda o: o.source_catalog_id)]
    catalog_table = pa.Table.from_pylist(catalog_rows)
    pq.write_table(catalog_table, args.out_dir / "object_catalog_vocab.parquet", compression="zstd")
    with (args.out_dir / "object_catalog_vocab.jsonl").open("w", encoding="utf-8") as handle:
        for row in catalog_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    scene_table = pa.Table.from_pylist(all_rows)
    # pa.Table.from_pylist infers a nullable int/None column as double (MOS
    # phase 0/1/2 mixed with Ego's None), which would silently round-trip
    # phase as 1.0 instead of 1 -- the exact class of float-corruption bug
    # this project has repeatedly had to catch and fix elsewhere. Force an
    # explicit nullable int64 column instead.
    phase_field_index = scene_table.schema.get_field_index("phase")
    phase_column = pa.array([None if row["phase"] is None else int(row["phase"]) for row in all_rows], type=pa.int64())
    scene_table = scene_table.set_column(phase_field_index, "phase", phase_column)
    pq.write_table(scene_table, args.out_dir / "scene_condition_vocab.parquet", compression="zstd")
    with (args.out_dir / "scene_condition_vocab.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    print(f"\ncatalog rows: {len(catalog_rows)}")
    print(f"scene-condition rows: {len(all_rows)}")
    print(f"MOS conditions: {len(mos_conditions)}  Ego conditions: {len(ego_scenes)}")
    print(f"unresolved mask entries (genuine catalog gap): {len(all_unresolved)}")
    print(f"excluded sentinel entries (unknown/unlabeled, by design): {len(all_excluded_sentinels)}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()

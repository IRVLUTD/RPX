"""Phase-E dry run: generate in-context GT for exactly scene001 (already
staged, no download) across all 3 mos phases + ego, write to a separate
dry-run output (never touches the real incremental run_incontext_gt.py
outputs), and print a validation summary."""
import json
import random
import sys
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt")
sys.path.insert(0, "/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot")

from generate_gt import gen_incontext_for_mos_phase, gen_incontext_for_ego_phase  # noqa: E402
from generate_spatial_gt import load_mapping  # noqa: E402
from lib import load_fewsol_lookup  # noqa: E402
import sos_catalog as sc  # noqa: E402
import sos_reference as sr  # noqa: E402

STAGE_ROOT = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged")
OUT_DIR = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/incontext_dryrun")
SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
REF_MANIFEST = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops/reference_crops_v1.parquet")
SCENE = "scene001"


def main():
    lookup = load_fewsol_lookup(SOS_ROOT)
    catalog = sc.load_catalog()
    catalog_by_scid = sc.by_source_catalog_id(catalog)
    ref_crops = {oid: sr.ReferenceCrop(**r) for oid, r in sr.load_reference_manifest(REF_MANIFEST).items()}
    rng = random.Random(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_items, all_drops = [], []
    for phase in (0, 1, 2):
        root = STAGE_ROOT / SCENE / str(phase)
        mapping = load_mapping(str(root / "sam2/mask_to_object.json"))
        items, drops, n = gen_incontext_for_mos_phase(
            SCENE, phase, str(root), mapping, 1000, lookup, catalog, catalog_by_scid,
            ref_crops, rng, 42, "main", require_diff_category=True, max_per_type=None)
        print(f"mos phase {phase}: {n} frames -> {len(items)} items, {len(drops)} drops", flush=True)
        all_items += items
        all_drops += drops

    ego_root = STAGE_ROOT / SCENE / "ego"
    mapping = load_mapping(str(ego_root / "sam2/mask_to_object.json"))
    items, drops, n = gen_incontext_for_ego_phase(
        SCENE, str(ego_root), mapping, 1000, lookup, catalog, catalog_by_scid,
        ref_crops, rng, 42, "main", require_diff_category=True, max_per_type=None)
    print(f"ego: {n} frames -> {len(items)} items, {len(drops)} drops", flush=True)
    all_items += items
    all_drops += drops

    with open(OUT_DIR / "items.jsonl", "w") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    with open(OUT_DIR / "drops.jsonl", "w") as f:
        for d in all_drops:
            f.write(json.dumps({"type": d.type, "reason": d.reason, "detail": d.detail}) + "\n")

    from collections import Counter
    print(f"\nTOTAL: {len(all_items)} items, {len(all_drops)} drops")
    print("by type:", dict(Counter(it["type"] for it in all_items)))
    print("by kind:", dict(Counter(it["kind"] for it in all_items)))
    print("drop reasons:", dict(Counter((d.type, d.reason) for d in all_drops)))
    print("centered:", sum(1 for it in all_items if it["answer_is_centered"]),
          "/ non-centered:", sum(1 for it in all_items if not it["answer_is_centered"]))
    print("distinct sample_ids:", len(set(it["sample_id"] for it in all_items)), "== total?",
          len(set(it["sample_id"] for it in all_items)) == len(all_items))


if __name__ == "__main__":
    main()

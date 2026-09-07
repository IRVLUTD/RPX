"""Generate ground truth for the three VQA tasks (attribute VQA, left-right/
up-down yes-no, spatial bbox) across any set of already-staged scenes.

This does not stage/download scene data itself -- point --staged-root at a
directory that already has the standard layout:

    <staged-root>/<scene>/<0|1|2>/sam2/{masks/,mask_to_object.json}
    <staged-root>/<scene>/<0|1|2>/depth/            (mos only)
    <staged-root>/<scene>/ego/sam2/{masks/,mask_to_object.json}

Scene selection: pass --scenes explicitly, or --tier {easy,medium,hard,all}
to pull the real scene list for that tier from the dataset's
splits/scene_splits.json on Hugging Face. Any named scene not present under
--staged-root is skipped with a warning, not an error, so this can run
incrementally as more scenes get staged.

Example -- every hard-tier scene, 30 frames per phase:

    python generate_gt.py \\
        --staged-root /path/to/staged \\
        --sos-root /path/to/single_objects/sos_wrapped \\
        --tier hard \\
        --frames-per-phase 30 \\
        --out gt_hard.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from lib import load_fewsol_lookup  # noqa: E402
from gt_attributes import collect_candidates, build_folder_to_fewsol  # noqa: E402
from gt_spatial import gen_spatial_questions  # noqa: E402
from generate_spatial_gt import compute_instances, imread_depth, imread_mask, load_mapping, select_frames  # noqa: E402
import dedup  # noqa: E402

# attr_synonym_yes/no are capped at 1 each regardless of max_per_type (see
# gt_attributes.py's CANDIDATE_BUCKETS comment) -- a balanced yes/no split
# per frame matters more here than volume. attr_count and attr_odd_one_out
# are left uncapped: their real pools are already small and dedup.distribute
# spreads them fairly on its own, no artificial cap needed.
UNCAPPED_ATTR_TYPES = {"attr_count", "attr_odd_one_out"}
SYNONYM_CAP = 1


def _attr_type_caps(max_per_type):
    caps = {t: max_per_type for t in
            ("attr_single_color", "attr_single_material", "attr_single_function", "attr_composition")}
    caps.update({t: None for t in UNCAPPED_ATTR_TYPES})
    caps["attr_synonym_yes"] = SYNONYM_CAP
    caps["attr_synonym_no"] = SYNONYM_CAP
    return caps


def get_scene_list(tier: str) -> list[str]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("IRVLUTD/RPX", repo_type="dataset",
                            filename="splits/scene_splits.json", revision="main")
    splits = json.loads(Path(path).read_text())["splits"]
    if tier == "all":
        return sorted(s for tier_scenes in splits.values() for s in tier_scenes)
    return sorted(splits[tier])


def _gen_attributes_for_phase(scene, kind, phase, mask_dir, selected, mapping,
                               lookup, sos_catalog_names, folder_to_fewsol, rng, max_per_type):
    """Two-pass, phase-wide: collect every frame's full candidate pool
    first, then distribute fairly across the frames each fact is valid in
    (dedup.py) -- object-intrinsic facts don't change with camera viewpoint
    within a phase (confirmed), so independent per-frame sampling wastes
    most of its budget on repeats. See gt_attributes.py's module docstring."""
    frame_candidates = {}
    for fid in selected:
        mask_path = os.path.join(mask_dir, f"{fid}.png")
        frame_candidates[fid] = collect_candidates(
            scene, kind, phase, fid, mask_path, mapping, lookup,
            sos_catalog_names, folder_to_fewsol, rng)

    assigned = dedup.distribute(frame_candidates, _attr_type_caps(max_per_type), rng)
    items = []
    for fid in selected:
        items += assigned[fid]
    return items


def gen_for_mos_phase(scene, phase, root, mapping, frames_per_phase, lookup,
                       sos_catalog_names, folder_to_fewsol, rng, max_per_type):
    mask_dir = os.path.join(root, "sam2", "masks")
    fids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(mask_dir, "*.png")))
    selected = select_frames(fids, frames_per_phase)

    items = _gen_attributes_for_phase(scene, "mos", phase, mask_dir, selected, mapping,
                                       lookup, sos_catalog_names, folder_to_fewsol, rng, max_per_type)

    # Spatial tasks are deliberately NOT deduped across frames: left/right,
    # closest, and farthest genuinely change with camera viewpoint within a
    # phase (confirmed: 90% of object pairs flipped spatial_lr_binary's
    # answer at least once across one phase) -- these are image-relative
    # questions, not object-intrinsic facts, so repeating them across frames
    # is a legitimate viewpoint-robustness test, not redundancy.
    for fid in selected:
        mask_path = os.path.join(mask_dir, f"{fid}.png")
        depth_path = os.path.join(root, "depth", f"{fid}.png")
        if os.path.exists(depth_path):
            mask = imread_mask(mask_path)
            depth = imread_depth(depth_path)
            instances = compute_instances(mask, mapping, depth)
            H, W = mask.shape
            items += gen_spatial_questions(scene, phase, fid, mask_path, instances, W, H, rng, max_per_type)

    return items, len(selected)


def gen_for_ego_phase(scene, root, mapping, frames_per_phase, lookup, sos_catalog_names,
                       folder_to_fewsol, rng, max_per_type):
    """Attribute VQA only -- ego has no depth, and the two spatial tasks
    both depend on it (see gt_spatial.py)."""
    mask_dir = os.path.join(root, "sam2", "masks")
    fids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(mask_dir, "*.png")))
    selected = select_frames(fids, frames_per_phase)

    items = _gen_attributes_for_phase(scene, "ego", None, mask_dir, selected, mapping,
                                       lookup, sos_catalog_names, folder_to_fewsol, rng, max_per_type)
    return items, len(selected)


# ---------------------------------------------------------------------------
# In-context (two-image) tasks -- additive only, does not touch anything
# above this line. Lazy-imports gt_incontext/sos_catalog (which reach HF for
# the 70-object catalog) so plain single-image CLI usage above never needs
# network access. catalog/catalog_by_scid/ref_crops are loaded ONCE by the
# caller (see pilot/run_incontext_gt.py) and threaded through -- reloading
# the 70-object catalog or re-scanning the reference-crop manifest per scene
# would be pure waste, they're invariant across the whole run.

def gen_incontext_for_mos_phase(scene, phase, root, mapping, frames_per_phase, lookup,
                                 catalog, catalog_by_scid, ref_crops, rng, seed, revision,
                                 require_diff_category=True, max_per_type=None):
    from gt_incontext import gen_incontext_general_for_phase, gen_incontext_spatial_for_frame
    from generate_spatial_gt import imread_mask as _imread_mask, imread_depth as _imread_depth

    mask_dir = os.path.join(root, "sam2", "masks")
    fids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(mask_dir, "*.png")))
    selected = select_frames(fids, frames_per_phase)

    items, drops = gen_incontext_general_for_phase(
        scene, "mos", phase, selected, mask_dir, mapping, lookup, catalog, catalog_by_scid,
        ref_crops, seed, revision, rng, max_per_type=max_per_type, require_diff_category=require_diff_category)

    for fid in selected:
        depth_path = os.path.join(root, "depth", f"{fid}.png")
        if not os.path.exists(depth_path):
            continue
        mask = _imread_mask(os.path.join(mask_dir, f"{fid}.png"))
        depth = _imread_depth(depth_path)
        sp_items, sp_drops = gen_incontext_spatial_for_frame(
            scene, phase, fid, mask, depth, mapping, catalog_by_scid, ref_crops, revision)
        items += sp_items
        drops += sp_drops

    return items, drops, len(selected)


def gen_incontext_for_ego_phase(scene, root, mapping, frames_per_phase, lookup,
                                 catalog, catalog_by_scid, ref_crops, rng, seed, revision,
                                 require_diff_category=True, max_per_type=None):
    """General-family only -- ego has no depth, and inctx_spatial_farthest
    depends on it exactly like the existing spatial_farthest does."""
    from gt_incontext import gen_incontext_general_for_phase

    mask_dir = os.path.join(root, "sam2", "masks")
    fids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(mask_dir, "*.png")))
    selected = select_frames(fids, frames_per_phase)

    items, drops = gen_incontext_general_for_phase(
        scene, "ego", None, selected, mask_dir, mapping, lookup, catalog, catalog_by_scid,
        ref_crops, seed, revision, rng, max_per_type=max_per_type, require_diff_category=require_diff_category)
    return items, drops, len(selected)


def run(staged_root: Path, sos_root: Path, scenes: list[str], frames_per_phase: int,
        seed: int, out_path: Path, max_per_type: int | None = 5):
    lookup = load_fewsol_lookup(sos_root)
    sos_catalog_names = sorted(p.name for p in sos_root.iterdir() if (p / "questionnaire.txt").is_file())
    folder_to_fewsol = build_folder_to_fewsol(lookup)  # built once, not per-frame
    rng = random.Random(seed)

    all_items = []
    for scene in scenes:
        scene_dir = staged_root / scene
        if not scene_dir.is_dir():
            print(f"[skip] {scene}: not staged under {staged_root}")
            continue
        print(f"=== {scene} ===")

        for phase in (0, 1, 2):
            root = scene_dir / str(phase)
            if not (root / "sam2" / "masks").is_dir():
                continue
            mapping = load_mapping(str(root / "sam2" / "mask_to_object.json"))
            items, n = gen_for_mos_phase(scene, phase, str(root), mapping, frames_per_phase,
                                          lookup, sos_catalog_names, folder_to_fewsol, rng, max_per_type)
            all_items += items
            print(f"  mos phase {phase}: {n} frames -> {len(items)} items")

        ego_root = scene_dir / "ego"
        if (ego_root / "sam2" / "masks").is_dir():
            mapping = load_mapping(str(ego_root / "sam2" / "mask_to_object.json"))
            items, n = gen_for_ego_phase(scene, str(ego_root), mapping, frames_per_phase,
                                         lookup, sos_catalog_names, folder_to_fewsol, rng, max_per_type)
            all_items += items
            print(f"  ego: {n} frames -> {len(items)} items")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")

    from collections import Counter
    print(f"\n{len(all_items)} total items -> {out_path}")
    print("by type:", dict(Counter(it["type"] for it in all_items)))
    return all_items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged-root", required=True, type=Path,
                     help="directory containing already-staged <scene>/<phase>/... data")
    ap.add_argument("--sos-root", required=True, type=Path,
                     help="local checkout of single_objects/sos_wrapped")
    scene_group = ap.add_mutually_exclusive_group(required=True)
    scene_group.add_argument("--scenes", nargs="+", help="explicit scene names, e.g. scene001 scene002")
    scene_group.add_argument("--tier", choices=["easy", "medium", "hard", "all"],
                              help="pull the scene list for this tier from splits/scene_splits.json")
    ap.add_argument("--frames-per-phase", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-type", type=int, default=5,
                     help="cap on spatial_lr_binary/spatial_ud_binary/spatial_farthest "
                          "and each attr_single_*/attr_composition field, per frame. "
                          "Pass 0 or a negative number for no cap (keep the full pool).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    scenes = args.scenes if args.scenes else get_scene_list(args.tier)
    print(f"scenes to process: {len(scenes)}")
    max_per_type = args.max_per_type if args.max_per_type and args.max_per_type > 0 else None
    run(args.staged_root, args.sos_root, scenes, args.frames_per_phase, args.seed, args.out, max_per_type)


if __name__ == "__main__":
    main()

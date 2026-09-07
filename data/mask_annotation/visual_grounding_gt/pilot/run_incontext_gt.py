"""Full 100-scene in-context (two-image) VQA GT generation. Same disk-bounded
staging pattern as run_full_dataset_gt.py: one scene at a time, delete after
GT is written (except KEEP_STAGED). Resumable via a .done marker file.

Generates exactly the three in-scope tasks (spec):
  - MOS in-context attribute/general bbox VQA   (inctx_attr_* types)
  - Ego in-context attribute/general bbox VQA    (inctx_attr_* types)
  - MOS in-context spatial bbox VQA              (inctx_spatial_farthest)

Untracked (pilot/ convention) -- the reused generator code lives in the
tracked worktree, imported explicitly below via WT_VQA_GT.
"""
import json
import os
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, os.path.dirname(WT_VQA_GT))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from generate_gt import get_scene_list, gen_incontext_for_mos_phase, gen_incontext_for_ego_phase  # noqa: E402
from generate_spatial_gt import load_mapping, select_frames  # noqa: E402
from lib import load_fewsol_lookup  # noqa: E402
import sos_catalog as sc  # noqa: E402
import sos_reference as sr  # noqa: E402

STAGE_ROOT = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/staged")
OUT_ROOT = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/incontext")
GENERAL_OUT = OUT_ROOT / "gt_incontext_general.jsonl"       # inctx_attr_* (mos + ego)
SPATIAL_OUT = OUT_ROOT / "gt_incontext_spatial.jsonl"        # inctx_spatial_farthest (mos only)
DROPS_OUT = OUT_ROOT / "gt_incontext_drops.jsonl"
DONE_PATH = OUT_ROOT / "gt_incontext.done"
SOS_ROOT = Path("/metadisk/itaykadosh/RPX/maskgen_2_scene_holder_for_refining/scenes_current_best/single_objects/sos_wrapped")
REF_MANIFEST = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops/reference_crops_v1.parquet")

FRAMES_PER_PHASE = 1000  # no-cap sentinel, matching run_full_dataset_gt.py
MAX_PER_TYPE = None       # full density -- no per-frame/per-phase cap on the in-context general family either
SEED = 42
REVISION = "main"
REQUIRE_DIFF_CATEGORY = True  # see report §"identity-only vs different-category feasibility"
KEEP_STAGED = {"scene001", "scene004", "scene010", "scene016", "scene046", "scene069", "scene098"}


def stage_phase(scene: str, kind: str, phase_idx):
    from huggingface_hub import hf_hub_download

    prefix = f"scenes/{scene}/{'ego' if kind == 'ego' else phase_idx}"
    dest = STAGE_ROOT / scene / (kind if kind == "ego" else str(phase_idx))
    marker = dest / ".staged"
    if marker.exists():
        return dest
    (dest / "rgb").mkdir(parents=True, exist_ok=True)
    (dest / "sam2").mkdir(parents=True, exist_ok=True)

    def pull(hf_path, subdest):
        for attempt in range(6):
            try:
                p = hf_hub_download("IRVLUTD/RPX", repo_type="dataset", filename=hf_path, revision=REVISION)
                break
            except Exception as e:
                if "429" not in str(e) or attempt == 5:
                    raise
                wait = 30 * (2 ** attempt)
                print(f"    429 on {hf_path}, backing off {wait}s (attempt {attempt+1}/6)", flush=True)
                time.sleep(wait)
        with tarfile.open(p) as tf:
            tf.extractall(subdest)
        real = os.path.realpath(p)
        os.remove(p)
        if os.path.exists(real):
            os.remove(real)

    pull(f"{prefix}/rgb.tar", dest)
    if (dest / "rgb" / "rgb").exists():
        for f in (dest / "rgb" / "rgb").iterdir():
            f.rename(dest / "rgb" / f.name)
        (dest / "rgb" / "rgb").rmdir()

    pull(f"{prefix}/labels/masks/v1.tar", dest / "sam2")
    if (dest / "sam2" / "sam2" / "masks").exists():
        (dest / "sam2" / "sam2" / "masks").rename(dest / "sam2" / "masks")

    pull(f"{prefix}/labels/sam2_meta/v1.tar", dest / "_meta_tmp")
    (dest / "_meta_tmp" / "sam2" / "mask_to_object.json").rename(dest / "sam2" / "mask_to_object.json")
    shutil.rmtree(dest / "_meta_tmp")
    shutil.rmtree(dest / "sam2" / "sam2", ignore_errors=True)

    if kind == "mos":
        (dest / "depth").mkdir(exist_ok=True)
        pull(f"{prefix}/depth.tar", dest)
        if (dest / "depth" / "depth").exists():
            for f in (dest / "depth" / "depth").iterdir():
                f.rename(dest / "depth" / f.name)
            (dest / "depth" / "depth").rmdir()

    marker.touch()
    return dest


def _write(f, items):
    for it in items:
        f.write(json.dumps(it) + "\n")


def _write_drops(f, drops, scene, kind, phase):
    for d in drops:
        f.write(json.dumps({"scene_id": scene, "kind": kind, "phase": phase,
                             "type": d.type, "reason": d.reason, "detail": d.detail}) + "\n")


def process_scene(scene, lookup, catalog, catalog_by_scid, ref_crops, rng, gen_f, spat_f, drop_f):
    total_general, total_spatial, total_drops = 0, 0, 0
    for phase in (0, 1, 2):
        dest = stage_phase(scene, "mos", phase)
        mapping = load_mapping(str(dest / "sam2" / "mask_to_object.json"))
        items, drops, n = gen_incontext_for_mos_phase(
            scene, phase, str(dest), mapping, FRAMES_PER_PHASE, lookup, catalog, catalog_by_scid,
            ref_crops, rng, SEED, REVISION, require_diff_category=REQUIRE_DIFF_CATEGORY, max_per_type=MAX_PER_TYPE)
        general_items = [it for it in items if it["type"] != "inctx_spatial_farthest"]
        spatial_items = [it for it in items if it["type"] == "inctx_spatial_farthest"]
        _write(gen_f, general_items)
        _write(spat_f, spatial_items)
        _write_drops(drop_f, drops, scene, "mos", phase)
        total_general += len(general_items)
        total_spatial += len(spatial_items)
        total_drops += len(drops)

    dest = stage_phase(scene, "ego", None)
    mapping = load_mapping(str(dest / "sam2" / "mask_to_object.json"))
    items, drops, n = gen_incontext_for_ego_phase(
        scene, str(dest), mapping, FRAMES_PER_PHASE, lookup, catalog, catalog_by_scid,
        ref_crops, rng, SEED, REVISION, require_diff_category=REQUIRE_DIFF_CATEGORY, max_per_type=MAX_PER_TYPE)
    _write(gen_f, items)
    _write_drops(drop_f, drops, scene, "ego", None)
    total_general += len(items)
    total_drops += len(drops)

    gen_f.flush()
    spat_f.flush()
    drop_f.flush()
    if scene not in KEEP_STAGED:
        shutil.rmtree(STAGE_ROOT / scene, ignore_errors=True)
    return total_general, total_spatial, total_drops


def main():
    lookup = load_fewsol_lookup(SOS_ROOT)
    catalog = sc.load_catalog(revision=REVISION)
    catalog_by_scid = sc.by_source_catalog_id(catalog)
    ref_crops_raw = sr.load_reference_manifest(REF_MANIFEST)
    ref_crops = {oid: sr.ReferenceCrop(**r) for oid, r in ref_crops_raw.items()}
    assert len(ref_crops) == 70, f"expected 70 reference crops, got {len(ref_crops)} -- run build_reference_crops.py first"
    rng = random.Random(SEED)
    scenes = get_scene_list("all")

    done = set(DONE_PATH.read_text().split()) if DONE_PATH.exists() else set()
    remaining = [s for s in scenes if s not in done]
    print(f"{len(scenes)} scenes total, {len(done)} already done, {len(remaining)} remaining", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(GENERAL_OUT, "a") as gen_f, open(SPATIAL_OUT, "a") as spat_f, \
            open(DROPS_OUT, "a") as drop_f, open(DONE_PATH, "a") as done_f:
        for i, scene in enumerate(remaining):
            print(f"[{i+1}/{len(remaining)}] {scene}...", flush=True)
            try:
                ng, ns, nd = process_scene(scene, lookup, catalog, catalog_by_scid, ref_crops, rng,
                                            gen_f, spat_f, drop_f)
                done_f.write(scene + "\n")
                done_f.flush()
                print(f"  -> {ng} general, {ns} spatial, {nd} drops", flush=True)
            except Exception as e:
                print(f"  [ERROR] {scene}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    print("done", flush=True)


if __name__ == "__main__":
    main()

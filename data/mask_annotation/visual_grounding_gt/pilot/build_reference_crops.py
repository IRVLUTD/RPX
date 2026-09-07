"""Driver: build the reference-crop cache for all 70 published SOS objects.
Untracked (pilot/ convention, matches run_full_dataset_gt.py) -- the reused
generator code lives in the tracked worktree, imported explicitly below."""
import sys
import time
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)

import sos_catalog as sc  # noqa: E402
import sos_reference as sr  # noqa: E402

OUT_DIR = Path("/home/rpx/Desktop/RPX/data/mask_annotation/visual_grounding_gt/pilot/out/reference_crops")
MANIFEST_PATH = OUT_DIR / "reference_crops_v1.parquet"


def main():
    catalog = sc.load_catalog()
    manifest = sr.load_reference_manifest(MANIFEST_PATH)
    print(f"{len(catalog)} catalog objects, {len(manifest)} already cached", flush=True)

    records = list(manifest.values())
    records = [sr.ReferenceCrop(**r) if not isinstance(r, sr.ReferenceCrop) else r for r in records]
    failures = []
    t0 = time.time()
    for i, (object_id, obj) in enumerate(sorted(catalog.items())):
        if object_id in manifest:
            continue
        try:
            rec = sr.build_reference_crop(object_id, obj, OUT_DIR / "crops")
            records.append(rec)
            print(f"[{i+1}/{len(catalog)}] {object_id}: frame {rec.frame_id} "
                  f"area_frac={rec.mask_area_frac:.4f} sharpness={rec.sharpness:.1f} "
                  f"sha256={rec.crop_sha256[:12]}", flush=True)
        except sr.NoAcceptableFrameError as e:
            failures.append((object_id, str(e)))
            print(f"[{i+1}/{len(catalog)}] {object_id}: DROPPED -- {e}", flush=True)

    sr.save_reference_manifest(records, MANIFEST_PATH)
    print(f"\n{len(records)}/{len(catalog)} crops built, {len(failures)} failures, "
          f"{time.time()-t0:.1f}s", flush=True)
    for object_id, reason in failures:
        print(f"  FAILED {object_id}: {reason}")


if __name__ == "__main__":
    main()

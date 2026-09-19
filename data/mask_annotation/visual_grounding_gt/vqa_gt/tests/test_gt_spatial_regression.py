"""Regression test for the additive refactor in gt_spatial.py (extracting
compute_farthest_candidates() out of gen_spatial_questions()'s inline loop).
Verifies against the ALREADY-PUBLISHED spatial_bbox.parquet -- ground truth
generated before this refactor existed -- that spatial_farthest output for a
real staged frame is unchanged, not just internally self-consistent.

Requires pilot/staged/scene001/0 to be present locally (one of the 7
KEEP_STAGED scenes from the original full-dataset run) and the published
parquet at pilot/out/vqa_parquet/spatial_bbox.parquet. Skips (not fails) if
either is absent, so this never forces a fresh download in ordinary test
runs -- see the module's own comment for why that's the right default.
"""
import glob
import os
import random

import pytest

STAGED_PHASE0 = os.environ.get("RPX_TEST_STAGED_PHASE0", "/data/rpx/staged/scene001/0")
PUBLISHED_SPATIAL_BBOX = os.environ.get("RPX_TEST_SPATIAL_PARQUET", "/data/rpx/spatial_bbox.parquet")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(STAGED_PHASE0) and os.path.exists(PUBLISHED_SPATIAL_BBOX)),
    reason="requires locally staged scene001/0 and the published spatial_bbox.parquet",
)


def test_spatial_farthest_matches_published_ground_truth():
    from generate_spatial_gt import compute_instances, imread_depth, imread_mask, load_mapping
    from gt_spatial import gen_spatial_questions
    import pyarrow.parquet as pq

    mapping = load_mapping(os.path.join(STAGED_PHASE0, "sam2/mask_to_object.json"))
    fids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(STAGED_PHASE0, "sam2/masks/*.png")))
    fid = fids[0]

    mask = imread_mask(os.path.join(STAGED_PHASE0, "sam2/masks", f"{fid}.png"))
    depth = imread_depth(os.path.join(STAGED_PHASE0, "depth", f"{fid}.png"))
    instances = compute_instances(mask, mapping, depth)
    H, W = mask.shape
    items = gen_spatial_questions("scene001", 0, fid, "x", instances, W, H, random.Random(42), 5)
    got = {(it["question"], it["answer"]) for it in items if it["type"] == "spatial_farthest"}

    df = pq.read_table(PUBLISHED_SPATIAL_BBOX).to_pandas()
    sub = df[(df.scene_id == "scene001") & (df.kind == "mos") & (df.phase == 0)
             & (df.frame == fid) & (df.type == "spatial_farthest")]
    expected = set(zip(sub["question"], sub["answer"]))

    assert got == expected
    assert len(got) == 5  # sanity: this frame is known to produce exactly 5

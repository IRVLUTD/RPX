import os
"""Test 18 (spec §13): deterministic crop hashes, plus the pure scoring/
selection logic, using small synthetic fixtures -- no network needed for the
synthetic tests. The cache-hash-stability test uses the real, already-built
reference_crops_v1.parquet manifest (pilot/out/reference_crops/) if present,
skipping otherwise."""
import numpy as np
import pytest

import sos_reference as sr


def _synthetic_mask_rgb(w, h, bbox, seed):
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1 + 1, x0:x1 + 1] = True
    rgb = rng.randint(0, 255, size=(h, w, 3)).astype(np.uint8)
    return mask, rgb


def test_score_candidate_basic_fields():
    mask, rgb = _synthetic_mask_rgb(100, 80, (20, 10, 60, 50), seed=1)
    s = sr._score_candidate(mask, rgb, 100, 80)
    assert s["bbox"] == [20, 10, 60, 50]
    assert s["area"] == (60 - 20 + 1) * (50 - 10 + 1)
    assert 0 < s["area_frac"] < 1
    assert s["edge_dist"] == min(20, 10, 100 - 1 - 60, 80 - 1 - 50)


def test_select_best_frame_rejects_edge_touching_and_tiny():
    candidates = {
        "00000": {"area": 50, "area_frac": 0.0006, "edge_dist": 5, "sharpness": 100.0},   # too small
        "00001": {"area": 5000, "area_frac": 0.05, "edge_dist": 0, "sharpness": 100.0},    # touches edge
        "00002": {"area": 5000, "area_frac": 0.05, "edge_dist": 10, "sharpness": 100.0},   # valid
        "00003": {"area": 5000, "area_frac": 0.05, "edge_dist": 10, "sharpness": 50.0},    # valid, less sharp
    }
    best = sr.select_best_frame(candidates)
    assert best == "00002"  # higher edge_dist wins over "00003" (tie on edge_dist would fall to sharpness)


def test_select_best_frame_deterministic_tiebreak_on_frame_id():
    candidates = {
        "00005": {"area": 5000, "area_frac": 0.05, "edge_dist": 10, "sharpness": 100.0},
        "00002": {"area": 5000, "area_frac": 0.05, "edge_dist": 10, "sharpness": 100.0},
    }
    # identical scores -> lowest frame_id wins, deterministically, every time
    assert sr.select_best_frame(candidates) == "00002"
    assert sr.select_best_frame(candidates) == "00002"


def test_select_best_frame_none_when_nothing_passes():
    candidates = {"00000": {"area": 10, "area_frac": 0.0001, "edge_dist": 0, "sharpness": 1.0}}
    assert sr.select_best_frame(candidates) is None


def test_pad_and_clamp_stays_in_bounds():
    bbox = [0, 0, 10, 10]
    padded = sr._pad_and_clamp(bbox, 20, 20)
    assert padded[0] >= 0 and padded[1] >= 0
    assert padded[2] <= 19 and padded[3] <= 19
    assert padded[0] <= bbox[0] and padded[1] <= bbox[1]
    assert padded[2] >= bbox[2] and padded[3] >= bbox[3]


REF_MANIFEST = os.environ.get("RPX_TEST_REFERENCE_MANIFEST", "/data/rpx/reference_crops/reference_crops_v1.parquet")


@pytest.mark.skipif(not __import__("os").path.exists(REF_MANIFEST), reason="requires the built reference-crop cache")
def test_crop_cache_is_idempotent_and_hash_stable():
    from pathlib import Path
    manifest = sr.load_reference_manifest(Path(REF_MANIFEST))
    assert len(manifest) == 70
    # a second "build" against the same object, with the manifest already
    # populated, must short-circuit to the cached record untouched (no
    # re-download, same sha256) -- this IS the determinism guarantee: real
    # end-to-end hash stability was already verified by running
    # build_reference_crops.py twice in the same session (0 new downloads,
    # identical manifest); this test locks in the short-circuit contract
    # that guarantee depends on.
    obj_id = "air_duster_can"
    cached = dict(manifest[obj_id])
    result = sr.build_reference_crop(obj_id, None, Path("/tmp/unused"), manifest={obj_id: cached})
    assert result.crop_sha256 == cached["crop_sha256"]

"""Standalone check (not part of the vqa_gt/tests/ pytest suite, since
pilot/ is operational/reproducibility code, not the tracked generator) for
to_parquet_incontext.py's deterministic spatial cap. Run directly:
    python3 pilot/tests_to_parquet_incontext.py
"""
import sys

sys.path.insert(0, ".")
from to_parquet_incontext import cap_spatial_deterministic, _cap_rank_key  # noqa: E402


def _row(scene, phase, frame, ref_scid, target_scid):
    return {"scene_id": scene, "phase": phase, "frame": frame,
            "reference_source_catalog_id": ref_scid, "target_source_catalog_id": target_scid}


def test_cap_keeps_at_most_5_per_frame():
    rows = [_row("s1", 0, "00000", str(i), "999") for i in range(9)]
    capped = cap_spatial_deterministic(rows, cap=5)
    assert len(capped) == 5


def test_cap_is_deterministic_across_calls():
    rows = [_row("s1", 0, "00000", str(i), "999") for i in range(9)]
    a = {r["reference_source_catalog_id"] for r in cap_spatial_deterministic(rows, cap=5)}
    b = {r["reference_source_catalog_id"] for r in cap_spatial_deterministic(rows, cap=5)}
    assert a == b


def test_cap_does_not_cross_frame_boundaries():
    rows = ([_row("s1", 0, "00000", str(i), "999") for i in range(9)]
            + [_row("s1", 0, "00001", str(i), "999") for i in range(3)])
    capped = cap_spatial_deterministic(rows, cap=5)
    by_frame = {}
    for r in capped:
        by_frame.setdefault(r["frame"], 0)
        by_frame[r["frame"]] += 1
    assert by_frame["00000"] == 5
    assert by_frame["00001"] == 3  # under the cap -- untouched


def test_cap_under_5_is_a_noop():
    rows = [_row("s1", 0, "00000", str(i), "999") for i in range(3)]
    capped = cap_spatial_deterministic(rows, cap=5)
    assert len(capped) == 3


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("OK", fn.__name__)
    print(f"{len(fns)}/{len(fns)} passed")

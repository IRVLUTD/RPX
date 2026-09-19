"""Q5: build a capped/paired version of the in-context general bank that
mirrors the PUBLISHED attribute generator's own capping convention -- WITHOUT
re-running generation or touching staged images.

Confirmed (Q3/Q4 of the audit): run_incontext_gt.py set MAX_PER_TYPE=None
uniformly for the whole general family. dedup.distribute() with cap=None
places every distinct (type, fact, target, reference) key in EVERY frame it
is valid in (pass 2's "fill every remaining valid placement" loop, see
dedup.py's own docstring) -- this is the documented, correct behavior for
the *uncapped* types in the original design (attr_count, attr_odd_one_out),
but the run used it for single_color/material/function/composition too,
which the published generator instead caps at 5 distinct facts/frame
(--max-per-type default). That mismatch, not a bug in the placement logic
itself, is why e.g. ego composition reached 126,497 rows over 23,050 frames
(avg 5.49/frame but max 20/frame -- a few facts valid across most of a
scene's frames, each contributing one row per valid frame).

This script does NOT need to re-derive candidates from images: the raw
uncapped bank already IS the full placement of every (fact, reference) pair
across every frame it was ever valid in (nothing was dropped by an uncapped
distribute -- it fills every valid slot). So the complete frame-validity
information dedup.distribute() needs is already recoverable from the raw
JSONL alone: group the existing rows back into dedup.py's
{frame_id: {type: [(key, item), ...]}} input shape (reconstructing each
row's original internal dedup key from its own stored fields -- exactly the
same key generation logic gt_incontext.py used at generation time), then
call distribute() again with the PUBLISHED caps (5 for single_color/
material/function/composition, None for odd_one_out).

The reference (target, fact) pairing itself is NOT re-decided here -- it
was already fixed, deterministically, at generation time and is preserved
verbatim on every kept row. Only WHICH of that fact's valid frames get to
keep a row is redecided, using a freshly seeded RNG per (scene, kind,
phase) group -- documented, deterministic, but NOT a bit-exact replay of
"what a --max-per-type=5 rerun would have produced" (that would require
re-collecting candidates from images, i.e. restaging, which this explicitly
avoids per the no-rerun constraint).

Writes NEW files, does not touch gt_incontext_general.jsonl or any existing
Parquet.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

WT_VQA_GT = "/home/rpx/Desktop/RPX-worktrees/itay-vqa-incontext/data/mask_annotation/visual_grounding_gt/vqa_gt"
sys.path.insert(0, WT_VQA_GT)
sys.path.insert(0, str(Path(WT_VQA_GT).parent))
from sos_catalog import normalize_value  # noqa: E402
import dedup  # noqa: E402

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "out/incontext/gt_incontext_general.jsonl"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "out/incontext_paired/gt_incontext_general_capped.jsonl"
BASE_SEED = 42
PUBLISHED_CAPS = {
    "inctx_attr_single_color": 5, "inctx_attr_single_material": 5,
    "inctx_attr_single_function": 5, "inctx_attr_composition": 5,
    "inctx_attr_odd_one_out": None,
}


def rebuild_fact_key(row):
    t = row["type"]
    if t == "inctx_attr_composition":
        return f"composition:{normalize_value(row['attribute_material'])}|{normalize_value(row['attribute_function'])}"
    if t == "inctx_attr_odd_one_out":
        return f"odd_one_out:{normalize_value(row['attribute_material'])}"
    return f"{row['attribute_kind']}:{normalize_value(row['attribute_value'])}"


def rebuild_dedup_key(row):
    fact_key = rebuild_fact_key(row)
    return f"{row['type']}|{fact_key}|{row['target_source_catalog_id']}|{row['reference_source_catalog_id']}"


def group_seed(scene_id, kind, phase):
    key = f"{BASE_SEED}|{scene_id}|{kind}|{phase}"
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main():
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    print(f"loading raw bank from {IN_PATH} ...", flush=True)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (scene,kind,phase) -> frame -> type -> [(key,row)]
    n_in = 0
    with open(IN_PATH) as f:
        for line in f:
            row = json.loads(line)
            n_in += 1
            gkey = (row["scene_id"], row["kind"], row["phase"])
            dedup_key = rebuild_dedup_key(row)
            groups[gkey][row["frame"]][row["type"]].append((dedup_key, row))
            if n_in % 200000 == 0:
                print(f"  ... {n_in} lines read", flush=True)
    print(f"{n_in} raw rows, {len(groups)} (scene,kind,phase) groups", flush=True)

    n_out = 0
    with open(OUT_PATH, "w") as out_f:
        for i, (gkey, frame_candidates) in enumerate(groups.items()):
            scene_id, kind, phase = gkey
            rng = random.Random(group_seed(scene_id, kind, phase))
            # dedup.distribute expects plain dicts, not defaultdicts, for its
            # own internal .get()/.setdefault() calls to behave as documented
            fc = {fid: dict(types) for fid, types in frame_candidates.items()}
            assigned = dedup.distribute(fc, PUBLISHED_CAPS, rng)
            for fid, items in assigned.items():
                for row in items:
                    out_f.write(json.dumps(row) + "\n")
                    n_out += 1
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(groups)} groups recapped", flush=True)

    print(f"\n{n_in} raw rows -> {n_out} capped rows ({n_in - n_out} removed by the published-mirroring cap)")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()

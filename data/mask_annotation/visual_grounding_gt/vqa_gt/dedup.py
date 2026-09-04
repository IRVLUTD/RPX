"""Phase-wide, order-fair candidate distribution.

Within one phase the object arrangement is static (confirmed visually: a
phase is a camera sweep around one fixed setup, not a sequence of different
arrangements), so several question types have a much smaller real pool of
distinct facts than (frames in phase) x (per-frame cap) would suggest.
Generating each frame independently produces two separate problems:

  1. Heavy repetition -- measured up to 96.7% on attr_count in one phase
     (every frame asked the same 3 facts because nothing capped or varied
     them), 56% on attr_composition (a large but still finite real pool,
     independently resampled 30 times).
  2. Order bias -- an earlier, simpler dedup fix (process frames in a fixed
     order, greedily take unseen facts first) solved (1) but introduced a
     new problem: coverage silently concentrates in low frame-IDs and the
     back half of every phase becomes pure repeat filler. A benchmarker
     subsampling this dataset's frames would get very different coverage
     depending on which frames they kept.

This module fixes both at once: every candidate fact is distributed across
the frames it's actually valid in (an object can be occluded from some
viewpoints even within a "static" phase, so validity is genuinely
per-frame, not phase-wide) as evenly as a round-robin allows, before
anything repeats -- and repeats, once necessary, are spread the same way
instead of dumped on whichever frames happen to run out of fresh material
last.
"""
import random


def distribute(frame_candidates: dict, max_per_type, rng: random.Random) -> dict:
    """frame_candidates: {frame_id: {type_name: [(key, item), ...]}} -- the
    full, uncapped candidate list per type, per frame. `key` identifies the
    underlying fact (e.g. the (material, function) pair for attr_composition,
    or the reference object for spatial_farthest) -- the same key can appear
    under multiple frame_ids if that fact is valid/visible in more than one
    of them, which is exactly the redundancy this function manages.

    max_per_type: either a single value (int cap, or None for no cap)
    applied to every type, or a dict[type_name -> cap] for per-type
    overrides (e.g. attr_synonym_yes/no capped at 1 each while
    attr_composition allows more) -- a type missing from the dict falls
    back to no cap.

    Returns {frame_id: [item, ...]} -- items chosen per frame, capped per
    type. Every distinct key gets placed at least once (in whichever of its
    valid frames currently has the lightest load for that type) before any
    key repeats; once every frame hits its cap or every key has been
    exhausted, remaining slots are filled with repeats using the same
    least-loaded-first rule, so repetition is spread evenly rather than
    concentrated.
    """
    frame_ids = list(frame_candidates.keys())
    types = set()
    for d in frame_candidates.values():
        types.update(d.keys())

    result = {fid: [] for fid in frame_ids}
    per_type_cap = max_per_type if isinstance(max_per_type, dict) else {t: max_per_type for t in types}

    for t in sorted(types):
        max_per_type = per_type_cap.get(t)
        by_key = {}
        for fid in frame_ids:
            for key, item in frame_candidates[fid].get(t, []):
                by_key.setdefault(key, []).append((fid, item))
        keys = list(by_key.keys())
        rng.shuffle(keys)

        load = {fid: 0 for fid in frame_ids}
        used_in_frame = {fid: set() for fid in frame_ids}  # keys already placed in this frame, this type

        def cap_ok(fid):
            return max_per_type is None or load[fid] < max_per_type

        def try_place(key):
            occs = [(fid, item) for fid, item in by_key[key]
                    if cap_ok(fid) and key not in used_in_frame[fid]]
            if not occs:
                return False
            occs.sort(key=lambda x: (load[x[0]], rng.random()))
            fid, item = occs[0]
            result[fid].append(item)
            load[fid] += 1
            used_in_frame[fid].add(key)
            return True

        # pass 1: every key gets one placement, spread to whichever of its
        # valid frames is currently lightest-loaded for this type
        for key in keys:
            try_place(key)

        # pass 2: fill every remaining valid placement (repeats), same
        # least-loaded-first spreading, looping until no frame has room left
        # (capped types) or every key has been placed in all its valid
        # frames (uncapped types -- e.g. attr_count, where a fact genuinely
        # is true in every frame it's valid in and there's no scarce budget
        # to be fair about, so full coverage is the correct outcome, not a
        # dedup target). Runs for BOTH capped and uncapped types -- an
        # earlier version of this loop only ran when max_per_type was set,
        # which silently limited every uncapped type to one placement total
        # instead of one per valid frame.
        progress = True
        while progress:
            progress = False
            for key in keys:
                if try_place(key):
                    progress = True

    return result

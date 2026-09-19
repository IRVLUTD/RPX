#!/usr/bin/env python3
"""Resume the stalled lossless-convert run: png-recompress + link phases only
(webp/rgb is already 100% done and verified). Submits work in BOUNDED CHUNKS
via a persistent ProcessPoolExecutor instead of the original tool's
"submit all N futures in one list comprehension" pattern -- that pattern is
almost certainly what deadlocked the original run at the png-recompress
phase (34,459 futures submitted at once; webp's ~22,899 apparently stayed
under whatever threshold, this one didn't). Idempotent: skips any dst file
that already exists.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/rpx/Desktop/RPX/benchmark")

from rpx_benchmark.dataset_hub.lossless_convert import (  # noqa: E402
    ConvertSpec,
    _plan_tree,
    _recompress_png_one,
    _link_one,
    ACTION_WEBP,
    ACTION_PNG_RECOMPRESS,
    ACTION_LINK,
    ACTION_CAM_POSE,
)

SRC = Path("/home/rpx/Downloads/RPX_HF_STAGE/DATA_ego_full")
OUT = Path("/home/rpx/Downloads/RPX_HF_STAGE/DATA_ego_full_v2")
CHUNK = 1000
WORKERS = 19

spec = ConvertSpec(src_root=SRC, out_root=OUT, workers=WORKERS)

print(f"[{time.strftime('%H:%M:%S')}] planning...", flush=True)
plan = _plan_tree(SRC, spec=spec)
by_action = {}
for src_p, action in plan:
    by_action.setdefault(action, []).append(src_p)
print(
    f"[{time.strftime('%H:%M:%S')}] plan: "
    + ", ".join(f"{a}={len(v)}" for a, v in sorted(by_action.items())),
    flush=True,
)


def dst_for(src_path: Path, action: str) -> Path:
    rel = src_path.relative_to(SRC)
    if action == ACTION_PNG_RECOMPRESS:
        return OUT / rel
    if action == ACTION_LINK:
        return OUT / rel
    raise ValueError(action)


def run_bounded(action: str, worker_fn, extra_args, items):
    todo = []
    skipped = 0
    for src_p in items:
        dst_p = dst_for(src_p, action)
        if dst_p.exists():
            skipped += 1
            continue
        todo.append((str(src_p), str(dst_p)))
    print(
        f"[{time.strftime('%H:%M:%S')}] {action}: {len(todo)} to do, "
        f"{skipped} already done, chunk={CHUNK}",
        flush=True,
    )
    done = 0
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i in range(0, len(todo), CHUNK):
            batch = todo[i : i + CHUNK]
            futures = [ex.submit(worker_fn, s, d, *extra_args) for s, d in batch]
            for fut in as_completed(futures):
                _, before_b, after_b, err = fut.result()
                done += 1
                if err:
                    failures.append(err)
            print(
                f"[{time.strftime('%H:%M:%S')}] {action}: {done}/{len(todo)} done "
                f"({len(failures)} failures so far)",
                flush=True,
            )
    return done, failures


total_failures = []

if ACTION_PNG_RECOMPRESS in by_action:
    n, fails = run_bounded(
        ACTION_PNG_RECOMPRESS,
        _recompress_png_one,
        (spec.verify, spec.png_compress_level),
        [Path(p) for p in by_action[ACTION_PNG_RECOMPRESS]],
    )
    total_failures += [f"[png-recompress] {f}" for f in fails]
    if fails:
        print(f"ABORTING after png-recompress: {len(fails)} failures, first: {fails[0]}", flush=True)
        sys.exit(1)

if ACTION_LINK in by_action:
    n, fails = run_bounded(
        ACTION_LINK,
        _link_one,
        (),
        [Path(p) for p in by_action[ACTION_LINK]],
    )
    total_failures += [f"[link] {f}" for f in fails]

print(f"[{time.strftime('%H:%M:%S')}] DONE. total_failures={len(total_failures)}", flush=True)
if total_failures:
    for f in total_failures[:20]:
        print("  FAILURE:", f, flush=True)
    sys.exit(1)
print("SUCCESS", flush=True)

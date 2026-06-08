#!/usr/bin/env python3
"""Bridge ``review_faulty_masks`` output → ``refine_masks_flow`` input.

The reviewer writes a list of *verified* frames to
``<phase_dir>/verified_masks.txt`` (full-ID format
``<scene>/<phase>/<frame>``). The refiner expects a list of *faulty*
integer frame IDs at ``<phase_dir>/sam2/iter{N}_faulty.txt``. This module
computes ``all_masks ∖ verified`` and writes the faulty list in the
refiner's format.

Usage::

    python -m maskgen_pipeline.gen_faulty_from_verified \\
        --scene_dir <phase_dir> --iter 1

Or as a library:

    from maskgen_pipeline.gen_faulty_from_verified import gen_faulty
    n = gen_faulty(Path("scene/0"), iter_num=1)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def gen_faulty(phase_dir: Path, iter_num: int = 1) -> int:
    """Compute the faulty-frame list for one phase. Returns the number of faulty frames."""
    phase_dir = Path(phase_dir)
    masks_dir = phase_dir / "sam2" / "masks"
    verified_fp = phase_dir / "verified_masks.txt"
    out_fp = phase_dir / "sam2" / f"iter{iter_num}_faulty.txt"

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks dir missing: {masks_dir}")

    all_ints = {int(p.stem) for p in masks_dir.glob("*.png") if p.stem.isdigit()}
    verified_ints: set[int] = set()
    if verified_fp.is_file():
        for line in verified_fp.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            # full-ID line: "<scene>/<phase>/<frame_id>"
            frame_id = s.rsplit("/", 1)[-1]
            try:
                verified_ints.add(int(frame_id))
            except ValueError:
                continue

    faulty = sorted(all_ints - verified_ints)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    if faulty:
        out_fp.write_text("\n".join(str(i) for i in faulty) + "\n")
    else:
        out_fp.write_text("")
    return len(faulty)


def main():
    ap = argparse.ArgumentParser(
        description="Generate iter{N}_faulty.txt from the reviewer's verified_masks.txt"
    )
    ap.add_argument("--scene_dir", type=str, required=True,
                    help="Phase directory (contains rgb/, sam2/, verified_masks.txt)")
    ap.add_argument("--iter", type=int, default=1,
                    help="Iteration number (writes sam2/iter{N}_faulty.txt). Default 1.")
    args = ap.parse_args()

    n = gen_faulty(Path(args.scene_dir), iter_num=args.iter)
    out = Path(args.scene_dir) / "sam2" / f"iter{args.iter}_faulty.txt"
    print(f"  wrote {n} faulty frame IDs → {out}")
    if n == 0:
        print("  (no faulty frames — nothing for refine_masks_flow to do)")


if __name__ == "__main__":
    main()

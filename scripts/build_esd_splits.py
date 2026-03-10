#!/usr/bin/env python3
"""
scripts/build_esd_splits.py
---------------------------
Compute and save Effort-Stratified Difficulty (ESD) splits for the RPX dataset.

Run this once on your local copy of the dataset:

    python scripts/build_esd_splits.py --root_dir /path/to/rpx/data

This produces:  splits/esd_splits.json

The JSON file is committed alongside the dataset and loaded automatically
by SplitManager.from_root() on subsequent uses.
"""

import argparse
from pathlib import Path
import sys

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rpx.data.splits import SplitManager, DIFFICULTIES, PHASES


def main():
    parser = argparse.ArgumentParser(
        description="Compute ESD splits from RPX annotation refinement statistics."
    )
    parser.add_argument(
        '--root_dir', type=str, required=True,
        help='Root directory of the RPX dataset (containing recorded_data/ or data/).'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for the JSON split file. '
             'Default: <root_dir>/splits/esd_splits.json'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Print difficulty distribution after computing.'
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    output = Path(args.output) if args.output else root / 'splits' / 'esd_splits.json'

    print(f"Computing ESD splits from: {root}")
    sm = SplitManager(root_dir=str(root))
    sm.build()
    sm.save(str(output))

    if args.summary:
        sm.summary()

    # Also print per-phase × per-difficulty breakdown
    labels = sm.get_all_labels()
    print("\n── Phase × Difficulty breakdown ──")
    for phase in PHASES:
        for diff in DIFFICULTIES:
            count = sum(
                1 for (_, p, _), d in labels.items()
                if p == phase and d == diff
            )
            print(f"  {phase:8s} × {diff:6s}: {count:5d} frames")


if __name__ == '__main__':
    main()

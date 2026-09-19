#!/usr/bin/env python3
"""Download RPX RGB-D segmentation shards through the published Quick Start API."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RPX rgbd_segmentation shards for all difficulty splits."
    )
    parser.add_argument("--repo-id", default="IRVLUTD/RPX")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from rpx_benchmark.dataset_hub import download_for_task

    for split in args.splits:
        result = download_for_task(
            task="rgbd_segmentation",
            split=split,
            repo_id=args.repo_id,
        )
        print(f"{split}: {result.local_dir}")
        print(f"  matched scenes: {len(result.matched_scenes)}")


if __name__ == "__main__":
    main()


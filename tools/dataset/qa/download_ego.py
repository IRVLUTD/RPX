#!/usr/bin/env python3
"""Download RPX egocentric MOS shards through the Hugging Face Hub API."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="anonymous/RPX")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Scene IDs to download, e.g. scene001 scene002. Default: all ego scene shards.",
    )
    parser.add_argument("--local-dir", type=Path, default=Path("rpx_ego_download"))
    parser.add_argument(
        "--include-preview",
        action="store_true",
        help="Also download ego preview/Data Studio files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from huggingface_hub import snapshot_download

    scene_patterns = (
        [f"scenes/{scene_id}/ego/**" for scene_id in args.scenes]
        if args.scenes
        else ["scenes/*/ego/**"]
    )
    allow_patterns = [
        "README.md",
        "manifest/current.json",
        "manifest/ego_frames_v1.csv",
        "manifest/ego_frames_v1.parquet",
        "manifest/frames_v2.parquet",
        "manifest/mos_ego_mask_object_map_v1.csv",
        "manifest/mos_ego_mask_object_map_v1.parquet",
        *scene_patterns,
    ]
    if args.include_preview:
        allow_patterns.extend(
            [
                "preview/ego_preview.csv",
                "preview/ego_preview.parquet",
                "preview/image_examples/ego_preview/**",
            ]
        )

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    print(local_dir)
    print("downloaded ego patterns:")
    for pattern in allow_patterns:
        print(f"  {pattern}")


if __name__ == "__main__":
    main()

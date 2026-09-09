#!/usr/bin/env python3
"""Fetch and verify every image required by a frozen VQA manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from rpx_benchmark.vqa.contract import load_manifest
from rpx_benchmark.vqa.hub_rgb import fetch_images_many


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(
        f"prefetching {len(samples)} rows; manifest_sha256={digest}",
        flush=True,
    )
    paths = fetch_images_many(samples, args.image_cache)
    missing = [sample.sample_id for sample in samples if sample.sample_id not in paths]
    if missing:
        raise SystemExit(f"prefetch coverage failure: {len(missing)} missing")
    print(f"PREFETCH COMPLETE: {len(paths)}/{len(samples)} rows", flush=True)


if __name__ == "__main__":
    main()

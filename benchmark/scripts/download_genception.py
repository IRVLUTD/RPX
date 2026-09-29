#!/usr/bin/env python3
"""Download, size-check and checksum official GenCeption assets for RPX."""

from __future__ import annotations

import argparse
from pathlib import Path

from genception_runtime import download_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("1.3b", "14b", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("~/.cache/rpx/genception").expanduser())
    args = parser.parse_args()
    variants = ("1.3b", "14b") if args.variant == "all" else (args.variant,)
    for variant in variants:
        manifest = download_assets(args.output, variant)
        print(f"ready {variant}: {manifest.checkpoint_dir}")


if __name__ == "__main__":
    main()

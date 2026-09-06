#!/usr/bin/env python3
"""Fetch the two current bbox VQA parquets at a frozen Hub revision."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REVISION = "614c90ff4bf6b5ddb314f6052d1c50c0cc034f8c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("attribute.parquet", "spatial_bbox.parquet"):
        source = hf_hub_download(
            repo_id="IRVLUTD/RPX",
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=f"vqa/{name}",
        )
        target = args.out / name
        if not target.exists():
            shutil.copy2(source, target)
        print(target)


if __name__ == "__main__":
    main()

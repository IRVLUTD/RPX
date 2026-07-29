#!/usr/bin/env python3
"""Operator launcher for all three RPX D3 splits followed by paper analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PINNED_DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sam2"], default="sam2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo", default="IRVLUTD/RPX")
    parser.add_argument("--revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--jedi-bounds")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    output_root = Path(args.output_root) / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    cells: list[str] = []
    for split in ("easy", "medium", "hard"):
        output_dir = output_root / split
        command = [
            sys.executable,
            str(scripts / "run_tracking.py"),
            "--model",
            args.model,
            "--split",
            split,
            "--repo",
            args.repo,
            "--revision",
            args.revision,
            "--cache-dir",
            args.cache_dir,
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--save-predictions",
            "--resume-predictions",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=os.environ.copy())
        cells.append(str(output_dir / "cells.parquet"))

    analysis_command = [
        sys.executable,
        str(scripts / "analyze_tracking_paper.py"),
        "--cells",
        *cells,
        "--output-dir",
        str(output_root),
    ]
    if args.jedi_bounds:
        analysis_command.extend(["--jedi-bounds", args.jedi_bounds])
    print("+", " ".join(analysis_command), flush=True)
    subprocess.run(analysis_command, check=True, env=os.environ.copy())
    print(f"RPX D3 complete: {output_root}")


if __name__ == "__main__":
    main()

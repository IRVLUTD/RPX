#!/usr/bin/env python3
"""Operator launcher for all three RPX D3 splits followed by paper analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tracking import TRACKING_DATASETS  # noqa: E402
from tracking_models import TRACKER_CLASSES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(TRACKER_CLASSES), default="sam2")
    parser.add_argument(
        "--dataset-protocol",
        choices=sorted(TRACKING_DATASETS),
        default="mos",
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo", default="anonymous/RPX")
    parser.add_argument("--revision")
    parser.add_argument("--jedi-bounds")
    parser.add_argument("--text-vocab")
    args = parser.parse_args()
    expected_revision = str(TRACKING_DATASETS[args.dataset_protocol]["revision"])
    if args.revision is None:
        args.revision = expected_revision
    if args.revision != expected_revision:
        raise SystemExit(
            f"{args.dataset_protocol} tracking requires dataset revision "
            f"{expected_revision}; got {args.revision}."
        )

    scripts = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    if args.dataset_protocol == "ego":
        output_root /= "ego"
    output_root /= args.model
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
            "--dataset-protocol",
            args.dataset_protocol,
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
        if args.text_vocab:
            command.extend(["--text-vocab", args.text_vocab])
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
        "--dataset-protocol",
        args.dataset_protocol,
    ]
    if args.jedi_bounds:
        analysis_command.extend(["--jedi-bounds", args.jedi_bounds])
    print("+", " ".join(analysis_command), flush=True)
    subprocess.run(analysis_command, check=True, env=os.environ.copy())
    print(f"RPX {args.dataset_protocol} tracking complete: {output_root}")


if __name__ == "__main__":
    main()

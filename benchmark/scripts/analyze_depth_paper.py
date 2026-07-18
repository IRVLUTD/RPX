#!/usr/bin/env python3
"""Generate the RPX D1-F paper statistics from Easy/Medium/Hard cell logs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Support direct source-tree execution as well as the installed Docker package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpx_benchmark.metrics.depth_paper import D1_CALIBRATION, PAPER_METRIC_KEYS
from rpx_benchmark.paper_depth_analysis import analyze_depth_cells, write_depth_analysis

PINNED_DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"


def _git_sha() -> str:
    env_sha = os.environ.get("RPX_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs=3, type=Path, help="Easy Medium Hard cells.parquet")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jedi-bounds", type=Path)
    args = parser.parse_args()

    analysis, cells = analyze_depth_cells(
        args.cells,
        jedi_bounds_path=args.jedi_bounds,
        provenance={
            "dataset_repo": "IRVLUTD/RPX",
            "dataset_revision": PINNED_DATASET_REVISION,
            "model_checkpoint": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            "git_sha": _git_sha(),
            "docker_digest": os.environ.get("RPX_DOCKER_DIGEST", "unknown"),
            "metric_order": list(PAPER_METRIC_KEYS),
            "calibration": D1_CALIBRATION.to_dict(),
            "formulas": {
                "standardization": "direction-normalize, then global population z-score over 300 cells",
                "manova": "H=SSP_reduced-SSP_full; E=SSP_full",
                "headline_phi": "wilks_lambda^(1/min(p,df_h))",
                "transition_phi": "one-df paired Hotelling wilks_lambda",
                "fscore": "harmonic mean of bidirectional NN fractions with strict distance<0.05m",
            },
        },
    )
    outputs = write_depth_analysis(analysis, cells, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

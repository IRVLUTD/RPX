#!/usr/bin/env python3
"""Combine three RPX D3 split logs and run paper statistics."""

from __future__ import annotations

import argparse
import json

from rpx_benchmark.paper_tracking_analysis import (
    analyze_tracking_cells,
    write_tracking_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jedi-bounds")
    args = parser.parse_args()
    analysis, cells = analyze_tracking_cells(
        args.cells,
        jedi_bounds_path=args.jedi_bounds,
    )
    paths = write_tracking_analysis(analysis, cells, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

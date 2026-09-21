#!/usr/bin/env python3
"""Build the RPX Python API documentation under ``toolkit-docs/``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("_site"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / ".nojekyll").touch()
    api_output = output / "toolkit-docs"
    subprocess.run(
        [sys.executable, "-m", "pdoc", "rpx_benchmark", "--output-directory", str(api_output)],
        check=True,
    )
    if not (api_output / "index.html").is_file():
        raise RuntimeError("pdoc did not create toolkit-docs/index.html")


if __name__ == "__main__":
    main()

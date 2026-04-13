"""Minimal runnable BYO-model example.

Runs the monocular-depth pipeline against an in-process synthetic
dataset with a stub model. No network, no torch, no external data —
useful for a quick sanity check that the install is healthy.

Run::

    make smoke
    # or
    python examples/run_depth.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import rpx_benchmark as rpx
from rpx_benchmark.loader import RPXDataset
from rpx_benchmark.runner import BenchmarkRunner


def constant_depth(rgb: np.ndarray) -> np.ndarray:
    """Stub model: always predict 2.0m depth everywhere."""
    return np.full(rgb.shape[:2], 2.0, dtype=np.float32)


def make_synthetic_manifest(root: Path, n: int = 3) -> Path:
    """Write N RGB + depth PNG pairs and a matching JSON manifest."""
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(n):
        Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(rgb_dir / f"{i}.png")
        Image.fromarray(np.full((32, 32), 2000, np.uint16)).save(depth_dir / f"{i}.png")
        samples.append({
            "id": f"synth_{i}",
            "rgb": f"rgb/{i}.png",
            "depth": f"depth/{i}.png",
            "phase": "clutter",
            "difficulty": "easy",
        })

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "task": "monocular_depth",
        "root": str(root),
        "samples": samples,
    }))
    return manifest


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_synthetic_manifest(Path(tmp), n=4)
        dataset = RPXDataset.from_manifest(manifest, batch_size=2)

        model = rpx.make_numpy_depth_model(constant_depth, name="constant_2m")
        runner = BenchmarkRunner(model=model, dataset=dataset)
        result = runner.run()

        print(f"ran {result.num_samples} samples")
        print("aggregated:", result.aggregated)


if __name__ == "__main__":
    main()

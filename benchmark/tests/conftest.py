"""Shared pytest fixtures for RPX benchmark tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rpx_benchmark.loader import RPXDataset


def _write_phase(
    root: Path,
    scene: str,
    phase_idx: str,
    frames: list[str],
    h: int = 60,
    w: int = 80,
) -> None:
    pdir = root / "scenes" / scene / phase_idx
    (pdir / "rgb").mkdir(parents=True, exist_ok=True)
    (pdir / "depth").mkdir(parents=True, exist_ok=True)
    for f in frames:
        Image.fromarray(np.full((h, w, 3), 128, np.uint8)).save(pdir / "rgb" / f"{f}.png")
        Image.fromarray(np.full((h, w), 2000, np.uint16)).save(pdir / "depth" / f"{f}.png")


@pytest.fixture
def synthetic_depth_dataset(tmp_path: Path) -> RPXDataset:
    """6-frame synthetic monocular-depth dataset across 3 phases × 2 difficulties."""
    samples = []
    for phase_idx, phase_name in (("0", "clutter"), ("1", "interaction"), ("2", "clean")):
        for difficulty in ("easy", "hard"):
            frame = f"0{difficulty[0]}"
            _write_phase(tmp_path, "scene_000", phase_idx, [frame])
            samples.append({
                "id": f"scene_000_{phase_name}_{frame}",
                "rgb":   f"scenes/scene_000/{phase_idx}/rgb/{frame}.png",
                "depth": f"scenes/scene_000/{phase_idx}/depth/{frame}.png",
                "phase": phase_name,
                "difficulty": difficulty,
            })
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "task": "monocular_depth",
        "root": str(tmp_path),
        "samples": samples,
    }))
    return RPXDataset.from_manifest(manifest_path, batch_size=1)

#!/usr/bin/env python
"""Build HuggingFace-datasets-ready Parquet shards from an RPX manifest.

Reads the legacy JSON manifest layout produced by the RPX capture
pipeline (``scenes/<scene>/<phase>/{rgb,depth,mask,pose,...}``) and
writes one Parquet shard per (task, difficulty split) pair plus a
``README.md`` frontmatter block that maps task configs to splits for
``datasets.load_dataset``.

This script is *offline-safe*: no Hub calls, no model downloads. Once
it finishes, ``huggingface-cli upload <dataset_repo> <output_dir>``
ships the artefacts.

Example
-------

::

    python scripts/build_hf_shards.py \\
        --source-root /mnt/rpx/scenes \\
        --manifests-root manifests/ \\
        --out build/hf_shards/

``manifests/`` is expected to hold files named
``{task}/{split}.json`` following the structure that
:func:`rpx_benchmark.hub.fetch_manifest` produces.
"""

from __future__ import annotations

import argparse
import json

# Local imports: allow running from the repo without ``pip install -e``.
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rpx_benchmark.api import Difficulty, TaskType  # noqa: E402


def _read_png_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _pose_matrix(path: Path) -> List[float]:
    """Load a T265 NPZ and flatten the 4×4 SE(3) camera-to-world pose."""
    data = np.load(path)
    position = data["position"].astype(np.float64)
    qx, qy, qz, qw = data["orientation"].astype(np.float64)
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = position.astype(np.float32)
    return T.flatten().tolist()


def _row_for_depth(entry: Dict[str, Any], root: Path) -> Dict[str, Any]:
    return {
        "id": entry["id"],
        "scene": entry.get("scene", ""),
        "phase": str(entry.get("phase", "")),
        "difficulty": str(entry.get("difficulty", "")),
        "rgb": _read_png_bytes(root / entry["rgb"]),
        "depth": _read_png_bytes(root / entry["depth"]),
        "camera_pose": _pose_matrix(root / entry["pose"]) if entry.get("pose") else [0.0] * 16,
    }


def _row_for_segmentation(entry: Dict[str, Any], root: Path) -> Dict[str, Any]:
    return {
        "id": entry["id"],
        "scene": entry.get("scene", ""),
        "phase": str(entry.get("phase", "")),
        "difficulty": str(entry.get("difficulty", "")),
        "rgb": _read_png_bytes(root / entry["rgb"]),
        "mask": _read_png_bytes(root / entry["mask"]),
        "camera_pose": _pose_matrix(root / entry["pose"]) if entry.get("pose") else [0.0] * 16,
    }


_ROW_BUILDERS = {
    TaskType.MONOCULAR_DEPTH: _row_for_depth,
    TaskType.OBJECT_SEGMENTATION: _row_for_segmentation,
    # Additional task row-builders plug in here. Keeping the ship-set
    # small on purpose: once M2 lands the full 10-task row set this
    # table will dispatch across every task.
}


def _write_readme(out_dir: Path, task_splits: Dict[str, List[str]]) -> None:
    """Write a README.md with dataset-configs frontmatter.

    The frontmatter tells ``datasets.load_dataset`` how to map config
    names (tasks) to splits (difficulties). Without it, the Hub would
    not know which shards belong to which config.
    """
    lines = ["---", "configs:"]
    for task, splits in sorted(task_splits.items()):
        lines.append(f"  - config_name: {task}")
        lines.append("    data_files:")
        for split in sorted(splits):
            lines.append(f"      - split: {split}")
            lines.append(f'        path: "shards/{task}/{split}-*.parquet"')
    lines += ["---", "", "# RPX — Robot Perception X"]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def build(source_root: Path, manifests_root: Path, out_dir: Path) -> None:
    try:
        from datasets import Dataset  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "scripts/build_hf_shards.py requires `datasets`. "
            "Install with: pip install 'rpx-benchmark[hf-datasets]'"
        ) from e

    from rpx_benchmark.data.features import features_for_task  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_root = out_dir / "shards"
    shard_root.mkdir(exist_ok=True)

    task_splits: Dict[str, List[str]] = {}

    for task in TaskType:
        builder = _ROW_BUILDERS.get(task)
        if builder is None:
            print(f"[skip] {task.value}: no row builder wired yet")
            continue
        task_dir = manifests_root / task.value
        if not task_dir.is_dir():
            print(f"[skip] {task.value}: no manifests under {task_dir}")
            continue
        for split in Difficulty:
            manifest_path = task_dir / f"{split.value}.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            rows = [builder(entry, source_root) for entry in manifest["samples"]]
            if not rows:
                continue
            feats = features_for_task(task)
            ds = Dataset.from_list(rows, features=feats)
            shard_dir = shard_root / task.value
            shard_dir.mkdir(parents=True, exist_ok=True)
            out_path = shard_dir / f"{split.value}-00000-of-00001.parquet"
            ds.to_parquet(str(out_path))
            print(f"[ok] wrote {out_path} ({len(ds)} rows)")
            task_splits.setdefault(task.value, []).append(split.value)

    _write_readme(out_dir, task_splits)
    print(f"[done] wrote README frontmatter to {out_dir / 'README.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Directory holding scenes/<scene>/<phase>/ modalities.",
    )
    parser.add_argument(
        "--manifests-root",
        required=True,
        type=Path,
        help="Directory holding <task>/<split>.json manifests.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory for the Parquet shards + README."
    )
    args = parser.parse_args()
    build(args.source_root, args.manifests_root, args.out)


if __name__ == "__main__":
    main()

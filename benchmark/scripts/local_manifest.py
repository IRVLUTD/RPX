"""Build a per-task per-split manifest locally from the HF-cached frames Parquet.

This is the bridging step that the dataset_hub writer side currently lacks:
``rpx_benchmark/hub.py:download_split`` expects ``manifests/<task>/<split>.json``
on the HF repo, but ``dataset_hub/`` only emits the all-frames Parquet
(``manifest/frames_v1.parquet``). Until that bridge ships in dataset_hub, this
script lives in ``scripts/`` so dev pipelines can run against the test mirror.

What it does
------------
1. Read ``manifest/frames_v1.parquet`` from a cached HF snapshot.
2. Filter rows to (a) the requested ``split`` and (b) frames that have every
   modality the task recipe requires.
3. Extract just those modality+frame pairs from the relevant tar shards into a
   flat ``extracted/scenes/<scene>/<phase>/<modality>/<frame>.png`` layout
   (one-time cost; cached across runs).
4. Write a manifest JSON consumable by ``RPXDataset.from_manifest``:

       {
         "task":   "monocular_depth",
         "split":  "easy",
         "root":   "<extracted_root>",
         "samples": [{"id": "...", "scene_id": "...", "phase": "0", "rgb": "...",
                      "depth": "...", "difficulty": "easy"}, ...]
       }

Once dataset_hub gains its own ``split_manifests`` module, the core logic in
``build_local_manifest`` can be lifted verbatim.

Usage
-----
    PYTHONPATH=. python scripts/local_manifest.py --task monocular_depth --split easy

    # In Python:
    from local_manifest import build_local_manifest
    path = build_local_manifest(task="monocular_depth", split="easy")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


# Map task → (required input modalities, ground-truth modality if any).
# Mirrors `MULTI_OBJECT_TASK_RECIPES` in `rpx_benchmark/dataset_hub/recipes.py`
# but expressed as plain strings so this script can run standalone.
TASK_MODALITIES = {
    "monocular_depth":  {"inputs": ["rgb"], "labels": ["depth"]},
    "rgbd_segmentation": {"inputs": ["rgb", "depth"], "labels": ["masks"]},
    "segmentation":      {"inputs": ["rgb"], "labels": ["masks"]},
    "relative_pose":     {"inputs": ["rgb"], "labels": ["cam_pose"]},
    "rgbd_relative_pose": {"inputs": ["rgb", "depth"], "labels": ["cam_pose"]},
    "object_tracking":   {"inputs": ["rgb"], "labels": ["masks"]},
}

# Path inside the tar where each modality's frames live.
# E.g. rgb.tar contains "rgb/00000.png", depth.tar contains "depth/00000.png",
# masks/v1.tar contains "masks/00000.png".
TAR_MEMBER_DIR = {
    "rgb": "rgb",
    "depth": "depth",
    "masks": "masks",
    "cam_pose": "cam_pose",
}


def _hf_snapshot_root(repo_id: str = "itaykadosh/rpx-test") -> Path:
    """Resolve the most recent local snapshot of an HF dataset cache."""
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = cache / f"datasets--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"no snapshots for {repo_id} under {cache}.",
            hint="Run `python -m rpx_benchmark.dataset_hub.cli download "
                 "--task <task> --split <split>` first to populate the HF cache.",
        )
    snaps = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snaps:
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"empty snapshots dir at {repo_dir}",
            hint="The HF cache exists but no snapshot has been downloaded. "
                 "Run download_for_task or rpx.load() to populate it.",
        )
    return snaps[0]


def _extract_member(tar_path: Path, member_name: str, out_path: Path) -> None:
    """Extract a single tar member to a target path. Idempotent."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        f = tf.extractfile(member_name)
        if f is None:
            from rpx_benchmark.exceptions import DatasetError
            raise DatasetError(
                f"{member_name} not in {tar_path}",
                hint="The cached tar shard is incomplete or its layout "
                     "doesn't match the parquet's frame_filename. Re-run "
                     "`dataset_hub.cli manifest` to regenerate.",
            )
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as g:
            g.write(f.read())
        tmp.rename(out_path)


def _modality_out_path(extracted_root: Path, scene: str, phase: int,
                        modality: str, frame_filename: str) -> Path:
    """Where each extracted frame lands locally."""
    return extracted_root / "scenes" / scene / str(phase) / modality / frame_filename


def _open_tars_once(snapshot_root: Path, df: pd.DataFrame,
                     modalities: Iterable[str]) -> dict[Path, tarfile.TarFile]:
    """Open every distinct tar exactly once. Caller closes."""
    handles: dict[Path, tarfile.TarFile] = {}
    cols = [f"shard_{m}" for m in modalities]
    for _, row in df.iterrows():
        for col in cols:
            tp = snapshot_root / row[col]
            if tp not in handles:
                handles[tp] = tarfile.open(tp, "r")
    return handles


@dataclass
class BuildResult:
    manifest_path: Path
    n_samples: int
    extracted_bytes: int
    extracted_root: Path


def build_local_manifest(
    *,
    task: str,
    split: str,
    repo_id: str = "itaykadosh/rpx-test",
    snapshot_root: Path | None = None,
    extracted_root: Path | None = None,
    max_samples: int | None = None,
) -> BuildResult:
    """Materialise a manifest the toolkit can consume.

    Parameters
    ----------
    task   one of TASK_MODALITIES keys (e.g. "monocular_depth").
    split  one of "easy" | "medium" | "hard".
    repo_id HuggingFace dataset (defaults to the dev test mirror).
    snapshot_root override the autodetected HF snapshot path.
    extracted_root  where to write extracted frames + the manifest JSON.
                    Defaults to ``<snapshot>/extracted/``.
    max_samples cap for smoke testing.
    """
    if task not in TASK_MODALITIES:
        from rpx_benchmark.exceptions import ConfigError
        raise ConfigError(
            f"unknown task {task!r}",
            hint=f"Known tasks: {sorted(TASK_MODALITIES)}.",
        )
    snap = snapshot_root or _hf_snapshot_root(repo_id)
    parquet = snap / "manifest" / "frames_v1.parquet"
    if not parquet.exists():
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"missing {parquet}",
            hint="Run download_for_task or rpx.load() first to populate the cache.",
        )

    df = pd.read_parquet(parquet)
    inputs = TASK_MODALITIES[task]["inputs"]
    labels = TASK_MODALITIES[task]["labels"]
    modalities = list(dict.fromkeys(inputs + labels))  # dedup, preserve order

    # Scene-wise splits (authoritative): every phase of a scene must share one
    # tier so STR / cross-phase comparisons aren't biased by tier reassignment.
    # The frames Parquet may carry per-(scene, phase) split values; we collapse
    # to per-scene by taking the mode (most common tier across that scene's
    # phases) and overwriting `split` accordingly.
    scene_split = df.groupby("scene_id")["split"].agg(
        lambda s: s.value_counts().idxmax()
    )
    df = df.drop(columns=["split"]).merge(
        scene_split.rename("split"), left_on="scene_id", right_index=True
    )

    # Filter: split match + every required modality present
    mask = df["split"].astype(str) == split
    for m in modalities:
        col = f"has_{m}"
        if col not in df.columns:
            from rpx_benchmark.exceptions import DatasetError
            raise DatasetError(
                f"frames Parquet has no column {col!r}",
                hint=f"The parquet at {parquet} is missing the modality "
                     f"presence column for {m!r}. Re-run "
                     "`dataset_hub.cli manifest` to regenerate.",
            )
        mask &= df[col].fillna(False).astype(bool)
    df = df[mask].reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
    if df.empty:
        from rpx_benchmark.exceptions import DatasetError
        raise DatasetError(
            f"no frames satisfy task={task} split={split} with all "
            f"modalities present in {parquet}",
            hint=f"Available splits in this parquet: "
                 f"{sorted(set(pd.read_parquet(parquet)['split'].dropna()))}. "
                 f"Required modalities for this task: {modalities}.",
        )

    extracted = extracted_root or (snap / "extracted")
    extracted.mkdir(parents=True, exist_ok=True)

    # Pre-open every tar exactly once for fast iteration
    log.info("[manifest] %d frames across %d scenes — extracting...",
             len(df), df["scene_id"].nunique())
    handles = _open_tars_once(snap, df, modalities)
    bytes_extracted = 0
    samples = []
    for _, row in df.iterrows():
        scene = str(row["scene_id"])
        phase = int(row["phase"])
        frame = str(row["frame_filename"])
        frame_stem = frame.removesuffix(".png")
        sample = {
            "id": f"{scene}__{phase}__{frame_stem}",
            "scene_id": scene,
            # int phase: loader._PHASE_INDEX maps {0: clutter, 1: interaction, 2: clean}
            "phase": int(phase),
            "frame_idx": int(row["frame_idx"]),
            "difficulty": split,
            # `metadata` rides through the loader untouched (loader.py:312) so
            # downstream model wrappers can resolve scene/phase/frame without
            # re-parsing the id. Avoids fragile id-string assumptions.
            "metadata": {
                "scene_id":  scene,
                "phase_idx": int(phase),
                "frame":     frame_stem,
            },
        }
        for m in modalities:
            tar_path = snap / row[f"shard_{m}"]
            member_name = f"{TAR_MEMBER_DIR.get(m, m)}/{frame}"
            out_path = _modality_out_path(extracted, scene, phase, m, frame)
            if not out_path.exists():
                tf = handles[tar_path]
                f = tf.extractfile(member_name)
                if f is None:
                    from rpx_benchmark.exceptions import DatasetError
                    raise DatasetError(
                        f"{member_name} not in {tar_path}",
                        hint="The cached tar shard is incomplete or its layout "
                             "doesn't match the parquet's frame_filename. Re-run "
                             "`dataset_hub.cli manifest` to regenerate.",
                    )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with tmp.open("wb") as g:
                    blob = f.read()
                    g.write(blob)
                    bytes_extracted += len(blob)
                tmp.rename(out_path)
            sample[m] = str(out_path.relative_to(extracted))
        samples.append(sample)
    for tf in handles.values():
        tf.close()

    manifest = {
        "task": task,
        "split": split,
        "root": str(extracted),
        "samples": samples,
    }
    out = extracted / "manifests" / task / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return BuildResult(
        manifest_path=out,
        n_samples=len(samples),
        extracted_bytes=bytes_extracted,
        extracted_root=extracted,
    )


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.0f} PB"


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--task", default="monocular_depth", choices=sorted(TASK_MODALITIES))
    ap.add_argument("--split", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--repo", default="itaykadosh/rpx-test")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap (smoke test)")
    ap.add_argument("--extracted-root", type=Path, default=None,
                    help="override extracted-files root (default: <snapshot>/extracted/)")
    args = ap.parse_args()
    try:
        result = build_local_manifest(
            task=args.task, split=args.split, repo_id=args.repo,
            extracted_root=args.extracted_root, max_samples=args.max_samples,
        )
    except Exception as e:  # noqa: BLE001
        sys.exit(f"local_manifest error: {type(e).__name__}: {e}")
    print(f"[manifest] wrote {result.manifest_path}")
    print(f"[manifest] samples: {result.n_samples}")
    print(f"[manifest] extracted: {_human(result.extracted_bytes)} new bytes")


if __name__ == "__main__":
    _cli()

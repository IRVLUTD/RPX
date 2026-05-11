#!/usr/bin/env python
"""Generate per-(task, difficulty) manifests from a local RPX scenes tree
and upload the whole dataset to a Hugging Face dataset repo.

Expected local layout (matches the on-disk format consumed by
``rpx_benchmark.loader.RPXDataset``)::

    <local-root>/
    ├── scenes/scene_000/
    │   ├── 0/                       # clutter
    │   │   ├── rgb/{frame}.png
    │   │   ├── depth/{frame}.png    # 16-bit mm
    │   │   ├── mask/{frame}.png     # integer instance IDs
    │   │   ├── pose/{frame}.npz
    │   │   ├── tracklets.json          (optional)
    │   │   ├── questionnaires.json     (optional)
    │   │   ├── spatial_qa.json         (optional)
    │   │   └── general_qa.json         (optional)
    │   ├── 1/ ...                  # interaction
    │   └── 2/ ...                  # clean
    ├── esd_scores.json                # [{scene, phase, difficulty, ...}]
    └── splits.json                    # {train: [...], val: [...], test: [...]}

Usage::

    python scripts/upload_to_hf.py \
        --local-root /data/rpx \
        --repo-id IRVLUTD/rpx-benchmark \
        --generate-manifests \
        --upload
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# The script is intentionally standalone (no rpx_benchmark import) so it can
# run from a clean checkout before the package is installed.

PHASE_NAMES = {"0": "clutter", "1": "interaction", "2": "clean"}

# Tasks the upload script can auto-generate manifests for from on-disk files.
# Pair-based tasks (relative_pose, NVS, keypoint_matching, sparse_depth) require
# a separate pair-selection step and are handled by a follow-up script.
SINGLE_FRAME_TASKS: Dict[str, Dict[str, Any]] = {
    "monocular_depth": {
        "required_modalities": ["rgb", "depth"],
        "sample_keys": ["rgb", "depth"],
    },
    "object_segmentation": {
        "required_modalities": ["rgb", "mask"],
        "sample_keys": ["rgb", "mask"],
    },
    "object_tracking": {
        "required_modalities": ["rgb", "mask"],
        "required_labels": ["tracklets.json"],
        "sample_keys": ["rgb", "mask"],
        "phase_label_keys": {"tracks": "tracklets.json"},
    },
    "visual_grounding": {
        "required_modalities": ["rgb"],
        "required_labels": ["spatial_qa.json"],
        "sample_keys": ["rgb"],
        "phase_label_keys": {"text_source": "spatial_qa.json"},
    },
}


# ------------------------------------------------------------------ #
# Local scanning
# ------------------------------------------------------------------ #


def load_esd_scores(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in data:
        scene = str(entry["scene"])
        phase = str(entry["phase"])
        out[(scene, phase)] = entry
    return out


def load_splits(path: Path | None) -> Dict[str, set]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items()}


def list_frames(phase_dir: Path, modality: str) -> List[str]:
    mod_dir = phase_dir / modality
    if not mod_dir.is_dir():
        return []
    return sorted(p.stem for p in mod_dir.iterdir() if p.is_file())


def scan_phase(scenes_root: Path, scene: str, phase: str) -> Dict[str, Any]:
    phase_dir = scenes_root / scene / phase
    info: Dict[str, Any] = {
        "scene": scene,
        "phase": phase,
        "phase_name": PHASE_NAMES.get(phase, phase),
        "present_modalities": set(),
        "present_labels": set(),
        "frames": [],
    }
    if not phase_dir.is_dir():
        return info
    for mod in ("rgb", "depth", "mask", "pose", "fisheye_left", "fisheye_right"):
        if (phase_dir / mod).is_dir():
            info["present_modalities"].add(mod)
    for label in ("tracklets.json", "questionnaires.json", "spatial_qa.json", "general_qa.json"):
        if (phase_dir / label).is_file():
            info["present_labels"].add(label)
    if "rgb" in info["present_modalities"]:
        info["frames"] = list_frames(phase_dir, "rgb")
    return info


def discover_phases(scenes_root: Path) -> List[Dict[str, Any]]:
    out = []
    for scene_dir in sorted(scenes_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for phase in ("0", "1", "2"):
            info = scan_phase(scenes_root, scene_dir.name, phase)
            if info["frames"]:
                out.append(info)
    return out


# ------------------------------------------------------------------ #
# Manifest construction
# ------------------------------------------------------------------ #


def _rel(scene: str, phase: str, modality: str, frame: str, ext: str) -> str:
    return f"scenes/{scene}/{phase}/{modality}/{frame}.{ext}"


def build_sample_entry(
    task: str,
    scene: str,
    phase: str,
    phase_name: str,
    frame: str,
    difficulty: str,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "id": f"{scene}_{phase_name}_{frame}",
        "scene": scene,
        "phase": phase_name,
        "difficulty": difficulty,
        "rgb": _rel(scene, phase, "rgb", frame, "png"),
    }
    task_cfg = SINGLE_FRAME_TASKS[task]
    if "depth" in task_cfg["sample_keys"]:
        entry["depth"] = _rel(scene, phase, "depth", frame, "png")
    if "mask" in task_cfg["sample_keys"]:
        entry["mask"] = _rel(scene, phase, "mask", frame, "png")
    if "pose" in task_cfg["sample_keys"]:
        entry["pose"] = _rel(scene, phase, "pose", frame, "npz")
    # Phase-level label files (one per phase, shared across frames).
    for key, fname in task_cfg.get("phase_label_keys", {}).items():
        entry[key] = f"scenes/{scene}/{phase}/{fname}"
    return entry


def phase_meets_task_requirements(phase_info: Dict[str, Any], task: str) -> bool:
    cfg = SINGLE_FRAME_TASKS[task]
    for m in cfg["required_modalities"]:
        if m not in phase_info["present_modalities"]:
            return False
    for lbl in cfg.get("required_labels", []):
        if lbl not in phase_info["present_labels"]:
            return False
    return True


def build_manifests(
    phases: List[Dict[str, Any]],
    esd: Dict[Tuple[str, str], Dict[str, Any]],
    splits: Dict[str, set],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return ``{(task, difficulty): manifest_dict}``."""
    manifests: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for task in SINGLE_FRAME_TASKS:
        buckets: Dict[str, Dict[str, Any]] = {
            d: {"task": task, "split": d, "scenes": [], "samples": []}
            for d in ("easy", "medium", "hard")
        }
        for info in phases:
            if not phase_meets_task_requirements(info, task):
                continue
            key = (info["scene"], info["phase"])
            esd_entry = esd.get(key)
            if esd_entry is None:
                continue
            difficulty = esd_entry["difficulty"]
            if difficulty not in buckets:
                continue
            # Optional scene-level train/val/test filter: skip scenes that
            # the user has marked for a non-evaluation split. For the
            # benchmark release we keep test scenes only in released
            # manifests; train/val stay available but are gated at consume
            # time. If no splits.json is provided, include everything.
            if splits and splits.get("test") and info["scene"] not in splits["test"]:
                continue
            bucket = buckets[difficulty]
            bucket["scenes"].append({"scene": info["scene"], "phase": info["phase"]})
            for frame in info["frames"]:
                bucket["samples"].append(
                    build_sample_entry(
                        task=task,
                        scene=info["scene"],
                        phase=info["phase"],
                        phase_name=info["phase_name"],
                        frame=frame,
                        difficulty=difficulty,
                    )
                )
        for difficulty, manifest in buckets.items():
            if manifest["samples"]:
                manifests[(task, difficulty)] = manifest
    return manifests


def write_manifests(
    manifests: Dict[Tuple[str, str], Dict[str, Any]],
    out_root: Path,
) -> List[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for (task, difficulty), data in manifests.items():
        out_path = out_root / task / f"{difficulty}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        written.append(out_path)
    return written


# ------------------------------------------------------------------ #
# Upload
# ------------------------------------------------------------------ #


def upload_to_hf(local_root: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print(
            "huggingface_hub is required for uploading. pip install 'huggingface_hub[hf_xet]'",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Prefer upload_large_folder for 75k files; fall back to upload_folder
    # on older huggingface_hub versions.
    uploader = getattr(api, "upload_large_folder", None)
    if uploader is not None:
        uploader(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_root),
            ignore_patterns=[".git/*", "__pycache__/*", "*.pyc"],
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_root),
            ignore_patterns=[".git/*", "__pycache__/*", "*.pyc"],
        )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def main() -> int:
    try:
        from rpx_benchmark.banner import show_banner

        show_banner(subtitle="scripts/upload_to_hf.py")
    except ImportError:
        pass  # rpx_benchmark not installed; run anyway

    script_dir = Path(__file__).resolve().parent
    default_card = script_dir.parent / "templates" / "hf_dataset_card.md"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root",
        required=True,
        type=Path,
        help="Root containing scenes/, esd_scores.json, splits.json",
    )
    parser.add_argument("--repo-id", default="IRVLUTD/rpx-benchmark")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--generate-manifests",
        action="store_true",
        help="Scan scenes/ and write manifests/ under local-root",
    )
    parser.add_argument("--upload", action="store_true", help="Push local-root to the HF repo")
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=default_card,
        help="Path to dataset card; copied to <local-root>/README.md",
    )
    args = parser.parse_args()

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "upload_to_hf — push RPX shards to HuggingFace",
        "manifests + tar shards + dataset card + Croissant metadata",
    )
    cli_ux.config(vars(args))

    local_root: Path = args.local_root
    scenes_root = local_root / "scenes"
    if not scenes_root.is_dir():
        print(f"error: {scenes_root} does not exist", file=sys.stderr)
        return 2

    if args.generate_manifests:
        esd_path = local_root / "esd_scores.json"
        if not esd_path.exists():
            print(f"error: {esd_path} not found — required for manifests", file=sys.stderr)
            return 2
        esd = load_esd_scores(esd_path)
        splits = load_splits(local_root / "splits.json")
        phases = discover_phases(scenes_root)
        print(f"scanned {len(phases)} (scene, phase) pairs")
        manifests = build_manifests(phases, esd, splits)
        written = write_manifests(manifests, local_root / "manifests")
        print(f"wrote {len(written)} manifests under {local_root / 'manifests'}")
        per_task: Dict[str, int] = defaultdict(int)
        for (task, _diff), m in manifests.items():
            per_task[task] += len(m["samples"])
        for task, n in sorted(per_task.items()):
            print(f"  {task:<22} {n} samples")

    if args.upload:
        if args.dataset_card and args.dataset_card.is_file():
            dest = local_root / "README.md"
            dest.write_text(args.dataset_card.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"copied dataset card → {dest}")
        print(f"uploading {local_root} → {args.repo_id}")
        upload_to_hf(local_root, args.repo_id, private=args.private)
        print("done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

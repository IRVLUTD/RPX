"""HuggingFace Hub integration for RPX benchmark.

Task-aware downloads: fetches only the modalities a given task needs,
reusing HF's content-addressed cache so switching tasks on the same
scenes only pulls the new label files.

Repo layout (on HF)::

    rpx-benchmark/
    ├── metadata/
    │   ├── scenes.parquet
    │   └── esd_scores.parquet
    ├── manifests/
    │   └── <task>/<difficulty>.json      # logical views, not duplicates
    └── scenes/scene_000/{0,1,2}/
        ├── rgb/*.png
        ├── depth/*.png                   # 16-bit mm
        ├── mask/*.png                    # integer instance IDs
        ├── pose/*.npz
        ├── tracklets.json
        ├── questionnaires.json
        ├── spatial_qa.json
        └── general_qa.json
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .api import Difficulty, TaskType
from .exceptions import DownloadError, ManifestError
from .loader import RPXDataset
from .logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_REPO_ID = os.environ.get("RPX_HF_REPO", "IRVLUTD/rpx-benchmark")
REPO_TYPE = "dataset"

# ------------------------------------------------------------------ #
# Task → modality mapping
# ------------------------------------------------------------------ #

# Modality globs, relative to a (scene, phase) directory on the hub.
RGB = "rgb/*"
DEPTH = "depth/*"
MASK = "mask/*"
POSE = "pose/*"
FISHEYE_L = "fisheye_left/*"
FISHEYE_R = "fisheye_right/*"
SPARSE_DEPTH_DIR = "sparse_depth/*"
KEYPOINTS_DIR = "keypoints/*"
TRACKLETS = "tracklets.json"
QUESTIONNAIRES = "questionnaires.json"
SPATIAL_QA = "spatial_qa.json"
GENERAL_QA = "general_qa.json"

TASK_MODALITIES: Dict[TaskType, List[str]] = {
    TaskType.MONOCULAR_DEPTH: [RGB, DEPTH],
    TaskType.SPARSE_DEPTH: [RGB, DEPTH, SPARSE_DEPTH_DIR],
    TaskType.OBJECT_SEGMENTATION: [RGB, MASK],
    TaskType.OBJECT_DETECTION: [RGB, MASK, TRACKLETS],
    TaskType.OPEN_VOCAB_DETECTION: [RGB, MASK, TRACKLETS, QUESTIONNAIRES],
    TaskType.OBJECT_TRACKING: [RGB, MASK, TRACKLETS],
    TaskType.RELATIVE_CAMERA_POSE: [RGB, POSE],
    TaskType.NOVEL_VIEW_SYNTHESIS: [RGB, DEPTH, POSE],
    TaskType.VISUAL_GROUNDING: [RGB, QUESTIONNAIRES, SPATIAL_QA],
    TaskType.KEYPOINT_MATCHING: [RGB, KEYPOINTS_DIR],
}

# QA-only tasks we may add later (not in TaskType enum yet); keep mapping
# for forward compatibility with the CLI.
EXTRA_TASK_ALIASES: Dict[str, List[str]] = {
    "qa_spatial": [RGB, DEPTH, SPATIAL_QA],
    "qa_general": [RGB, GENERAL_QA],
}


# ------------------------------------------------------------------ #
# Hub helpers
# ------------------------------------------------------------------ #


def _hub():
    """Lazy-import ``huggingface_hub`` and re-raise as :class:`DownloadError`.

    Raises
    ------
    DownloadError
        If ``huggingface_hub`` is not installed.
    """
    try:
        import huggingface_hub as hf
    except ImportError as e:
        raise DownloadError(
            "huggingface_hub is required for RPX hub operations.",
            hint="Install with: pip install 'rpx-benchmark[hub]'",
        ) from e
    return hf


def _rpx_cache_dir() -> Path:
    """Location for resolved-manifest files (outside the HF blob cache)."""
    base = os.environ.get("RPX_CACHE_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "rpx_benchmark"


def _manifest_repo_path(task: TaskType | str, split: Difficulty | str) -> str:
    task_name = task.value if isinstance(task, TaskType) else str(task)
    split_name = split.value if isinstance(split, Difficulty) else str(split)
    return f"manifests/{task_name}/{split_name}.json"


def _modalities_for(task: TaskType | str) -> List[str]:
    if isinstance(task, TaskType):
        return TASK_MODALITIES[task]
    if task in EXTRA_TASK_ALIASES:
        return EXTRA_TASK_ALIASES[task]
    return TASK_MODALITIES[TaskType(task)]


# ------------------------------------------------------------------ #
# Manifest handling
# ------------------------------------------------------------------ #


def fetch_manifest(
    task: TaskType | str,
    split: Difficulty | str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Dict[str, Any]:
    """Download and parse the task-level manifest for ``(task, split)``.

    Manifests are small (hundreds of KB) and are fetched eagerly so the
    caller can discover which (scene, phase) dirs the split references
    before kicking off a bulk download.

    Parameters
    ----------
    task : TaskType or str
    split : Difficulty or str
    repo_id : str
        HuggingFace dataset repo id. Defaults to
        :data:`DEFAULT_REPO_ID` (``"IRVLUTD/rpx-benchmark"``).
    cache_dir : str or Path, optional
    revision : str, optional

    Returns
    -------
    dict
        Parsed manifest JSON.

    Raises
    ------
    DownloadError
        If the download fails (network, auth, bad repo id) or the
        manifest file does not exist on the hub.
    ManifestError
        If the downloaded file is not valid JSON.
    """
    hf = _hub()
    repo_path = _manifest_repo_path(task, split)
    try:
        local = hf.hf_hub_download(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            filename=repo_path,
            cache_dir=str(cache_dir) if cache_dir else None,
            revision=revision,
        )
    except Exception as e:
        raise DownloadError(
            f"Failed to download manifest {repo_path!r} from {repo_id}: {e}",
            hint=(
                "Check your network connection and that the repo id is "
                "spelled correctly. Private repos need HF_TOKEN set."
            ),
        ) from e
    try:
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ManifestError(
            f"Manifest file at {local} is not valid JSON.",
        ) from e


def _extract_scene_phase_pairs(manifest: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """Unique (scene_dir, phase_dir) pairs referenced by a manifest."""
    pairs: Set[Tuple[str, str]] = set()
    if "scenes" in manifest:
        for entry in manifest["scenes"]:
            pairs.add((str(entry["scene"]), str(entry["phase"])))
        return pairs
    for sample in manifest.get("samples", []):
        for key in ("rgb", "depth", "mask"):
            p = sample.get(key)
            if not p:
                continue
            parts = Path(p).parts
            if len(parts) >= 3 and parts[0] == "scenes":
                pairs.add((parts[1], parts[2]))
                break
    return pairs


def _build_allow_patterns(
    modalities: Sequence[str],
    scene_phase_pairs: Iterable[Tuple[str, str]],
) -> List[str]:
    patterns: List[str] = []
    for scene, phase in sorted(scene_phase_pairs):
        base = f"scenes/{scene}/{phase}"
        for m in modalities:
            patterns.append(f"{base}/{m}")
    return patterns


# ------------------------------------------------------------------ #
# Public download API
# ------------------------------------------------------------------ #


def download_split(
    task: TaskType | str,
    split: Difficulty | str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    extra_modalities: Sequence[str] | None = None,
    max_workers: int = 8,
) -> Path:
    """Download only the files (task, split) needs, return resolved manifest path.

    The resolved manifest is a JSON file whose ``root`` field points to
    the local HF snapshot directory, so it can be fed directly to
    :meth:`RPXDataset.from_manifest`.
    """
    hf = _hub()
    task_enum = (
        TaskType(task) if isinstance(task, str) and task in TaskType._value2member_map_ else task
    )
    split_enum = Difficulty(split) if isinstance(split, str) else split

    manifest = fetch_manifest(task_enum, split_enum, repo_id, cache_dir, revision)

    pairs = _extract_scene_phase_pairs(manifest)
    if not pairs:
        raise ManifestError(
            f"Manifest {task}/{split} references no scenes; cannot derive download patterns.",
            hint="This usually means the manifest was generated against "
            "an empty scene list — re-run the upload script.",
        )

    modalities = list(_modalities_for(task_enum))
    if extra_modalities:
        modalities.extend(extra_modalities)

    allow_patterns = _build_allow_patterns(modalities, pairs)
    allow_patterns.append(_manifest_repo_path(task_enum, split_enum))

    log.info(
        "downloading %d file patterns for task=%s split=%s from %s",
        len(allow_patterns),
        task_enum.value if isinstance(task_enum, TaskType) else task_enum,
        split_enum.value if isinstance(split_enum, Difficulty) else split_enum,
        repo_id,
    )
    try:
        snapshot_root = hf.snapshot_download(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_dir) if cache_dir else None,
            revision=revision,
            max_workers=max_workers,
        )
    except Exception as e:
        raise DownloadError(
            f"snapshot_download failed for {repo_id}: {e}",
            hint="Rerun with --cache-dir pointing at a writable directory "
            "or set HF_HUB_OFFLINE=1 to use a prebuilt local cache.",
        ) from e

    # The HF tree ships tar shards; the per-task per-split JSON we just
    # downloaded references *extracted* PNG/NPZ paths. Run the extraction
    # step here so the user can hand the manifest to RPXDataset directly.
    # Idempotent — files already on disk are skipped.
    n_extracted, n_skipped = _extract_snapshot_tars(Path(snapshot_root))
    log.info(
        "extracted %d new files (%d already on disk) from tar shards",
        n_extracted,
        n_skipped,
    )

    resolved = dict(manifest)
    resolved["root"] = str(snapshot_root)
    resolved.setdefault(
        "task",
        task_enum.value if isinstance(task_enum, TaskType) else str(task_enum),
    )

    task_name = task_enum.value if isinstance(task_enum, TaskType) else str(task_enum)
    out_dir = _rpx_cache_dir() / repo_id.replace("/", "__") / "manifests" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"{split_enum.value if isinstance(split_enum, Difficulty) else split_enum}.json"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(resolved, f)
    return out_path


def load(
    task: TaskType | str,
    split: Difficulty | str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    batch_size: int = 1,
) -> RPXDataset:
    """Download (task, split) and return an iterable :class:`RPXDataset`.

    Incremental re-use::

        # First run: fetches rgb + depth for 'hard' scenes.
        depth_ds = rpx.load("monocular_depth", "hard")

        # Second run: rgb/depth already cached, only spatial_qa.json fetched.
        qa_ds    = rpx.load("visual_grounding", "hard")
    """
    manifest_path = download_split(
        task=task,
        split=split,
        repo_id=repo_id,
        cache_dir=cache_dir,
        revision=revision,
    )
    return RPXDataset.from_manifest(manifest_path, batch_size=batch_size)


# ------------------------------------------------------------------ #
# fsspec mount (preview / debug only — do not use for training)
# ------------------------------------------------------------------ #


def mount(repo_id: str = DEFAULT_REPO_ID):
    """Return an ``HfFileSystem`` rooted at the RPX repo for lazy browsing.

    Each read goes over the network; prefer :func:`load` for real workloads.
    """
    hf = _hub()
    fs = hf.HfFileSystem()
    return fs, f"datasets/{repo_id}"


# ------------------------------------------------------------------ #
# Tar extraction (post-download)
# ------------------------------------------------------------------ #


def _extract_snapshot_tars(snapshot_root: Path) -> Tuple[int, int]:
    """Extract every tar shard under ``snapshot_root`` into ``snapshot_root/extracted/``.

    The HF dataset tree ships tar shards (``scenes/<scene>/<phase>/rgb.tar``,
    ``scenes/<scene>/<phase>/labels/masks/v1.tar``, ...). Per-task per-split
    manifests reference *extracted* paths
    (``extracted/scenes/<scene>/<phase>/rgb/<frame>.png``,
     ``extracted/scenes/<scene>/<phase>/sam2/masks/<frame>.png``, ...). This
    helper materialises that layout post-download.

    Idempotency
    -----------
    For each tar member, we check if the destination file already exists
    (correct size). Only missing or wrong-size files are extracted. The
    write is atomic-ish: extract to ``<dest>.part`` then ``rename`` so a
    half-written file never collides with a re-run.

    Path mapping
    ------------
    Tar location ``scenes/<scene>/<phase>/<...>.tar`` → extraction root
    ``extracted/scenes/<scene>/<phase>/`` (with the tar's *own* member
    names appended). Examples:

    * ``scenes/scene1/0/rgb.tar``               members ``rgb/00000.png``  → ``extracted/scenes/scene1/0/rgb/00000.png``
    * ``scenes/scene1/0/labels/masks/v1.tar``   members ``sam2/masks/00000.png`` → ``extracted/scenes/scene1/0/sam2/masks/00000.png``
    * ``scenes/scene1/0/labels/cam_pose/v1.tar`` members ``cam_pose/00000.npz`` → ``extracted/scenes/scene1/0/cam_pose/00000.npz``

    Parameters
    ----------
    snapshot_root
        Root of an HF snapshot (the path returned by ``snapshot_download``).

    Returns
    -------
    (n_extracted, n_skipped)
        Number of files newly extracted and number already present.
    """
    extracted_root = snapshot_root / "extracted"
    n_new = 0
    n_skip = 0
    for tar_path in sorted(snapshot_root.rglob("*.tar")):
        if extracted_root in tar_path.parents:
            # Don't re-process artefacts already under extracted/ from a
            # previous run that happened to leave tars behind.
            continue
        try:
            rel = tar_path.relative_to(snapshot_root)
        except ValueError:  # symlinks pointing outside; skip
            continue
        # Locate the (scene, phase) prefix: the parts up to and including
        # the first numeric component (the phase index 0/1/2). Example:
        # ('scenes', 'scene1', '0', 'rgb.tar')         → 'scenes/scene1/0'
        # ('scenes', 'scene1', '0', 'labels', 'masks', 'v1.tar') → same.
        prefix_parts: List[str] = []
        for p in rel.parts[:-1]:  # stop before the .tar filename
            prefix_parts.append(p)
            if p.isdigit():
                break
        if not prefix_parts or not prefix_parts[-1].isdigit():
            # Not a per-scene-per-phase shard (e.g. an unrelated tar at
            # repo root); skip rather than guess.
            continue
        out_base = extracted_root.joinpath(*prefix_parts)
        out_base.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    out = out_base / member.name
                    if out.exists() and out.stat().st_size == member.size:
                        n_skip += 1
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(out.suffix + ".part")
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    with tmp.open("wb") as g:
                        # Stream-copy in chunks; member.size can be ~MB
                        # for masks, no point loading whole thing in RAM.
                        while True:
                            chunk = f.read(1 << 20)
                            if not chunk:
                                break
                            g.write(chunk)
                    tmp.rename(out)
                    n_new += 1
        except tarfile.TarError as e:
            log.warning("failed to extract %s: %s", tar_path, e)
            continue
    return n_new, n_skip

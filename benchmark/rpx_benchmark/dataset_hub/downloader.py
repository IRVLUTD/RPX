"""Selective download from the RPX HuggingFace dataset repo.

Resolves a ``(task, split, modalities, label_versions)`` request into a
list of glob patterns and hands them to ``snapshot_download``. The HF
cache is content-addressed by file hash, so re-pulling a previously
fetched modality is a no-op — that's how delta downloads work for free
when the team switches tasks on already-cached scenes.

Wire-up::

    pull = download_for_task(
        task="segmentation", split="easy", repo_id="IRVLUTD/RPX",
    )
    pull.local_dir       # local snapshot dir, ready to feed to a loader
    pull.manifest_table  # pyarrow.Table — per-frame metadata for the slice
    pull.bytes_fetched   # ≈ what the network actually moved (cache hits subtract)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
)

from ..exceptions import ConfigError, DownloadError
from ..logging_utils import get_logger
from .manifest import SCHEMA_VERSION
from .recipes import (
    DEFAULT_REPO_ID,
    SceneType,
    TaskRecipe,
    resolve_recipe,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    import pyarrow as pa


log = get_logger(__name__)


# Modalities the packer treats as raw — no version suffix in the repo path.
_RAW_MODALITIES = frozenset({"rgb", "depth", "fisheye", "cam_pose"})


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one selective pull."""

    repo_id: str
    revision: Optional[str]
    task: str
    split: Optional[str]
    scene_type: SceneType
    local_dir: Path
    allow_patterns: List[str]
    matched_scenes: List[str]
    bytes_fetched: int
    files_fetched: int
    cache_hits: int = 0
    manifest_table: Optional["pa.Table"] = None


def _hub():
    try:
        import huggingface_hub as hf  # noqa: WPS433
    except ImportError as e:
        raise DownloadError(
            "huggingface_hub is required for RPX downloads.",
            hint="Install with: pip install 'rpx-benchmark[hub]'",
        ) from e
    return hf


def _arrow():
    try:
        import pyarrow as pa  # noqa: WPS433
        import pyarrow.parquet as pq  # noqa: WPS433
    except ImportError as e:
        raise DownloadError(
            "pyarrow is required to read the dataset manifest.",
            hint="Install with: pip install 'rpx-benchmark[hub]'",
        ) from e
    return pa, pq


# --------------------------------------------------------------------- #
# Manifest helpers (pulled and read first; small file, always cheap)
# --------------------------------------------------------------------- #

def _fetch_manifest(
    hf, repo_id: str, revision: Optional[str], cache_dir: Optional[Path],
) -> tuple[Path, Path]:
    """Download just ``manifest/{frames_v1.parquet, current.json}``."""
    parquet_path = hf.hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename=f"manifest/frames_{SCHEMA_VERSION}.parquet",
        revision=revision, cache_dir=str(cache_dir) if cache_dir else None,
    )
    current_path = hf.hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename="manifest/current.json",
        revision=revision, cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(parquet_path), Path(current_path)


def _filter_manifest(
    table: "pa.Table",
    scene_type: SceneType,
    split: Optional[str],
) -> "pa.Table":
    pa, _ = _arrow()
    import pyarrow.compute as pc  # noqa: WPS433

    mask = pc.equal(table["scene_type"], pa.scalar(scene_type.value))
    if split is not None and scene_type is SceneType.MULTI_OBJECT:
        mask = pc.and_(mask, pc.equal(table["split"], pa.scalar(split)))
    return table.filter(mask)


# --------------------------------------------------------------------- #
# Pattern resolution
# --------------------------------------------------------------------- #

def _resolve_label_version(
    modality: str, current: Mapping[str, Any], explicit: Mapping[str, str],
) -> str:
    """Pick the label version for ``modality``: explicit > current.json > 'v1'."""
    if modality in explicit:
        return explicit[modality]
    versions = current.get("label_versions", {})
    return versions.get(modality, "v1")


def _scene_root_for(scene_type: SceneType) -> str:
    return "scenes" if scene_type is SceneType.MULTI_OBJECT else "objects"


def _build_allow_patterns(
    matched_scenes: Iterable[str],
    scene_type: SceneType,
    modalities: Iterable[str],
    current: Mapping[str, Any],
    label_versions: Mapping[str, str],
) -> List[str]:
    """One glob per (scene, modality)."""
    root = _scene_root_for(scene_type)
    patterns: List[str] = []
    for scene in sorted(matched_scenes):
        for m in sorted(modalities):
            if m in _RAW_MODALITIES:
                patterns.append(f"{root}/{scene}/*/{m}.tar")
            else:
                v = _resolve_label_version(m, current, label_versions)
                patterns.append(f"{root}/{scene}/*/labels/{m}/{v}.tar")
    return patterns


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #

def download_for_task(
    task: str,
    split: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    extra_modalities: Sequence[str] = (),
    label_versions: Optional[Mapping[str, str]] = None,
    scene_type: Optional[SceneType] = None,
    max_workers: int = 8,
    return_manifest: bool = True,
) -> DownloadResult:
    """Pull just the tar shards needed to run ``task`` on ``split``.

    Parameters
    ----------
    task : str
        Recipe key from :data:`MULTI_OBJECT_TASK_RECIPES` or
        :data:`SINGLE_OBJECT_TASK_RECIPES`.
    split : str, optional
        ``"easy" | "medium" | "hard"``. Required for multi-object tasks,
        forbidden for single-object tasks (which have no splits).
    repo_id : str
        HuggingFace dataset repo (default: ``IRVLUTD/RPX``).
    extra_modalities : Sequence[str]
        Add modalities beyond the recipe's defaults (e.g. ``["masks_aux"]``
        if you want the visualisation overlays alongside ``masks``).
    label_versions : Mapping[str, str], optional
        Override which label version to pull for one or more modalities.
        Default behaviour is to follow the repo's ``manifest/current.json``.
    scene_type : SceneType, optional
        Disambiguates when a task name exists in both recipe tables.

    Returns
    -------
    DownloadResult
        ``local_dir`` is the local snapshot root; combine with
        ``allow_patterns`` and ``manifest_table`` to feed a downstream
        loader. ``bytes_fetched`` reports the size on disk after the pull
        (for true delta accounting prefer your own ``du -sh``).
    """
    hf = _hub()
    recipe: TaskRecipe = resolve_recipe(task, scene_type=scene_type)

    if recipe.scene_type is SceneType.SINGLE_OBJECT and split is not None:
        raise ConfigError(
            f"task {task!r} targets single-object scenes, which have no splits.",
            hint="Drop the --split argument for single-object recipes.",
        )
    if recipe.scene_type is SceneType.MULTI_OBJECT and split is None:
        raise ConfigError(
            f"task {task!r} targets multi-object scenes — pick a split.",
            hint="Pass split='easy' | 'medium' | 'hard'.",
        )

    parquet_path, current_path = _fetch_manifest(
        hf, repo_id, revision, cache_dir,
    )
    current = json.loads(current_path.read_text(encoding="utf-8"))
    _, pq = _arrow()
    table = pq.read_table(parquet_path)
    sliced = _filter_manifest(table, recipe.scene_type, split)

    matched_scenes: Set[str] = set(sliced["scene_id"].to_pylist())
    if not matched_scenes:
        raise DownloadError(
            f"No scenes matched task={task!r} split={split!r} on {repo_id}.",
            hint=(
                "Verify the manifest has rows for this scene_type/split. "
                "If the dataset was uploaded without a `splits=...` argument "
                "to build_frame_manifest, multi-object rows will all have "
                "split=null."
            ),
        )

    modalities = set(recipe.all_modalities()) | set(extra_modalities)
    explicit = dict(label_versions or {})
    patterns = _build_allow_patterns(
        matched_scenes, recipe.scene_type, modalities, current, explicit,
    )

    log.info("download plan: task=%s split=%s scenes=%d modalities=%s "
              "patterns=%d", task, split, len(matched_scenes),
              sorted(modalities), len(patterns))

    try:
        local_dir = hf.snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            allow_patterns=patterns,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            max_workers=max_workers,
        )
    except Exception as e:
        raise DownloadError(
            f"snapshot_download failed for {repo_id}: {e}",
            hint=(
                "Re-run to retry; HF caches partial downloads. If auth is the "
                "issue, run `hf auth login` and confirm read access."
            ),
        ) from e

    local = Path(local_dir)
    files = [p for p in local.rglob("*") if p.is_file()]
    bytes_ = sum(p.stat().st_size for p in files)

    return DownloadResult(
        repo_id=repo_id, revision=revision,
        task=task, split=split, scene_type=recipe.scene_type,
        local_dir=local,
        allow_patterns=patterns,
        matched_scenes=sorted(matched_scenes),
        bytes_fetched=bytes_, files_fetched=len(files),
        cache_hits=0,                 # HF doesn't expose per-call hit count
        manifest_table=sliced if return_manifest else None,
    )

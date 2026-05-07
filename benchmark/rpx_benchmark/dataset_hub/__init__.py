"""RPX dataset hub — selective HuggingFace download/upload pipeline.

Layered on top of :mod:`rpx_benchmark.hub` (which already implements
modality-aware ``snapshot_download``) and adds:

  * :mod:`.recipes`   — task → modality recipes (multi-object + single-object).
  * :mod:`.mock`      — synthetic capture-tree generator for tests / dry runs.
  * :mod:`.scanner`   — walks the on-disk capture layout, reports inventory.
  * :mod:`.packer`    — converts a capture tree into per-modality tar shards.
  * :mod:`.manifest`  — writes the per-frame Parquet manifest the hub ships.
  * :mod:`.uploader`  — pushes a packed staging dir to a HF dataset repo.
  * :mod:`.cli`       — ``python -m rpx_benchmark.dataset_hub.cli`` entry points.

End-to-end usage::

    src   = Path("/path/to/test_dataset_aggregated")
    stage = Path("/tmp/rpx_staging")
    scan  = scan_capture_root(src)
    pack  = pack_capture_tree(PackPlan(src, stage), scan)
    build_frame_manifest(scan, pack, stage, splits=splits_dict)
    upload_staging(UploadPlan(stage, repo_id="IRVLUTD/RPX"))
"""

from __future__ import annotations

from .croissant import (
    CroissantPatch,
    stage_croissant,
)
from .dataset_card import (
    CardSpec,
    write_dataset_card,
)
from .downloader import (
    DownloadResult,
    download_for_task,
)
from .manifest import (
    SCHEMA_VERSION,
    ManifestPaths,
    build_frame_manifest,
    read_frame_manifest,
)
from .packer import (
    RAW_MODALITIES,
    PackedShard,
    PackPlan,
    PackResult,
    SharedArtefact,
    pack_capture_tree,
    pack_objects_meta,
)
from .recipes import (
    DEFAULT_REPO_ID,
    MULTI_OBJECT_TASK_RECIPES,
    SINGLE_OBJECT_TASK_RECIPES,
    SceneType,
    TaskRecipe,
    resolve_recipe,
)
from .scanner import (
    ModalityInventory,
    PhaseInventory,
    ScanResult,
    SceneInventory,
    scan_capture_root,
)
from .split_manifests import (
    write_split_manifests,
)
from .staging import (
    SCENE_SPLITS_JSON,
    SPLIT_TIER_FILES,
    StagedFile,
    load_scene_splits,
    stage_paths,
    stage_splits,
)
from .uploader import (
    DEFAULT_IGNORE_PATTERNS,
    UploadPlan,
    UploadResult,
    upload_paths,
    upload_staging,
)

__all__ = [
    # recipes
    "DEFAULT_REPO_ID",
    "MULTI_OBJECT_TASK_RECIPES",
    "SINGLE_OBJECT_TASK_RECIPES",
    "SceneType",
    "TaskRecipe",
    "resolve_recipe",
    # scanner
    "ModalityInventory",
    "PhaseInventory",
    "SceneInventory",
    "ScanResult",
    "scan_capture_root",
    # packer
    "RAW_MODALITIES",
    "PackPlan",
    "PackResult",
    "PackedShard",
    "SharedArtefact",
    "pack_capture_tree",
    "pack_objects_meta",
    # manifest
    "SCHEMA_VERSION",
    "ManifestPaths",
    "build_frame_manifest",
    "read_frame_manifest",
    # split manifests (per-task, per-split)
    "write_split_manifests",
    # uploader
    "DEFAULT_IGNORE_PATTERNS",
    "UploadPlan",
    "UploadResult",
    "upload_paths",
    "upload_staging",
    # croissant
    "CroissantPatch",
    "stage_croissant",
    # dataset card
    "CardSpec",
    "write_dataset_card",
    # downloader
    "DownloadResult",
    "download_for_task",
    # staging
    "SPLIT_TIER_FILES",
    "SCENE_SPLITS_JSON",
    "StagedFile",
    "load_scene_splits",
    "stage_paths",
    "stage_splits",
]

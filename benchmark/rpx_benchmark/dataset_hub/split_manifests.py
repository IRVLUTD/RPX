"""Per-task, per-split manifest writer for the dataset hub.

The toolkit's :func:`rpx_benchmark.hub.download_split` expects a JSON file at
``manifests/<recipe_key>/<split>.json`` on the published HF repo. The hub's
existing ``manifest.py`` only emits the all-frames Parquet
(``manifest/frames_v1.parquet``); this module fills the gap so a freshly
published HF release works for every task the loader supports.

Per-task entry contracts
------------------------
Each TaskType the loader can parse expects a specific set of keys on each
manifest entry. Using the wrong key (``cam_pose`` instead of ``pose``,
``masks`` instead of ``mask``, ``fisheye`` instead of
``fisheye_left``/``fisheye_right``) is a silent 404 at load time.

This module owns the per-task contract:

==================  ===================  ============================================
Recipe key          TaskType.value       Required entry keys + GT keys
==================  ===================  ============================================
monocular_depth     monocular_depth      rgb, depth
segmentation        object_segmentation  rgb, mask
rgbd_segmentation   object_segmentation  rgb, depth, mask
stereo_depth        monocular_depth      fisheye_left, fisheye_right, depth
relative_pose       relative_camera_pose rgb, rgb_b, pose_a, pose_b           [paired]
rgbd_relative_pose  relative_camera_pose rgb, depth, rgb_b, depth_b,
                                          pose_a, pose_b                       [paired]
object_tracking     object_tracking      rgb, mask, tracks                    [per-phase]
vqa                 visual_grounding     rgb, text                            [needs labels]
==================  ===================  ============================================

Modality file layout (where the actual files land after extraction)
-------------------------------------------------------------------
On-disk:

==========  ============================================  ============
Modality    Subdir under ``extracted/scenes/<s>/<p>/``    Suffix
==========  ============================================  ============
rgb         ``rgb/``                                       ``.png``
depth       ``depth/``                                     ``.png`` (16-bit mm)
fisheye     ``fisheye/{left,right}/``                      ``.png``
masks       ``sam2/masks/``                                ``.png``
cam_pose    ``cam_pose/``                                  ``.npz``
==========  ============================================  ============

Splits — scene-wise, mode-aggregated
------------------------------------
Splits are scene-wise (every phase of a scene shares one tier) so STR /
cross-phase comparisons aren't biased by tier reassignment.

Usage
-----
::

    from rpx_benchmark.dataset_hub.split_manifests import write_split_manifests
    written = write_split_manifests("/path/to/staging")
    # → {(recipe_key, split): manifest_path, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .recipes import MULTI_OBJECT_TASK_RECIPES, TaskRecipe

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Per-modality on-disk layout (after extraction)
# ---------------------------------------------------------------------- #
#
# (parquet column suffix, on-disk subdir, file extension)
#
# The parquet's ``has_<m>`` / ``shard_<m>`` columns use the column suffix.
# The on-disk subdir is what ``hub._extract_snapshot_tars`` materialises
# under ``extracted/scenes/<scene>/<phase>/``.  The file extension is
# what each frame ends in once extracted (rgb/depth → .png from the
# 16-bit mm, cam_pose → .npz containing the SE(3) matrix).

_MODALITY_LAYOUT: Dict[str, Tuple[str, str, str]] = {
    # modality_name → (parquet_col_suffix, extracted_subdir, ext)
    "rgb":          ("rgb",       "rgb",            ".png"),
    "depth":        ("depth",     "depth",          ".png"),
    "fisheye":      ("fisheye",   "fisheye",        ".png"),     # see note for left/right
    "fisheye_left": ("fisheye",   "fisheye/left",   ".png"),
    "fisheye_right":("fisheye",   "fisheye/right",  ".png"),
    "masks":        ("masks",     "sam2/masks",     ".png"),
    "cam_pose":     ("cam_pose",  "cam_pose",       ".npz"),
}


def _modality_path(scene: str, phase: int, modality: str, frame_stem: str) -> str:
    """Build the manifest's path string for one (scene, phase, modality, frame).

    Path is relative to the manifest's ``root`` (set by ``download_split``
    to the snapshot root). After ``_extract_snapshot_tars`` runs, the
    file lives under ``extracted/scenes/...``.
    """
    if modality not in _MODALITY_LAYOUT:
        from ..exceptions import ConfigError
        raise ConfigError(
            f"unknown modality {modality!r}",
            hint=f"Known modalities: {', '.join(sorted(_MODALITY_LAYOUT))}",
        )
    _, subdir, ext = _MODALITY_LAYOUT[modality]
    return f"extracted/scenes/{scene}/{phase}/{subdir}/{frame_stem}{ext}"


def _has_col(modality: str) -> str:
    """Return the parquet ``has_<col>`` column name for a modality."""
    col_suffix, _, _ = _MODALITY_LAYOUT[modality]
    return f"has_{col_suffix}"


# ---------------------------------------------------------------------- #
# Per-task spec — links recipe keys → TaskType value + entry-builder
# ---------------------------------------------------------------------- #

class _TaskSpec:
    """How to build manifest entries for one recipe key.

    Subclasses override ``build_entries`` to return a list of dicts
    matching the loader's expected entry shape for that task.
    """

    #: The string the loader's ``TaskType()`` constructor needs to see.
    task_type_value: str = ""

    #: Required modalities, by parquet column suffix. Used to filter the
    #: parquet rows (a row enters the manifest only if every required
    #: modality is present).
    required_modalities: Tuple[str, ...] = ()

    def build_entries(self, df) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _SingleFrameDepthSpec(_TaskSpec):
    """``monocular_depth``: rgb + depth GT, one entry per frame."""
    task_type_value = "monocular_depth"
    required_modalities = ("rgb", "depth")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        return [_single_frame_entry(row, ["rgb", "depth"]) for _, row in df.iterrows()]


class _SegmentationSpec(_TaskSpec):
    """``segmentation``: rgb + mask (singular!) GT.

    Loader's ``_load_ground_truth`` does ``self._load_mask(entry["mask"])``;
    the entry key is ``mask`` (singular) but the modality directory is
    ``sam2/masks``. We rename the entry key explicitly here.
    """
    task_type_value = "object_segmentation"
    required_modalities = ("rgb", "masks")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            e = _base_entry(row)
            scene = e["scene_id"]
            phase = e["phase"]
            stem = _frame_stem(row)
            e["rgb"]  = _modality_path(scene, phase, "rgb",   stem)
            e["mask"] = _modality_path(scene, phase, "masks", stem)
            rows.append(e)
        return rows


class _RGBDSegmentationSpec(_TaskSpec):
    """``rgbd_segmentation``: rgb + depth + mask. Loader's task is still
    OBJECT_SEGMENTATION (depth is an extra input the model sees, not GT)."""
    task_type_value = "object_segmentation"
    required_modalities = ("rgb", "depth", "masks")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            e = _base_entry(row)
            scene = e["scene_id"]
            phase = e["phase"]
            stem = _frame_stem(row)
            e["rgb"]   = _modality_path(scene, phase, "rgb",   stem)
            e["depth"] = _modality_path(scene, phase, "depth", stem)
            e["mask"]  = _modality_path(scene, phase, "masks", stem)
            rows.append(e)
        return rows


class _StereoDepthSpec(_TaskSpec):
    """``stereo_depth``: fisheye_left + fisheye_right inputs, depth GT.

    Loader's TaskType is MONOCULAR_DEPTH (depth GT is the same;
    stereo-vs-mono is an input-side distinction).
    """
    task_type_value = "monocular_depth"
    required_modalities = ("rgb", "depth", "fisheye")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            e = _base_entry(row)
            scene = e["scene_id"]
            phase = e["phase"]
            stem = _frame_stem(row)
            e["rgb"]            = _modality_path(scene, phase, "rgb",           stem)
            e["depth"]          = _modality_path(scene, phase, "depth",         stem)
            e["fisheye_left"]   = _modality_path(scene, phase, "fisheye_left",  stem)
            e["fisheye_right"]  = _modality_path(scene, phase, "fisheye_right", stem)
            rows.append(e)
        return rows


class _RelativePoseSpec(_TaskSpec):
    """``relative_pose``: paired-frame task.

    Each entry pairs frame N (`rgb`, `pose_a`) with frame N+stride
    (`rgb_b`, `pose_b`) within the same scene+phase. Stride defaults to 5
    frames (~0.5 s at 10 Hz capture; matches existing
    ``scripts/generate_keypoint_pairs.py`` convention).
    """
    task_type_value = "relative_camera_pose"
    required_modalities = ("rgb", "cam_pose")
    pair_stride: int = 5

    def build_entries(self, df) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        # Group rows by (scene, phase) — pairs only within a phase.
        group_iter = df.groupby(["scene_id", "phase"], sort=False)
        for (scene, phase), grp in group_iter:
            grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
            n = len(grp_sorted)
            for i in range(n - self.pair_stride):
                row_a = grp_sorted.iloc[i]
                row_b = grp_sorted.iloc[i + self.pair_stride]
                stem_a = _frame_stem(row_a)
                stem_b = _frame_stem(row_b)
                e: Dict[str, Any] = {
                    "id":         f"{scene}__{phase}__{stem_a}__{stem_b}",
                    "scene_id":   scene,
                    "phase":      int(phase),
                    "frame_idx":  int(row_a["frame_idx"]),
                    "frame_idx_b":int(row_b["frame_idx"]),
                    "difficulty": str(row_a["split"]),
                    "metadata": {
                        "scene_id":  scene,
                        "phase_idx": int(phase),
                        "frame":     stem_a,
                        "frame_b":   stem_b,
                        "pair_stride": self.pair_stride,
                    },
                    "rgb":    _modality_path(scene, int(phase), "rgb",      stem_a),
                    "rgb_b":  _modality_path(scene, int(phase), "rgb",      stem_b),
                    "pose_a": _modality_path(scene, int(phase), "cam_pose", stem_a),
                    "pose_b": _modality_path(scene, int(phase), "cam_pose", stem_b),
                }
                rows.append(e)
        return rows


class _RGBDRelativePoseSpec(_RelativePoseSpec):
    """``rgbd_relative_pose``: paired with depth on both frames."""
    required_modalities = ("rgb", "depth", "cam_pose")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        rows = super().build_entries(df)
        # Augment each paired entry with depth_a / depth_b.
        for e in rows:
            scene = e["scene_id"]
            phase = e["phase"]
            stem_a = e["metadata"]["frame"]
            stem_b = e["metadata"]["frame_b"]
            e["depth"]   = _modality_path(scene, phase, "depth", stem_a)
            e["depth_b"] = _modality_path(scene, phase, "depth", stem_b)
        return rows


class _ObjectTrackingSpec(_TaskSpec):
    """``object_tracking``: rgb + mask + tracks.

    The loader's ``_load_tracklets`` reads ``entry["tracks"]`` as a JSON
    file containing aggregated tracklets for one (scene, phase). That
    JSON is currently produced by external tooling (SAM2 video output);
    we reference its expected location and let the publisher script
    generate it. If the tracklets JSON isn't on disk, this manifest's
    samples will fail to load — but that's a publishing-side gap, not
    a manifest-shape bug.
    """
    task_type_value = "object_tracking"
    required_modalities = ("rgb", "masks")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            e = _base_entry(row)
            scene = e["scene_id"]
            phase = e["phase"]
            stem = _frame_stem(row)
            e["rgb"]    = _modality_path(scene, phase, "rgb",   stem)
            e["mask"]   = _modality_path(scene, phase, "masks", stem)
            # Per-(scene, phase) aggregated tracklets JSON. Convention:
            # ``extracted/scenes/<scene>/<phase>/tracklets/v1.json``.
            e["tracks"] = f"extracted/scenes/{scene}/{phase}/tracklets/v1.json"
            rows.append(e)
        return rows


class _VQASpec(_TaskSpec):
    """``vqa``: not yet wired — needs spatial_qa / questionnaire labels.

    Returns no entries until the team's VQA generation pipeline lands
    (tracked separately in `paper-submission/neurips-2026/VQA_DESIGN.md`).
    Logging an explicit warning so this is visible in the upload run.
    """
    task_type_value = "visual_grounding"
    required_modalities = ("rgb",)

    def build_entries(self, df) -> List[Dict[str, Any]]:
        log.warning(
            "vqa: spec exists but no entries produced — needs the team's "
            "VQA label generation pipeline (see VQA_DESIGN.md). Skipping."
        )
        return []


# Recipe-key → spec instance.
_TASK_SPECS: Dict[str, _TaskSpec] = {
    "monocular_depth":     _SingleFrameDepthSpec(),
    "segmentation":        _SegmentationSpec(),
    "rgbd_segmentation":   _RGBDSegmentationSpec(),
    "stereo_depth":        _StereoDepthSpec(),
    "relative_pose":       _RelativePoseSpec(),
    "rgbd_relative_pose":  _RGBDRelativePoseSpec(),
    "object_tracking":     _ObjectTrackingSpec(),
    "vqa":                 _VQASpec(),
}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

_SPLITS: Tuple[str, ...] = ("easy", "medium", "hard")


def _frame_stem(row) -> str:
    """Return the frame's stem (no extension)."""
    return Path(str(row["frame_filename"])).stem


def _base_entry(row) -> Dict[str, Any]:
    """Common per-row fields: id, scene_id, phase, frame_idx, difficulty,
    metadata. Task-specific specs add modality paths on top."""
    scene = str(row["scene_id"])
    phase = int(row["phase"])
    stem = _frame_stem(row)
    return {
        "id":         f"{scene}__{phase}__{stem}",
        "scene_id":   scene,
        "phase":      phase,
        "frame_idx":  int(row["frame_idx"]),
        "difficulty": str(row["split"]),
        "metadata": {
            "scene_id":  scene,
            "phase_idx": phase,
            "frame":     stem,
        },
    }


def _single_frame_entry(row, modalities: Iterable[str]) -> Dict[str, Any]:
    """Build a single-frame entry where each modality's *entry key* is the
    same as its *modality name* (only correct for monocular_depth where
    the loader's keys are literally ``rgb`` + ``depth``)."""
    e = _base_entry(row)
    scene = e["scene_id"]
    phase = e["phase"]
    stem = e["metadata"]["frame"]
    for m in modalities:
        e[m] = _modality_path(scene, phase, m, stem)
    return e


def _parquet_path(staging_root: Path) -> Path:
    cand = staging_root / "manifest" / "frames_v1.parquet"
    if cand.is_file():
        return cand
    matches = list((staging_root / "manifest").glob("frames_*.parquet"))
    if not matches:
        from ..exceptions import DatasetError
        raise DatasetError(
            f"no frames Parquet under {staging_root}/manifest/",
            hint="Run `dataset_hub.cli manifest` first to build the parquet "
                 "from your captures.",
        )
    return matches[-1]


def _scene_wise_splits(df):
    """Mode-aggregate per-row split into per-scene tier."""
    return df.groupby("scene_id")["split"].agg(
        lambda s: s.dropna().value_counts().idxmax() if s.notna().any() else None
    )


def _filter_required(df, required_cols: Iterable[str]):
    """Return rows that have every required modality column set to True."""
    sub = df
    for col in required_cols:
        if col not in sub.columns:
            return None
        sub = sub[sub[col].fillna(False).astype(bool)]
    return sub


# ---------------------------------------------------------------------- #
# Public entry point
# ---------------------------------------------------------------------- #

def write_split_manifests(
    staging_root: str | Path,
    *,
    tasks: Optional[Iterable[str]] = None,
    splits: Iterable[str] = _SPLITS,
) -> Dict[Tuple[str, str], Path]:
    """Write per-(task, split) manifest JSONs under
    ``<staging_root>/manifests/<task>/<split>.json``.

    Returns ``{(recipe_key, split): path}`` for every JSON written.
    Empty (task, split) combinations (no rows after filtering, or specs
    that intentionally emit nothing — e.g. VQA) are skipped silently.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "pandas is required for split_manifests; install via "
            "`pip install pandas`."
        ) from e

    staging_root = Path(staging_root)
    parquet = _parquet_path(staging_root)
    df = pd.read_parquet(parquet)

    # Authoritative scene-wise splits.
    scene_split = _scene_wise_splits(df)
    df = df.drop(columns=["split"]).merge(
        scene_split.rename("split"), left_on="scene_id", right_index=True,
    )

    chosen = list(tasks) if tasks is not None else list(_TASK_SPECS.keys())
    manifests_dir = staging_root / "manifests"
    written: Dict[Tuple[str, str], Path] = {}

    for recipe_key in chosen:
        if recipe_key not in _TASK_SPECS:
            log.warning("unknown task %r — skipping", recipe_key)
            continue
        spec = _TASK_SPECS[recipe_key]

        # Filter to rows that have every required modality.
        required_cols = [_has_col(m) for m in spec.required_modalities]
        sub_all = _filter_required(df, required_cols)
        if sub_all is None or sub_all.empty:
            log.info(
                "no parquet rows satisfy %s's required modalities (%s); "
                "skipping all splits.",
                recipe_key, list(spec.required_modalities),
            )
            continue

        for split in splits:
            sub = sub_all[sub_all["split"].astype(str) == split]
            if sub.empty:
                log.info("no samples for task=%s split=%s; skipping",
                         recipe_key, split)
                continue

            samples = spec.build_entries(sub)
            if not samples:
                # Spec opted out (e.g., VQA needs labels not present yet).
                continue

            payload = {
                "task":   spec.task_type_value,   # ← TaskType.value, not recipe_key
                "split":  split,
                "root":   None,                    # filled at download time
                "samples": samples,
            }
            out = manifests_dir / recipe_key / f"{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("wrote %s (%d samples, task_type=%s)",
                     out, len(samples), spec.task_type_value)
            written[(recipe_key, split)] = out

    return written


__all__ = ["write_split_manifests"]

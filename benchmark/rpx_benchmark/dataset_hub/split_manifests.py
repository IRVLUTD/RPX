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
object_tracking     object_tracking      rgb, mask                            [per-frame]
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

from .packer import SCENE_ROOT_BY_TYPE, phase_segment
from .recipes import SceneType

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
    # modality_name → (parquet_col_suffix, extracted_subdir, default_ext)
    #
    # ``default_ext`` is the v1-style extension. Real-world datasets are
    # described by ``current.json``'s ``modality_extensions`` block
    # (written by the manifest builder); when a per-modality entry is
    # present there, it overrides this default at write-split-manifests
    # time. Anything missing from ``current.json`` keeps the v1 default
    # for backward compatibility with already-uploaded datasets.
    "rgb": ("rgb", "rgb", ".png"),
    "depth": ("depth", "depth", ".png"),
    "fisheye": ("fisheye", "fisheye", ".png"),  # see note for left/right
    "fisheye_left": ("fisheye", "fisheye/left", ".png"),
    "fisheye_right": ("fisheye", "fisheye/right", ".png"),
    "masks": ("masks", "sam2/masks", ".png"),
    "cam_pose": ("cam_pose", "cam_pose", ".npz"),
}


# Active layout for the duration of one ``write_split_manifests`` call.
# Module-level rather than a thread-local because this whole subsystem
# is invoked single-threadedly from the CLI. Reset to ``None`` on the
# way out so a subsequent call without overrides falls back to defaults.
_active_layout: Optional[Dict[str, Tuple[str, str, str]]] = None


def _current_layout() -> Dict[str, Tuple[str, str, str]]:
    return _active_layout if _active_layout is not None else _MODALITY_LAYOUT


def _layout_with_overrides(
    extensions: Mapping[str, str],
) -> Dict[str, Tuple[str, str, str]]:
    """Return a copy of ``_MODALITY_LAYOUT`` with per-modality extensions
    swapped in from ``extensions``. Unknown modality names in
    ``extensions`` are ignored (forward compatibility with any new
    modality the manifest writer might add later)."""
    out: Dict[str, Tuple[str, str, str]] = {}
    for modality, (col, subdir, default_ext) in _MODALITY_LAYOUT.items():
        ext = extensions.get(modality, default_ext)
        out[modality] = (col, subdir, ext)
    return out


def _modality_path(
    scene: str, phase: int, modality: str, frame_stem: str,
    scene_type: str = SceneType.MULTI_OBJECT.value,
) -> str:
    """Build the manifest's path string for one (scene, phase, modality, frame).

    Path is relative to the manifest's ``root`` (set by ``download_split``
    to the snapshot root). After ``_extract_snapshot_tars`` runs, the file
    lives under ``extracted/<repo_root>/...`` where ``<repo_root>`` is
    ``scenes`` for mos, ``objects`` for sos, ``ego`` for ego (see
    packer.SCENE_ROOT_BY_TYPE — the single source of truth this must stay
    in sync with). ``scene_type`` defaults to multi_object/"scenes" so
    every pre-ego call site (all mos-only specs) is unaffected; only the
    specs actually shared with ego (_SegmentationSpec, _ObjectTrackingSpec)
    pass the row's real scene_type.

    Reads the active layout (set by :func:`write_split_manifests`); falls
    back to :data:`_MODALITY_LAYOUT` defaults outside that context.
    """
    layout = _current_layout()
    if modality not in layout:
        from ..exceptions import ConfigError

        raise ConfigError(
            f"unknown modality {modality!r}",
            hint=f"Known modalities: {', '.join(sorted(layout))}",
        )
    _, subdir, ext = layout[modality]
    st = SceneType(scene_type)
    repo_root = SCENE_ROOT_BY_TYPE[st]
    seg = phase_segment(st, phase)
    return f"extracted/{repo_root}/{scene}/{seg}/{subdir}/{frame_stem}{ext}"


def _has_col(modality: str) -> str:
    """Return the parquet ``has_<col>`` column name for a modality."""
    col_suffix, _, _ = _MODALITY_LAYOUT[modality]
    return f"has_{col_suffix}"


def _load_modality_extensions(staging_root: Path) -> Dict[str, str]:
    """Read ``modality_extensions`` from ``<staging>/manifest/current.json``.

    Returns an empty dict if the file is missing, malformed, or
    pre-dates the ``modality_extensions`` field — that's the backward-
    compatible signal to keep the v1 hardcoded extensions.
    """
    cur_path = staging_root / "manifest" / "current.json"
    if not cur_path.is_file():
        return {}
    try:
        payload = json.loads(cur_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    raw = payload.get("modality_extensions") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


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

    #: scan/manifest's SceneType.value strings this spec applies to.
    #: Defaults to multi_object (mos) — every spec below predates ego and
    #: was written assuming only mos rows would ever reach it (nothing here
    #: checked scene_type at all, so ego rows silently satisfied the same
    #: rgb+masks requirement and leaked into e.g. "segmentation" instead of
    #: a distinct "ego_segmentation" — see the Ego* specs below, which are
    #: the fix, not new functionality).
    scene_types: Tuple[str, ...] = ("multi_object",)

    def build_entries(self, df) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _SingleFrameDepthSpec(_TaskSpec):
    """``monocular_depth``: rgb + depth GT, one entry per frame."""

    task_type_value = "monocular_depth"
    required_modalities = ("rgb", "depth")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        return [_single_frame_entry(row, ["rgb", "depth"]) for _, row in df.iterrows()]


class _VideoDepthSpec(_TaskSpec):
    """``video_depth``: rgb + depth GT, one entry per ``(scene, phase)`` clip.

    Emits ONE entry per (scene, phase) — not per-frame — with the
    relevant frame filename lists pre-resolved so ``VideoDepthDataset`` can
    iterate clips directly. The structure differs from every other
    spec on this page because Video Depth's iteration unit is a clip, not a
    frame; the cell-log key for the runner becomes
    ``(model, task, scene, phase, frame_budget)`` and one row per clip
    is the right granularity.
    """

    task_type_value = "video_depth"
    required_modalities = ("rgb", "depth")

    def build_entries(self, df) -> List[Dict[str, Any]]:
        # Group the per-frame parquet rows by (scene_id, phase) and
        # collapse each group into a single clip entry.
        rows: List[Dict[str, Any]] = []
        # GroupBy preserves first-row metadata for scene_type/difficulty
        # which are constant within a (scene, phase) cell.
        for (scene, phase), group in df.groupby(["scene_id", "phase"], sort=True):
            stems = sorted(_frame_stem(r) for _, r in group.iterrows())
            rgb_paths = [_modality_path(scene, int(phase), "rgb", s) for s in stems]
            depth_paths = [_modality_path(scene, int(phase), "depth", s) for s in stems]
            pose_paths = [_modality_path(scene, int(phase), "cam_pose", s) for s in stems]
            # Use the first row for cell-level metadata (split, difficulty,
            # scene_type). All rows in this group carry the same values.
            first = next(iter(group.itertuples()))
            rows.append(
                {
                    "id": f"{scene}__{int(phase)}",
                    "scene_id": scene,
                    "phase": int(phase),
                    "scene_type": getattr(first, "scene_type", None),
                    "difficulty": str(getattr(first, "split", "")) or None,
                    "frame_filenames": rgb_paths,
                    "depth_filenames": depth_paths,
                    "pose_filenames": pose_paths,
                    "metadata": {
                        "scene_id": scene,
                        "phase_idx": int(phase),
                        "n_frames": len(stems),
                    },
                }
            )
        return rows


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
            st = str(row["scene_type"])
            e["rgb"] = _modality_path(scene, phase, "rgb", stem, scene_type=st)
            e["mask"] = _modality_path(scene, phase, "masks", stem, scene_type=st)
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
            e["rgb"] = _modality_path(scene, phase, "rgb", stem)
            e["depth"] = _modality_path(scene, phase, "depth", stem)
            e["mask"] = _modality_path(scene, phase, "masks", stem)
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
            e["rgb"] = _modality_path(scene, phase, "rgb", stem)
            e["depth"] = _modality_path(scene, phase, "depth", stem)
            e["fisheye_left"] = _modality_path(scene, phase, "fisheye_left", stem)
            e["fisheye_right"] = _modality_path(scene, phase, "fisheye_right", stem)
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
                    "id": f"{scene}__{phase}__{stem_a}__{stem_b}",
                    "scene_id": scene,
                    "phase": int(phase),
                    "frame_idx": int(row_a["frame_idx"]),
                    "frame_idx_b": int(row_b["frame_idx"]),
                    "difficulty": str(row_a["split"]),
                    "metadata": {
                        "scene_id": scene,
                        "phase_idx": int(phase),
                        "frame": stem_a,
                        "frame_b": stem_b,
                        "pair_stride": self.pair_stride,
                    },
                    "rgb": _modality_path(scene, int(phase), "rgb", stem_a),
                    "rgb_b": _modality_path(scene, int(phase), "rgb", stem_b),
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
            e["depth"] = _modality_path(scene, phase, "depth", stem_a)
            e["depth_b"] = _modality_path(scene, phase, "depth", stem_b)
        return rows


class _ObjectTrackingSpec(_TaskSpec):
    """``object_tracking``: RGB plus temporally consistent instance masks.

    A mask value greater than zero is the persistent instance ID within one
    ``(scene, phase)`` clip.  The production tracking runner groups these
    frame entries into clips and uses the first mask as model initialisation.
    No separate tracklet JSON is needed (and the pinned RPX release does not
    publish one).
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
            st = str(row["scene_type"])
            e["rgb"] = _modality_path(scene, phase, "rgb", stem, scene_type=st)
            e["mask"] = _modality_path(scene, phase, "masks", stem, scene_type=st)
            rows.append(e)
        return rows


class _EgoSegmentationSpec(_SegmentationSpec):
    """``ego_segmentation``: same shape as ``segmentation`` (rgb + mask),
    scoped to ego (egocentric/GoPro) scenes only. Reuses
    TaskType.OBJECT_SEGMENTATION — same evaluation task, just a distinct
    manifest key so a downloader can pull ego's slice without also
    pulling mos's rig-camera segmentation data (and vice versa)."""

    scene_types = ("ego",)


class _EgoObjectTrackingSpec(_ObjectTrackingSpec):
    """``ego_object_tracking``: same shape as ``object_tracking``, scoped
    to ego scenes only. See _EgoSegmentationSpec for why this is a
    separate manifest key rather than a new TaskType."""

    scene_types = ("ego",)


class _VQASpec(_TaskSpec):
    """``vqa``: not yet wired — needs spatial_qa / questionnaire labels.

    Returns no entries until the team's VQA generation pipeline lands
    (tracked separately in `paper-submission/VQA_DESIGN.md`).
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


class _EgoVQASpec(_VQASpec):
    """``ego_vqa``: same reserved-slot shape as ``vqa``, scoped to ego
    scenes only. See _EgoSegmentationSpec for why this is a separate
    manifest key rather than a new TaskType."""

    scene_types = ("ego",)

    def build_entries(self, df) -> List[Dict[str, Any]]:
        log.warning(
            "ego_vqa: spec exists but no entries produced — needs the team's "
            "VQA label generation pipeline (see VQA_DESIGN.md). Skipping."
        )
        return []


# Recipe-key → spec instance.
_TASK_SPECS: Dict[str, _TaskSpec] = {
    "monocular_depth": _SingleFrameDepthSpec(),
    "video_depth": _VideoDepthSpec(),
    "segmentation": _SegmentationSpec(),
    "rgbd_segmentation": _RGBDSegmentationSpec(),
    "stereo_depth": _StereoDepthSpec(),
    "relative_pose": _RelativePoseSpec(),
    "rgbd_relative_pose": _RGBDRelativePoseSpec(),
    "object_tracking": _ObjectTrackingSpec(),
    "ego_segmentation": _EgoSegmentationSpec(),
    "ego_object_tracking": _EgoObjectTrackingSpec(),
    "vqa": _VQASpec(),
    "ego_vqa": _EgoVQASpec(),
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
        "id": f"{scene}__{phase}__{stem}",
        "scene_id": scene,
        "phase": phase,
        "frame_idx": int(row["frame_idx"]),
        "difficulty": str(row["split"]),
        "metadata": {
            "scene_id": scene,
            "phase_idx": phase,
            "frame": stem,
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
            hint="Run `dataset_hub.cli manifest` first to build the parquet from your captures.",
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
            "pandas is required for split_manifests; install via `pip install pandas`."
        ) from e

    staging_root = Path(staging_root)
    parquet = _parquet_path(staging_root)
    df = pd.read_parquet(parquet)

    # Authoritative scene-wise splits.
    scene_split = _scene_wise_splits(df)
    df = df.drop(columns=["split"]).merge(
        scene_split.rename("split"),
        left_on="scene_id",
        right_index=True,
    )

    # Read per-modality extensions from current.json (v2+ datasets).
    # Datasets without this field (v1) fall through to the hardcoded
    # defaults in _MODALITY_LAYOUT.
    extensions = _load_modality_extensions(staging_root)
    layout_override = _layout_with_overrides(extensions) if extensions else None
    if extensions:
        log.info(
            "split_manifests: using modality_extensions from current.json: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(extensions.items())),
        )

    chosen = list(tasks) if tasks is not None else list(_TASK_SPECS.keys())
    manifests_dir = staging_root / "manifests"
    written: Dict[Tuple[str, str], Path] = {}

    global _active_layout
    _active_layout = layout_override
    try:
        written = _write_loop(
            chosen=chosen,
            df=df,
            splits=splits,
            manifests_dir=manifests_dir,
        )
    finally:
        _active_layout = None

    return written


def _write_loop(
    *,
    chosen: List[str],
    df,
    splits: Iterable[str],
    manifests_dir: Path,
) -> Dict[Tuple[str, str], Path]:
    """The per-task per-split write loop, factored out so the layout
    override at module level wraps it in try/finally cleanly."""
    written: Dict[Tuple[str, str], Path] = {}
    for recipe_key in chosen:
        if recipe_key not in _TASK_SPECS:
            log.warning("unknown task %r — skipping", recipe_key)
            continue
        spec = _TASK_SPECS[recipe_key]

        # Scope to this spec's scene family FIRST — required_modalities
        # alone (rgb+masks) is satisfied by both mos and ego rows, so
        # without this an ego frame would qualify for "segmentation" too,
        # not just "ego_segmentation".
        sub_all = df[df["scene_type"].isin(spec.scene_types)]

        # Filter to rows that have every required modality.
        required_cols = [_has_col(m) for m in spec.required_modalities]
        sub_all = _filter_required(sub_all, required_cols)
        if sub_all is None or sub_all.empty:
            log.info(
                "no parquet rows satisfy %s's required modalities (%s); skipping all splits.",
                recipe_key,
                list(spec.required_modalities),
            )
            continue

        for split in splits:
            sub = sub_all[sub_all["split"].astype(str) == split]
            if sub.empty:
                log.info("no samples for task=%s split=%s; skipping", recipe_key, split)
                continue

            samples = spec.build_entries(sub)
            if not samples:
                # Spec opted out (e.g., VQA needs labels not present yet).
                continue

            payload = {
                "task": spec.task_type_value,  # ← TaskType.value, not recipe_key
                "split": split,
                "root": None,  # filled at download time
                "samples": samples,
            }
            out = manifests_dir / recipe_key / f"{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("wrote %s (%d samples, task_type=%s)", out, len(samples), spec.task_type_value)
            written[(recipe_key, split)] = out

    return written


__all__ = ["write_split_manifests"]

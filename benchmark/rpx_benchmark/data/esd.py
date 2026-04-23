"""Effort-Stratified Difficulty (ESD) feature extraction.

Computes the 18 per-(scene, phase) features defined in
``paper-submission/neurips-2026/overleaf/text/12_appendix.tex``
(``\\section{ESD Feature Definitions}``). The output of this module
is the raw feature table; weight calibration and tertile assignment
live downstream.

On-disk layout expected
-----------------------

::

    <data_root>/
      <scene_dir>/                       # e.g. "scene1.GDC.cse" or just "scene1"
        0/  1/  2/                       # phase indices (clutter/interaction/clean)
          rgb/      00000.png ...        # uint8 RGB
          depth/    00000.png ...        # uint16 millimetres, 0 = invalid
          cam_pose/ 00000.npz            # T265 pose: position + orientation [x,y,z,w]
          sam2/
            masks/  00000.png            # 1-channel, pixel value > 0 = instance ID
            mask_to_object.json          # {"<id>": "<label>", ...} (optional)
            iter1_faulty.txt             # frame indices still faulty after iter i
            iter2_faulty.txt             # (one entry per line, basename or int)
            iter3_faulty.txt
            iter4_faulty.txt

Frame indices align across modalities by basename
(``00000.png`` ↔ ``00000.npz``).

Implementation notes
--------------------

The per-phase loop is **streaming**: one frame's RGB / depth / mask /
pose is decoded, contributes to per-feature accumulators, then is
released. Peak memory per worker is ~tens of MB regardless of phase
length, so the CLI can default to ``os.cpu_count()`` workers.

Mask geometry uses a single ``np.bincount`` per frame to derive
visible-IDs, per-instance areas, and the population for occlusion
bboxes — replacing the previous N-pass-over-mask design.
"""

from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..exceptions import DatasetError
from ..logging_utils import get_logger

log = get_logger(__name__)

# Optional cv2 fast-path for PNG decode. Falls back to PIL if cv2 isn't
# installed; tests run on PIL so behaviour stays identical.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


# Names of the 18 features, in the order they appear in the appendix.
# Tests should diff against this list to catch silent additions / renames.
FEATURE_NAMES: Tuple[str, ...] = (
    # Annotation effort
    "iter_mean", "iter_max",
    # Scene complexity
    "obj_mean", "obj_std", "obj_consist",
    # Occlusion
    "occ_mean", "occ_p90", "occ_heavy",
    # Depth quality
    "depth_invalid", "depth_invalid_mask", "depth_std", "depth_std_mask",
    # Photometric--depth conflict (D435 RGB + invalid depth)
    "specular", "dark",
    # Temporal annotation stability
    "area_cv", "area_drop", "vis_instability",
    # Camera motion
    "trans_mean", "trans_p90", "rot_mean", "rot_p90", "jerk",
    # Fisheye / stereo (T265 fisheye pairs) — 0.0 if no fisheye dir present
    "fisheye_dark", "fisheye_bright", "fisheye_sharpness",
    "fisheye_corr", "fisheye_texture",
)

# Threshold used by ``occ-heavy`` (appendix §"Occlusion").
_OCC_HEAVY_THRESHOLD: float = 0.3
# Photometric thresholds used by ``specular`` / ``dark`` / ``fisheye_{dark,bright}``.
_SPECULAR_LUMA_MIN: int = 230
_DARK_LUMA_MAX: int = 30
# Threshold used by ``area_drop`` (old paper §3.4 eq 15).
# Relative area collapse from frame t-1 to frame t exceeding this fraction counts
# as a "drop event". 0.5 = 50% area lost between consecutive frames.
_AREA_DROP_THRESHOLD: float = 0.5


@dataclass
class PhaseFeatures:
    """Features and provenance for one (scene, phase) pair."""

    scene_id: str               # full scene-dir basename
    phase: int                  # 0 / 1 / 2 (clutter / interaction / clean)
    n_frames_total: int         # frames present in rgb/
    n_frames_used: int          # frames where every required modality was present
    features: Dict[str, float] = field(default_factory=dict)

    def as_json(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Frame discovery / loaders
# --------------------------------------------------------------------------- #

def _stem_set(d: Path, suffix: str) -> set[str]:
    if not d.is_dir():
        return set()
    return {p.stem for p in d.iterdir() if p.suffix == suffix and not p.name.startswith(".")}


def _aligned_frames(phase_dir: Path) -> List[str]:
    """Frame stems that exist across rgb / depth / mask / pose, sorted.

    Logs a warning summarising any per-modality misalignment so silent
    data loss is visible without spamming the log per missing frame.
    """
    rgb   = _stem_set(phase_dir / "rgb",        ".png")
    depth = _stem_set(phase_dir / "depth",      ".png")
    masks = _stem_set(phase_dir / "sam2/masks", ".png")
    pose  = _stem_set(phase_dir / "cam_pose",   ".npz")
    common = rgb & depth & masks & pose

    union = rgb | depth | masks | pose
    if union and len(common) < len(union):
        missing = {
            "rgb":   len(union - rgb),
            "depth": len(union - depth),
            "mask":  len(union - masks),
            "pose":  len(union - pose),
        }
        gaps = ", ".join(f"{k}:{v}" for k, v in missing.items() if v > 0)
        log.warning(
            "%s: dropped %d frames due to modality misalignment (missing in: %s)",
            phase_dir, len(union) - len(common), gaps,
        )
    return sorted(common)


@functools.lru_cache(maxsize=None)
def _warn_multichannel_once(parent_dir: str, n_channels: int) -> None:
    """Emit one multi-channel-PNG warning per parent directory.

    A phase with hundreds of mistakenly-RGB-saved masks would otherwise spam
    the log with one warning per frame; this collapses them to one line per
    directory. ``lru_cache`` is the dedupe; tests call ``cache_clear()`` to
    reset between cases.
    """
    log.warning(
        "%s: multi-channel PNGs detected (%d ch); using first channel "
        "(further warnings for this directory suppressed)",
        parent_dir, n_channels,
    )


def _decode_png_gray(path: Path) -> np.ndarray:
    """Decode a single-channel PNG. Uses cv2 if available, PIL otherwise.

    Multi-channel inputs (e.g. masks accidentally saved as RGB) are collapsed
    to the first channel; one warning is emitted per containing directory
    (see :func:`_warn_multichannel_once`).
    """
    if _HAS_CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise DatasetError(f"failed to decode PNG: {path}",
                               hint="file may be corrupt or zero-byte.")
        if arr.ndim == 3:
            _warn_multichannel_once(str(path.parent), arr.shape[2])
            arr = arr[..., 0]
        return arr
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        _warn_multichannel_once(str(path.parent), arr.shape[2])
        arr = arr[..., 0]
    return arr


def _load_depth_m(path: Path) -> np.ndarray:
    """16-bit PNG millimetres → float32 metres; 0 stays 0 (sentinel)."""
    arr = _decode_png_gray(path).astype(np.float32) / 1000.0
    return arr


def _load_mask(path: Path) -> np.ndarray:
    """Single-channel PNG; pixel value > 0 = instance ID. Returned as int32."""
    arr = _decode_png_gray(path)
    return arr.astype(np.int32, copy=False)


def _load_rgb_luma(path: Path) -> np.ndarray:
    """RGB → ITU-R BT.601 luma (uint8) for photometric thresholds."""
    if _HAS_CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
        if arr is None:
            raise DatasetError(f"failed to decode PNG: {path}",
                               hint="file may be corrupt or zero-byte.")
        # cv2 returns BGR; BT.601 weights stay the same when applied to channels.
        b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
        return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return luma.astype(np.uint8)


def _load_pose(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """T265 pose: ``position`` (3,), ``orientation`` (4,) in [x,y,z,w]."""
    with np.load(path) as f:
        pos = np.asarray(f["position"], dtype=np.float64).reshape(3)
        quat = np.asarray(f["orientation"], dtype=np.float64).reshape(4)
    return pos, quat


# --------------------------------------------------------------------------- #
# Annotation-effort extractor (no per-frame I/O — reads small txt files only)
# --------------------------------------------------------------------------- #

def _annotation_effort(phase_dir: Path, frame_stems: List[str]) -> Dict[str, float]:
    """``iter-mean`` / ``iter-max``.

    ``k_j = 1 + |{ i : frame j ∈ iter_i_faulty.txt }|``. A frame absent from
    every iter_i_faulty.txt was accepted on the first pass (k=1); a frame
    listed in all four was still faulty after iter 4 (k=5).
    """
    sam_dir = phase_dir / "sam2"
    counts = np.ones(len(frame_stems), dtype=np.int32)
    if not sam_dir.is_dir() or counts.size == 0:
        return {"iter_mean": 1.0, "iter_max": 1.0}

    stem_to_idx = {s: i for i, s in enumerate(frame_stems)}
    for i in (1, 2, 3, 4):
        f = sam_dir / f"iter{i}_faulty.txt"
        if not f.is_file():
            continue
        for raw in f.read_text().splitlines():
            entry = raw.strip()
            if not entry:
                continue
            stem = Path(entry).stem  # tolerates "00007", "00007.png", "/abs/.../00007.png"
            idx = stem_to_idx.get(stem)
            if idx is not None:
                counts[idx] += 1

    return {"iter_mean": float(counts.mean()), "iter_max": float(counts.max())}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _bbox_xywh(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox of a boolean mask, or None if empty."""
    rows = np.flatnonzero(binary_mask.any(axis=1))
    cols = np.flatnonzero(binary_mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _occlusion_from_per_frame_bboxes(
    bboxes_per_frame: List[Dict[int, Tuple[int, int, int, int]]],
) -> Dict[str, float]:
    """``occ-mean`` / ``occ-p90`` / ``occ-heavy`` from per-frame bbox dicts.

    Per-object occlusion = mean over frames (where the object is visible) of
    that frame's bbox-overlap-fraction. Then aggregate over objects.
    """
    per_object: Dict[int, List[float]] = {}
    for bboxes in bboxes_per_frame:
        if not bboxes:
            continue
        if len(bboxes) < 2:
            for i in bboxes:
                per_object.setdefault(i, []).append(0.0)
            continue
        for i, (xi, yi, wi, hi) in bboxes.items():
            area_i = wi * hi
            if area_i == 0:
                continue
            inter_sum = 0
            for j, (xj, yj, wj, hj) in bboxes.items():
                if i == j:
                    continue
                ix0, iy0 = max(xi, xj), max(yi, yj)
                ix1, iy1 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
                if ix1 > ix0 and iy1 > iy0:
                    inter_sum += (ix1 - ix0) * (iy1 - iy0)
            per_object.setdefault(i, []).append(inter_sum / area_i)

    if not per_object:
        return {"occ_mean": 0.0, "occ_p90": 0.0, "occ_heavy": 0.0}

    per_object_mean = np.asarray(
        [float(np.mean(v)) for v in per_object.values()], dtype=np.float32,
    )
    return {
        "occ_mean":  float(per_object_mean.mean()),
        "occ_p90":   float(np.percentile(per_object_mean, 90)),
        "occ_heavy": float((per_object_mean > _OCC_HEAVY_THRESHOLD).mean()),
    }


def _temporal_stability_from_areas(areas_T_by_id: np.ndarray) -> Dict[str, float]:
    """``area-cv`` / ``area-drop`` / ``vis-instability`` from a (T, n_ids) area matrix.

    Column ``j`` is the per-frame pixel count of instance ID ``j+1``.

    ``area_drop`` follows old paper §3.4 eq 15: per-instance fraction of frames
    where ``(A_{t-1} - A_t) / A_{t-1} > δ`` (with ``A_{t-1} > 0``), then
    averaged over instances. Captures sudden mask collapses (typically caused
    by an arm sweeping across the object during interaction).
    """
    T, n_ids = areas_T_by_id.shape
    if n_ids == 0 or T == 0:
        return {"area_cv": 0.0, "area_drop": 0.0, "vis_instability": 0.0}

    present = areas_T_by_id > 0
    cvs, drops, instabilities = [], [], []
    for j in range(n_ids):
        col = areas_T_by_id[:, j]
        pres_col = present[:, j]

        if not pres_col.any():
            continue  # instance never observed (column would be all zeros)

        positive = col[pres_col]
        mean_a = float(positive.mean())
        if mean_a > 0:
            cvs.append(float(positive.std() / mean_a))

        if T > 1:
            flips = int(np.sum(pres_col[1:] != pres_col[:-1]))
            instabilities.append(flips / T)

            prev = col[:-1].astype(np.float64)
            curr = col[1:].astype(np.float64)
            valid = prev > 0
            if valid.any():
                ratios = np.zeros_like(prev)
                ratios[valid] = (prev[valid] - curr[valid]) / prev[valid]
                n_drop = int(np.sum(ratios > _AREA_DROP_THRESHOLD))
                drops.append(n_drop / T)

    return {
        "area_cv":         float(np.mean(cvs))           if cvs           else 0.0,
        "area_drop":       float(np.mean(drops))         if drops         else 0.0,
        "vis_instability": float(np.mean(instabilities)) if instabilities else 0.0,
    }


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle (rad) between two unit quaternions [x,y,z,w]."""
    n1 = float(np.linalg.norm(q1))
    n2 = float(np.linalg.norm(q2))
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = abs(float(np.dot(q1 / n1, q2 / n2)))
    if dot > 1.0:
        dot = 1.0
    return float(2.0 * np.arccos(dot))


def _camera_motion_from_arrays(
    positions: np.ndarray, quats: np.ndarray,
) -> Dict[str, float]:
    """Camera motion features.

    - ``trans_mean`` / ``trans_p90``: per-frame translation magnitude (m)
    - ``rot_mean`` / ``rot_p90``: per-frame rotation angle (rad)
    - ``jerk``: mean |Δtranslation_{t+1} - Δtranslation_t| (m/frame²)

    The p90 features (old paper §3.5 eqs 20, 22) capture heavy-tail motion
    spikes; mean features alone hide brief but extreme camera jerks.
    """
    if positions.shape[0] < 2:
        return {"trans_mean": 0.0, "trans_p90": 0.0,
                "rot_mean":   0.0, "rot_p90":   0.0,
                "jerk":       0.0}

    delta_t = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    delta_r = np.asarray(
        [_quat_angle(quats[i], quats[i + 1]) for i in range(quats.shape[0] - 1)],
        dtype=np.float64,
    )
    jerk = np.abs(np.diff(delta_t)) if delta_t.size >= 2 else np.zeros(0)

    return {
        "trans_mean": float(delta_t.mean()),
        "trans_p90":  float(np.percentile(delta_t, 90)),
        "rot_mean":   float(delta_r.mean()),
        "rot_p90":    float(np.percentile(delta_r, 90)),
        "jerk":       float(jerk.mean()) if jerk.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# Fisheye / stereo features (old paper §3.7 eqs 25-29)
# --------------------------------------------------------------------------- #

_FISHEYE_FEATURE_NAMES: Tuple[str, ...] = (
    "fisheye_dark", "fisheye_bright", "fisheye_sharpness",
    "fisheye_corr", "fisheye_texture",
)
# Regex for ``*_L.png`` / ``*_R.png`` style fisheye pairs.
_FISHEYE_LR_RE = re.compile(r"^(?P<stem>.+?)[_-](?P<side>[LRlr])$")


def _discover_fisheye(
    fisheye_dir: Path,
) -> Tuple[List[str], Dict[str, Path], Dict[str, Path]]:
    """Find fisheye frames; return (sorted_stems, left_path_map, right_path_map).

    Auto-detects three layouts in priority order:

    1. Subdirs ``left/`` + ``right/`` with matching basenames.
    2. Files like ``00000_L.png`` + ``00000_R.png`` (or ``-L``/``-R``).
    3. Single PNGs in the dir → treated as left only; ``fisheye_corr`` will
       be unset for those frames.

    Returns ``([], {}, {})`` if the directory is missing or empty.
    """
    if not fisheye_dir.is_dir():
        return [], {}, {}

    # Layout A: left/ + right/ subdirectories.
    ld, rd = fisheye_dir / "left", fisheye_dir / "right"
    if ld.is_dir() and rd.is_dir():
        left  = {p.stem: p for p in ld.iterdir() if p.suffix.lower() in (".png", ".jpg")}
        right = {p.stem: p for p in rd.iterdir() if p.suffix.lower() in (".png", ".jpg")}
        common = sorted(set(left) & set(right))
        return common, left, right

    # Layout B: <stem>_L.png + <stem>_R.png.
    files = [p for p in fisheye_dir.iterdir()
             if p.is_file() and p.suffix.lower() in (".png", ".jpg")]
    left, right = {}, {}
    for f in files:
        m = _FISHEYE_LR_RE.match(f.stem)
        if m:
            stem = m.group("stem")
            if m.group("side").lower() == "l":
                left[stem] = f
            else:
                right[stem] = f
    if left and right:
        common = sorted(set(left) & set(right))
        return common, left, right

    # Layout C: single fisheye images (no stereo pair). corr → undefined.
    singles = {p.stem: p for p in files}
    if singles:
        return sorted(singles), singles, {}

    return [], {}, {}


def _laplacian_var(img: np.ndarray) -> float:
    """Variance of Laplacian — image sharpness / blur measure."""
    if _HAS_CV2:
        lap = cv2.Laplacian(img, cv2.CV_32F)
        return float(lap.var())
    f = img.astype(np.float32)
    lap = (np.roll(f, 1, 0) + np.roll(f, -1, 0)
           + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4.0 * f)
    return float(lap.var())


def _gradient_magnitude_mean(img: np.ndarray) -> float:
    """Mean of pixelwise |∇F| — texture density."""
    f = img.astype(np.float32)
    gy, gx = np.gradient(f)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _pearson_corr_2d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson correlation between two same-size single-channel images.

    Resizes ``b`` to ``a``'s shape if they differ — matches the literal paper
    formula ``corr(F^L, F^R)`` without requiring stereo rectification.
    """
    if a.shape != b.shape:
        if _HAS_CV2:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        else:
            b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0])))
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = math.sqrt((af * af).sum() * (bf * bf).sum())
    if denom == 0:
        return None  # constant image — correlation undefined
    return float((af * bf).sum() / denom)


def _fisheye_features(phase_dir: Path) -> Dict[str, float]:
    """Compute the 5 fisheye features for one (scene, phase).

    Returns all-zero dict if no fisheye dir / no decodable frames found.
    Logs a warning when stereo pairs are missing (``fisheye_corr`` then
    aggregates only over frames that DO have pairs; if none, → 0).
    """
    empty = {name: 0.0 for name in _FISHEYE_FEATURE_NAMES}
    stems, left_paths, right_paths = _discover_fisheye(phase_dir / "fisheye")
    if not stems:
        return empty

    has_right = bool(right_paths)
    if not has_right:
        log.warning(
            "%s/fisheye: no stereo pairs found; fisheye_corr will be 0",
            phase_dir,
        )

    dark_acc:   List[float] = []
    bright_acc: List[float] = []
    sharp_acc:  List[float] = []
    tex_acc:    List[float] = []
    corr_acc:   List[float] = []

    for s in stems:
        try:
            L = _decode_png_gray(left_paths[s])
        except Exception as e:
            log.warning("fisheye decode failed for %s: %s", left_paths[s], e)
            continue

        dark_acc.append(float((L < _DARK_LUMA_MAX).mean()))
        bright_acc.append(float((L > _SPECULAR_LUMA_MIN).mean()))
        sharp_acc.append(_laplacian_var(L))
        tex_acc.append(_gradient_magnitude_mean(L))

        if has_right and s in right_paths:
            try:
                R = _decode_png_gray(right_paths[s])
                c = _pearson_corr_2d(L, R)
                if c is not None:
                    corr_acc.append(c)
            except Exception as e:
                log.warning("fisheye-R decode failed for %s: %s", right_paths[s], e)

    return {
        "fisheye_dark":      float(np.mean(dark_acc))   if dark_acc   else 0.0,
        "fisheye_bright":    float(np.mean(bright_acc)) if bright_acc else 0.0,
        "fisheye_sharpness": float(np.mean(sharp_acc))  if sharp_acc  else 0.0,
        "fisheye_corr":      float(np.mean(corr_acc))   if corr_acc   else 0.0,
        "fisheye_texture":   float(np.mean(tex_acc))    if tex_acc    else 0.0,
    }


# --------------------------------------------------------------------------- #
# Top-level driver — streaming, single I/O pass
# --------------------------------------------------------------------------- #

def _scene_phase_from_dir(phase_dir: Path) -> Tuple[str, int]:
    scene_id = phase_dir.parent.name
    try:
        phase_idx = int(phase_dir.name)
    except ValueError as e:
        raise DatasetError(
            f"phase dir name must be an integer, got {phase_dir.name!r}",
            hint="phase_dir is expected to be <data_root>/<scene>/<phase_idx>; "
                 "rename or pass a deeper path.",
        ) from e
    return scene_id, phase_idx


def _empty_features() -> Dict[str, float]:
    return {k: 0.0 for k in FEATURE_NAMES}


def extract_phase_features(phase_dir: Path) -> PhaseFeatures:
    """Compute all 18 ESD features for one (scene, phase) directory.

    ``phase_dir`` is expected to be ``<data_root>/<scene_dir>/<phase_idx>``;
    e.g. ``test_dataset_aggregated/scene1.GDC.cse/0``. The scene_id and
    phase fields of the result are inferred from this path.

    The per-frame loop is streaming: one frame's modalities are decoded,
    fed into per-feature accumulators, and discarded. Peak memory is
    independent of phase length.
    """
    phase_dir = Path(phase_dir)
    scene_id, phase_idx = _scene_phase_from_dir(phase_dir)

    rgb_dir = phase_dir / "rgb"
    n_total = len(_stem_set(rgb_dir, ".png")) if rgb_dir.is_dir() else 0
    stems = _aligned_frames(phase_dir)

    if not stems:
        log.warning("no aligned frames found for %s/%d", scene_id, phase_idx)
        return PhaseFeatures(scene_id=scene_id, phase=phase_idx,
                             n_frames_total=n_total, n_frames_used=0,
                             features=_empty_features())

    T = len(stems)

    # Annotation-effort features need no per-frame I/O.
    feats: Dict[str, float] = dict(_annotation_effort(phase_dir, stems))

    # Streaming accumulators.
    obj_counts          = np.zeros(T, dtype=np.int32)
    depth_invalid_frac  = np.zeros(T, dtype=np.float64)
    depth_inv_mask_frac = np.full(T, np.nan, dtype=np.float64)  # NaN → no mask pixels
    depth_std_per_frame = np.full(T, np.nan, dtype=np.float64)  # NaN → all-invalid
    depth_std_mask_per_frame = np.full(T, np.nan, dtype=np.float64)  # NaN → no valid depth in mask
    spec_frac           = np.zeros(T, dtype=np.float64)
    dark_frac           = np.zeros(T, dtype=np.float64)
    bboxes_per_frame: List[Dict[int, Tuple[int, int, int, int]]] = []
    # Per-frame dense bincount of mask values (length varies by frame max ID);
    # zipped at the end into a (T, max_id) matrix for area_cv / vis_instability.
    bincount_per_frame: List[np.ndarray] = []
    positions = np.zeros((T, 3), dtype=np.float64)
    quats     = np.zeros((T, 4), dtype=np.float64)

    for t, s in enumerate(stems):
        depth_m  = _load_depth_m(phase_dir / "depth"      / f"{s}.png")
        mask     = _load_mask   (phase_dir / "sam2/masks" / f"{s}.png")
        rgb_luma = _load_rgb_luma(phase_dir / "rgb"       / f"{s}.png")
        pos, q   = _load_pose   (phase_dir / "cam_pose"   / f"{s}.npz")

        # ── Depth-quality features ────────────────────────────────────────
        invalid = depth_m <= 0
        depth_invalid_frac[t] = float(invalid.mean())

        in_mask = mask > 0
        if in_mask.any():
            depth_inv_mask_frac[t] = float(invalid[in_mask].mean())

        valid = depth_m[~invalid]
        if valid.size > 0:
            depth_std_per_frame[t] = float(valid.std())

        # In-mask depth std: only valid-depth pixels that also fall inside any
        # instance mask. Captures object-relevant depth noise (paper's
        # depth_std is whole-image and dominated by background).
        valid_in_mask = depth_m[(~invalid) & in_mask]
        if valid_in_mask.size > 0:
            depth_std_mask_per_frame[t] = float(valid_in_mask.std())

        # ── Photometric-conflict features ─────────────────────────────────
        spec_frac[t] = float(((rgb_luma > _SPECULAR_LUMA_MIN) & invalid).mean())
        dark_frac[t] = float(((rgb_luma < _DARK_LUMA_MAX)     & invalid).mean())

        # ── Mask-derived features (one bincount per frame, reused 3×) ─────
        flat = mask.ravel()
        # Drop negative IDs (defensive — masks should be unsigned).
        if flat.min() < 0:
            flat = flat[flat >= 0]
        counts = np.bincount(flat)        # counts[i] = pixel count for ID i
        ids = np.flatnonzero(counts[1:]) + 1 if counts.size > 1 else np.empty(0, dtype=np.int64)
        obj_counts[t] = ids.size
        bincount_per_frame.append(counts)

        bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        for oid in ids:
            b = _bbox_xywh(mask == oid)
            if b is not None:
                bboxes[int(oid)] = b
        bboxes_per_frame.append(bboxes)

        # ── Pose ──────────────────────────────────────────────────────────
        positions[t] = pos
        quats[t]     = q

    # ── n_total_objects: prefer mask_to_object.json, else union of observed ──
    obj_json = phase_dir / "sam2" / "mask_to_object.json"
    if obj_json.is_file():
        try:
            n_total_objects = len(json.loads(obj_json.read_text()))
        except (ValueError, OSError) as e:
            log.warning("%s: cannot parse, falling back to observed IDs (%s)", obj_json, e)
            n_total_objects = max(
                (int(c.size - 1) for c in bincount_per_frame if c.size > 1), default=0,
            )
    else:
        all_ids = {int(i) for c in bincount_per_frame
                   for i in (np.flatnonzero(c[1:]) + 1 if c.size > 1 else ())}
        n_total_objects = len(all_ids)

    # ── Scene-complexity features ────────────────────────────────────────
    feats["obj_mean"] = float(obj_counts.mean())
    feats["obj_std"]  = float(obj_counts.std())
    feats["obj_consist"] = (
        float((obj_counts == n_total_objects).mean()) if n_total_objects > 0 else 0.0
    )

    # ── Occlusion ────────────────────────────────────────────────────────
    feats.update(_occlusion_from_per_frame_bboxes(bboxes_per_frame))

    # ── Depth aggregates (NaN means "skip frame for this aggregate") ─────
    feats["depth_invalid"]      = float(depth_invalid_frac.mean())
    feats["depth_invalid_mask"] = (
        float(np.nanmean(depth_inv_mask_frac))
        if not np.isnan(depth_inv_mask_frac).all() else 0.0
    )
    feats["depth_std"] = (
        float(np.nanmean(depth_std_per_frame))
        if not np.isnan(depth_std_per_frame).all() else 0.0
    )
    feats["depth_std_mask"] = (
        float(np.nanmean(depth_std_mask_per_frame))
        if not np.isnan(depth_std_mask_per_frame).all() else 0.0
    )

    # ── Photometric ──────────────────────────────────────────────────────
    feats["specular"] = float(spec_frac.mean())
    feats["dark"]     = float(dark_frac.mean())

    # ── Temporal stability: build dense (T, max_id) area matrix ──────────
    max_id = max((int(c.size - 1) for c in bincount_per_frame if c.size > 1), default=0)
    if max_id >= 1:
        areas = np.zeros((T, max_id), dtype=np.int64)
        for t, c in enumerate(bincount_per_frame):
            n = min(int(c.size) - 1, max_id)
            if n > 0:
                areas[t, :n] = c[1:1 + n]
        feats.update(_temporal_stability_from_areas(areas))
    else:
        feats["area_cv"] = 0.0
        feats["vis_instability"] = 0.0

    # ── Camera motion ────────────────────────────────────────────────────
    feats.update(_camera_motion_from_arrays(positions, quats))

    # ── Fisheye / stereo (independent I/O — uses fisheye/, not the
    # rgb/depth/mask/pose modalities streamed above) ─────────────────────
    feats.update(_fisheye_features(phase_dir))

    # Stable column order matching FEATURE_NAMES; missing keys treated as 0.
    feats = {k: float(feats.get(k, 0.0)) for k in FEATURE_NAMES}

    return PhaseFeatures(
        scene_id=scene_id, phase=phase_idx,
        n_frames_total=n_total, n_frames_used=T,
        features=feats,
    )


def iter_phase_dirs(data_root: Path) -> Iterator[Path]:
    """Yield every ``<scene>/<phase>`` directory under ``data_root``.

    A phase directory is any immediate child of a scene directory whose
    name is purely numeric and that contains an ``rgb/`` subdirectory.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise DatasetError(f"data_root is not a directory: {data_root}",
                           hint="check the --data-root argument.")
    for scene_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for child in sorted(scene_dir.iterdir()):
            if child.is_dir() and child.name.isdigit() and (child / "rgb").is_dir():
                yield child

"""Re-encode a capture tree into denser lossless formats for HF upload.

Three modality-aware re-encodings are applied (each can be skipped via
:class:`ConvertSpec` toggles). Every encoded artefact is round-trip
verified per-frame against the source: the worker decodes the freshly
written file and asserts ``numpy.array_equal`` (or full pose-vector
equality for cam_pose) before the run is allowed to continue. The first
verification failure aborts the entire run with the offending filename
and a diagnostic — no silent loss is possible.

Modalities and what each becomes
--------------------------------

* **rgb / fisheye** (and any nested ``rgb/`` like ``ego/rgb/``)
  ``*.png`` → ``*.webp`` (libwebp lossless, method=4).
  ~45-55% denser on real-world photo content. Decoded pixels are
  bit-identical to source via PIL / cv2.

* **depth, sam2/masks, sam2/masks_verified**
  ``*.png`` → ``*.png`` re-encoded at ``compress_level=9``.
  ~18-23% denser; format and mode are preserved exactly — PNG is
  lossless at every compression level by spec. Decoded pixel arrays
  are bit-identical to source via the same loaders adapters already
  use, with no API change.

* **cam_pose**
  ``*.npz`` (per-frame zip wrapping ``position`` + ``orientation``)
  → ``*.npy`` (per-frame raw, single ``(7,) float64`` vector packing
  ``[x, y, z, qx, qy, qz, qw]``). ~68% denser per frame (90% of the
  current size is per-file zip overhead). Per-frame access is
  preserved; the loader dispatches on suffix and remains backward
  compatible with the legacy ``.npz`` form.

* **everything else** — hard-linked (or copied on cross-filesystem)
  into the output tree unchanged. Hard-linking is zero-cost on disk.

Why the format zoo is safe
--------------------------

Every change preserves the decoded pixel/value array the model sees.
The encoder asserts ``np.array_equal(src, decoded(encoded(src)))`` at
write time, so each artefact on disk is proven bit-faithful before the
run continues. PNG-lossless and WebP-lossless are deterministic by
spec (RFC 2083 / RFC 9649) — different decoder versions on different
machines produce the same pixel arrays. ``.npy`` round-trips IEEE-754
floats exactly.

Why we did not go further (deferred to later work)
--------------------------------------------------

The 2-channel hi/lo split that would let depth/masks fit into
8-bit-WebP-lossless (~+19 GB on the full dataset) was rejected to keep
the on-disk file format standard: contributors must not be required to
go through the canonical loader to recover correct depth/masks values.
That optimisation can be revisited after the loader is wrapped in a
typed API that adapters cannot bypass.

Usage
-----

::

    from rpx_benchmark.dataset_hub.lossless_convert import (
        ConvertSpec, convert_capture_tree,
    )
    spec = ConvertSpec(src_root=Path("DATA"), out_root=Path("DATA_v2"))
    result = convert_capture_tree(spec)
    print(f"saved {result.saved_bytes/1e9:.1f} GB")
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..exceptions import DatasetError
from ..logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Per-file actions
# --------------------------------------------------------------------------- #

ACTION_WEBP = "webp"  # 8-bit PNG → lossless WebP
ACTION_PNG_RECOMPRESS = "png-recompress"  # PNG → PNG at compress_level=9
ACTION_CAM_POSE = "cam-pose"  # per-frame .npz → per-frame .npy
ACTION_LINK = "link"  # hardlink (or copy)

# Modality directory names whose 8-bit PNG frames are re-encoded as
# lossless WebP. A file is classified ``webp`` iff its suffix is .png
# AND any ancestor directory between src_root and the file is in this
# set. Covers ``rgb/``, ``fisheye/``, and nested cases like
# ``ego/rgb/`` produced by the GoPro frame extractor.
WEBP_PARENT_DIRS = frozenset({"rgb", "fisheye"})

# Modality directory names whose PNGs are re-encoded as PNG at
# compress_level=9 (lossless, mode/dtype preserved, ~20% denser).
# Depth must stay PNG because libwebp does not support 16-bit;
# masks must stay PNG to preserve palette/I;16 instance-ID semantics.
PNG_RECOMPRESS_PARENT_DIRS = frozenset({"depth", "masks", "masks_verified"})

# Directory name that gates the cam_pose path.
CAM_POSE_PARENT_DIR = "cam_pose"

CONVERT_FROM_SUFFIXES = frozenset({".png", ".PNG"})
WEBP_SUFFIX = ".webp"
NPY_SUFFIX = ".npy"
NPZ_SUFFIX = ".npz"


@dataclass(frozen=True)
class ConvertSpec:
    """Inputs to :func:`convert_capture_tree`."""

    src_root: Path
    out_root: Path
    workers: int = max(1, (os.cpu_count() or 4) - 1)
    dry_run: bool = False
    verify: bool = True
    overwrite_out: bool = False
    # libwebp lossless effort: 0=fastest/largest, 6=slowest/smallest.
    # method=4 is the sweet spot on photo content (method=6 takes ~10x
    # longer for ~0% extra savings on natural images).
    webp_method: int = 4
    # PNG re-encode compression level (0-9). 9 = densest, still lossless.
    png_compress_level: int = 9
    # Per-modality opt-outs (default: all three paths enabled).
    skip_rgb_webp: bool = False
    skip_png_recompress: bool = False
    skip_cam_pose: bool = False


@dataclass
class ActionStats:
    """Counts and byte totals for one action."""

    files: int = 0
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def saved_bytes(self) -> int:
        return self.bytes_before - self.bytes_after

    @property
    def saved_pct(self) -> float:
        if self.bytes_before == 0:
            return 0.0
        return 100.0 * self.saved_bytes / self.bytes_before


@dataclass
class ConvertResult:
    """Outputs of :func:`convert_capture_tree`."""

    files_seen: int = 0
    by_action: dict = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    def _stats(self, action: str) -> ActionStats:
        return self.by_action.setdefault(action, ActionStats())

    @property
    def files_converted(self) -> int:
        """Files that produced a re-encoded artefact (sum of non-link actions)."""
        return sum(s.files for a, s in self.by_action.items() if a != ACTION_LINK)

    @property
    def files_linked(self) -> int:
        return self._stats(ACTION_LINK).files

    @property
    def bytes_before(self) -> int:
        return sum(s.bytes_before for s in self.by_action.values())

    @property
    def bytes_after(self) -> int:
        return sum(s.bytes_after for s in self.by_action.values())

    @property
    def saved_bytes(self) -> int:
        return self.bytes_before - self.bytes_after

    @property
    def saved_pct(self) -> float:
        if self.bytes_before == 0:
            return 0.0
        return 100.0 * self.saved_bytes / self.bytes_before


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _classify(path: Path, src_root: Path, spec: Optional[ConvertSpec] = None) -> str:
    """Return one of the ``ACTION_*`` constants for ``path``.

    Classification is purely structural — it looks at ancestor directory
    names and file suffix, never at file contents. The optional ``spec``
    lets a path that *would* be re-encoded fall back to ``ACTION_LINK``
    when its modality is opted out via a ``skip_*`` toggle.
    """
    parts = path.relative_to(src_root).parts[:-1]  # excludes the filename

    # cam_pose: .npz under cam_pose/
    if path.suffix == NPZ_SUFFIX and CAM_POSE_PARENT_DIR in parts:
        if spec is not None and spec.skip_cam_pose:
            return ACTION_LINK
        return ACTION_CAM_POSE

    # PNG variants
    if path.suffix in CONVERT_FROM_SUFFIXES:
        if any(p in WEBP_PARENT_DIRS for p in parts):
            if spec is not None and spec.skip_rgb_webp:
                return ACTION_LINK
            return ACTION_WEBP
        if any(p in PNG_RECOMPRESS_PARENT_DIRS for p in parts):
            if spec is not None and spec.skip_png_recompress:
                return ACTION_LINK
            return ACTION_PNG_RECOMPRESS

    return ACTION_LINK


def _plan_tree(src_root: Path, spec: Optional[ConvertSpec] = None) -> List[Tuple[Path, str]]:
    """Walk ``src_root`` once and return ``[(path, action), ...]``."""
    out: List[Tuple[Path, str]] = []
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        out.append((p, _classify(p, src_root, spec)))
    return out


# --------------------------------------------------------------------------- #
# Worker functions (top-level so ProcessPoolExecutor can pickle them)
# --------------------------------------------------------------------------- #


def _convert_webp_one(
    src_path_s: str,
    dst_path_s: str,
    verify: bool,
    webp_method: int,
) -> Tuple[str, int, int, Optional[str]]:
    """Encode one PNG → lossless WebP. Optionally round-trip verify.

    Returns ``(dst_str, bytes_before, bytes_after, err_or_None)``.
    """
    import numpy as np
    from PIL import Image

    src_path = Path(src_path_s)
    dst_path = Path(dst_path_s)
    try:
        with Image.open(src_path) as src:
            src.load()
            mode = src.mode
            src_arr = np.asarray(src)
        before_b = src_path.stat().st_size
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(src_arr, mode=mode).save(
            dst_path,
            format="WEBP",
            lossless=True,
            quality=100,
            method=webp_method,
        )
        if verify:
            with Image.open(dst_path) as restored:
                restored.load()
                if restored.mode != mode:
                    restored = restored.convert(mode)
                restored_arr = np.asarray(restored)
            if not np.array_equal(src_arr, restored_arr):
                return (
                    dst_path_s,
                    0,
                    0,
                    (
                        f"WebP round-trip mismatch for {src_path} "
                        f"(src shape={src_arr.shape} mode={mode}, "
                        f"restored shape={restored_arr.shape})"
                    ),
                )
        return (dst_path_s, before_b, dst_path.stat().st_size, None)
    except Exception as e:  # noqa: BLE001 — surface any worker error
        return (dst_path_s, 0, 0, f"{src_path}: {e!r}")


def _recompress_png_one(
    src_path_s: str,
    dst_path_s: str,
    verify: bool,
    png_compress_level: int,
) -> Tuple[str, int, int, Optional[str]]:
    """Re-encode one PNG at higher compress_level. Mode/dtype preserved.

    Returns ``(dst_str, bytes_before, bytes_after, err_or_None)``.
    """
    import numpy as np
    from PIL import Image

    src_path = Path(src_path_s)
    dst_path = Path(dst_path_s)
    try:
        with Image.open(src_path) as src:
            src.load()
            src_mode = src.mode
            src_arr = np.asarray(src)
            before_b = src_path.stat().st_size
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # Re-save through the original Image so palette / I;16 / L
            # modes are preserved exactly.
            src.save(dst_path, format="PNG", compress_level=png_compress_level)
        if verify:
            with Image.open(dst_path) as restored:
                restored.load()
                restored_mode = restored.mode
                restored_arr = np.asarray(restored)
            if restored_mode != src_mode:
                return (
                    dst_path_s,
                    0,
                    0,
                    (
                        f"PNG mode drift for {src_path}: "
                        f"src mode={src_mode}, restored mode={restored_mode}"
                    ),
                )
            if not np.array_equal(src_arr, restored_arr):
                return (
                    dst_path_s,
                    0,
                    0,
                    (
                        f"PNG round-trip mismatch for {src_path} "
                        f"(shape={src_arr.shape}, mode={src_mode})"
                    ),
                )
        return (dst_path_s, before_b, dst_path.stat().st_size, None)
    except Exception as e:  # noqa: BLE001
        return (dst_path_s, 0, 0, f"{src_path}: {e!r}")


def _convert_cam_pose_one(
    src_path_s: str,
    dst_path_s: str,
    verify: bool,
) -> Tuple[str, int, int, Optional[str]]:
    """Per-frame ``.npz`` (position + orientation) → ``.npy`` ``(7,) float64``.

    The output is a single float64 vector packing
    ``[x, y, z, qx, qy, qz, qw]``. Per-frame round-trip verification
    asserts every slice is bit-identical to the source npz arrays.

    Returns ``(dst_str, bytes_before, bytes_after, err_or_None)``.
    """
    import numpy as np

    src_path = Path(src_path_s)
    dst_path = Path(dst_path_s)
    try:
        data = np.load(src_path)
        if "position" not in data or "orientation" not in data:
            return (
                dst_path_s,
                0,
                0,
                (
                    f"cam_pose .npz missing required keys at {src_path}: "
                    f"found {list(data.keys())}, need 'position' + 'orientation'"
                ),
            )
        position = data["position"].astype(np.float64)
        orientation = data["orientation"].astype(np.float64)
        if position.shape != (3,) or orientation.shape != (4,):
            return (
                dst_path_s,
                0,
                0,
                (
                    f"cam_pose .npz unexpected shapes at {src_path}: "
                    f"position={position.shape}, orientation={orientation.shape}"
                ),
            )
        pose7 = np.concatenate([position, orientation])  # (7,) float64
        before_b = src_path.stat().st_size
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_path, pose7)
        if verify:
            restored = np.load(dst_path)
            if restored.shape != (7,) or restored.dtype != np.float64:
                return (
                    dst_path_s,
                    0,
                    0,
                    (
                        f"cam_pose .npy round-trip shape/dtype mismatch at "
                        f"{dst_path}: shape={restored.shape}, "
                        f"dtype={restored.dtype}"
                    ),
                )
            if not (
                np.array_equal(restored[:3], position) and np.array_equal(restored[3:], orientation)
            ):
                return (
                    dst_path_s,
                    0,
                    0,
                    f"cam_pose round-trip value mismatch at {src_path}",
                )
        return (dst_path_s, before_b, dst_path.stat().st_size, None)
    except Exception as e:  # noqa: BLE001
        return (dst_path_s, 0, 0, f"{src_path}: {e!r}")


def _link_one(
    src_path_s: str,
    dst_path_s: str,
) -> Tuple[str, int, int, Optional[str]]:
    """Hard-link (or copy if cross-fs) a file into the output tree."""
    src_path = Path(src_path_s)
    dst_path = Path(dst_path_s)
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        try:
            os.link(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)
        sz = dst_path.stat().st_size
        return (dst_path_s, sz, sz, None)
    except Exception as e:  # noqa: BLE001
        return (dst_path_s, 0, 0, f"{src_path}: {e!r}")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _ensure_disjoint(src: Path, out: Path) -> None:
    """Refuse src/out overlap — would otherwise lead to infinite walking."""
    src_r = src.resolve()
    out_r = out.resolve()
    if src_r == out_r:
        raise DatasetError(
            f"src_root and out_root must differ; got {src_r}",
            hint="Pass --out pointing at a fresh directory.",
        )
    try:
        if out_r.is_relative_to(src_r) or src_r.is_relative_to(out_r):
            raise DatasetError(
                f"src_root and out_root overlap: src={src_r}, out={out_r}",
                hint="Pick a sibling directory outside src for --out.",
            )
    except AttributeError:
        # Python <3.9 — relative-to check unsupported; fall back to string prefix.
        if str(out_r).startswith(str(src_r) + os.sep) or str(src_r).startswith(str(out_r) + os.sep):
            raise DatasetError(
                f"src_root and out_root overlap: src={src_r}, out={out_r}",
            ) from None


def _dst_path_for(src_path: Path, action: str, src_root: Path, out_root: Path) -> Path:
    """Return the destination path for ``src_path`` given the action."""
    rel = src_path.relative_to(src_root)
    if action == ACTION_WEBP:
        return out_root / rel.with_suffix(WEBP_SUFFIX)
    if action == ACTION_CAM_POSE:
        return out_root / rel.with_suffix(NPY_SUFFIX)
    # png-recompress and link keep the same filename
    return out_root / rel


def convert_capture_tree(spec: ConvertSpec) -> ConvertResult:
    """Walk ``spec.src_root`` and produce a re-encoded capture tree.

    On success, ``spec.out_root`` contains a complete capture tree where
    rgb/fisheye/ego frames are lossless WebP, depth/mask PNGs are
    re-encoded at maximum compression, and cam_pose frames are
    consolidated to per-frame .npy. Everything else is hard-linked.
    The output tree is a drop-in replacement for the source as input to
    :func:`pack_capture_tree`.
    """
    if not spec.src_root.is_dir():
        raise DatasetError(f"src_root does not exist or is not a directory: {spec.src_root}")
    _ensure_disjoint(spec.src_root, spec.out_root)

    if spec.out_root.exists() and not spec.overwrite_out:
        if any(spec.out_root.iterdir()):
            raise DatasetError(
                f"out_root exists and is non-empty: {spec.out_root}",
                hint="Pass overwrite_out=True or pick a fresh directory.",
            )
    spec.out_root.mkdir(parents=True, exist_ok=True)

    plan = _plan_tree(spec.src_root, spec=spec)
    action_counts = {}
    for _, a in plan:
        action_counts[a] = action_counts.get(a, 0) + 1
    log.info(
        "lossless-convert: scanned %d files under %s; per-action: %s",
        len(plan),
        spec.src_root,
        ", ".join(f"{a}={n}" for a, n in sorted(action_counts.items())),
    )

    result = ConvertResult(files_seen=len(plan))

    if spec.dry_run:
        for action, n in action_counts.items():
            result._stats(action).files = n
        return result

    # Build per-action worklists.
    worklists: dict = {
        ACTION_WEBP: [],
        ACTION_PNG_RECOMPRESS: [],
        ACTION_CAM_POSE: [],
        ACTION_LINK: [],
    }
    for src_path, action in plan:
        dst_path = _dst_path_for(src_path, action, spec.src_root, spec.out_root)
        worklists[action].append((str(src_path), str(dst_path)))

    # Convert each action's worklist in parallel. Each pass aborts on
    # first verification failure so a corrupt output tree cannot escape.
    def run_pass(action: str, fn, *extra_args):
        worklist = worklists[action]
        if not worklist:
            return
        log.info("lossless-convert: %s — %d files", action, len(worklist))
        with ProcessPoolExecutor(max_workers=spec.workers) as ex:
            futures = [ex.submit(fn, src_s, dst_s, *extra_args) for src_s, dst_s in worklist]
            done = 0
            for fut in as_completed(futures):
                _, before_b, after_b, err = fut.result()
                done += 1
                if err:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise DatasetError(
                        f"lossless-convert ({action}) aborted at {done}/{len(worklist)}: {err}",
                        hint=(
                            "Inspect the offending file and the relevant codec "
                            "library version on this machine."
                        ),
                    )
                stats = result._stats(action)
                stats.files += 1
                stats.bytes_before += before_b
                stats.bytes_after += after_b
                if done % 500 == 0:
                    log.info("  ... %d / %d done", done, len(worklist))

    run_pass(ACTION_WEBP, _convert_webp_one, spec.verify, spec.webp_method)
    run_pass(ACTION_PNG_RECOMPRESS, _recompress_png_one, spec.verify, spec.png_compress_level)
    run_pass(ACTION_CAM_POSE, _convert_cam_pose_one, spec.verify)

    # Linking pass: failures here are collected (not fatal mid-pass) and
    # raised at the end so we surface every broken file at once.
    link_failures: List[str] = []
    if worklists[ACTION_LINK]:
        log.info("lossless-convert: link — %d files", len(worklists[ACTION_LINK]))
        with ProcessPoolExecutor(max_workers=spec.workers) as ex:
            futures = [ex.submit(_link_one, s, d) for s, d in worklists[ACTION_LINK]]
            for fut in as_completed(futures):
                _, before_b, after_b, err = fut.result()
                if err:
                    link_failures.append(err)
                    continue
                stats = result._stats(ACTION_LINK)
                stats.files += 1
                stats.bytes_before += before_b
                stats.bytes_after += after_b
    if link_failures:
        result.failures = link_failures
        raise DatasetError(
            f"lossless-convert: {len(link_failures)} link/copy failures; first: {link_failures[0]}",
        )

    return result

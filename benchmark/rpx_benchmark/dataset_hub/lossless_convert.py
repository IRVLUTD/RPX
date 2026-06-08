"""Convert RGB and fisheye PNGs in a capture tree to lossless WebP.

Why
---

PNG and WebP-lossless both reconstruct the encoded pixel array
bit-for-bit. WebP-lossless is just a denser container: on real-world
8-bit RGB photographs it is ~20-30% smaller than the equivalent PNG,
which translates to a ~25 GB saving on the full RPX upload and a
proportional speed-up for users downloading from the HF hub. Because
the format is lossless, every downstream stage of the toolkit (the
manifest builder, the dataset card, the loader, every model adapter)
remains correct without modification: ``PIL.Image.open`` and
``cv2.imread`` both sniff the file header, so the in-memory pixel array
the benchmark sees is identical regardless of whether the bytes on disk
were ``.png`` or ``.webp``.

What gets converted
-------------------

Only ``rgb/`` and ``fisheye/`` modality directories are re-encoded.
Other modalities are *hard-linked* (or copied, on cross-filesystem)
into the output tree unchanged:

* ``depth/*.png``        — 16-bit single-channel. WebP-lossless does
                           not support 16-bit, so depth stays PNG.
* ``masks/*.png``        — palette-/binary-mode PNGs whose semantics
                           rely on PNG behaviour; conversion would
                           change the on-disk channel layout.
* ``cam_pose/*``,
  ``sam2_meta/*``        — non-image data, copied verbatim.

Hard-linking is the default for non-converted files: it costs zero
extra disk space (the inode is shared with the source tree) and keeps
the output tree self-contained for the packer.

Verification
------------

Every converted frame is *round-trip verified* — the script decodes the
freshly-written WebP, compares it against the in-memory source array
with ``numpy.array_equal``, and aborts the whole run on the first
mismatch. The output tree, if conversion succeeds, is proven
bit-faithful by construction.

Usage
-----

::

    from rpx_benchmark.dataset_hub.lossless_convert import (
        ConvertSpec, convert_capture_tree,
    )

    spec = ConvertSpec(
        src_root=Path("DATA"),
        out_root=Path("DATA_webp"),
        workers=8,
    )
    result = convert_capture_tree(spec)
    print(f"saved {result.saved_bytes/1e9:.1f} GB ({result.saved_pct:.1f}%)")
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


# Modality directory names whose 8-bit PNG frames are re-encoded as
# lossless WebP. A file is converted iff its path contains one of these
# segments AND it has a PNG extension.
CONVERTIBLE_MODALITIES = frozenset({"rgb", "fisheye"})

CONVERT_FROM_SUFFIXES = frozenset({".png", ".PNG"})
CONVERTED_SUFFIX = ".webp"


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
    # Benchmarks on real 1920x1080 GoPro photo content show method=4 and
    # method=6 produce byte-identical output sizes (~50% smaller than
    # PNG), but method=6 takes ~10x longer. method=4 is the sweet spot.
    webp_method: int = 4


@dataclass
class ConvertResult:
    """Outputs of :func:`convert_capture_tree`."""

    files_seen: int
    files_converted: int
    files_linked: int
    bytes_before: int
    bytes_after: int
    failures: List[str] = field(default_factory=list)

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


def _classify(path: Path, src_root: Path) -> str:
    """Return ``"convert"`` if the file should be re-encoded, else ``"link"``.

    A file is converted iff (a) its suffix is a PNG variant, and (b) any
    directory between ``src_root`` and the file itself is named in
    :data:`CONVERTIBLE_MODALITIES`.
    """
    if path.suffix not in CONVERT_FROM_SUFFIXES:
        return "link"
    parts = path.relative_to(src_root).parts[:-1]  # excludes the filename
    if any(p in CONVERTIBLE_MODALITIES for p in parts):
        return "convert"
    return "link"


def _plan_tree(src_root: Path) -> List[Tuple[Path, str]]:
    """Walk ``src_root`` once and return ``[(path, action), ...]``."""
    out: List[Tuple[Path, str]] = []
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        out.append((p, _classify(p, src_root)))
    return out


# --------------------------------------------------------------------------- #
# Worker functions (top-level so ProcessPoolExecutor can pickle them)
# --------------------------------------------------------------------------- #


def _convert_one(
    src_path_s: str,
    dst_path_s: str,
    verify: bool,
    webp_method: int,
) -> Tuple[str, int, int, Optional[str]]:
    """Encode one PNG → lossless WebP; optionally round-trip verify.

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
                        f"round-trip mismatch for {src_path} "
                        f"(src shape={src_arr.shape}, "
                        f"restored shape={restored_arr.shape})"
                    ),
                )
        return (dst_path_s, before_b, dst_path.stat().st_size, None)
    except Exception as e:  # noqa: BLE001 — surface any worker error
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
        if str(out_r).startswith(str(src_r) + os.sep) or str(src_r).startswith(
            str(out_r) + os.sep
        ):
            raise DatasetError(
                f"src_root and out_root overlap: src={src_r}, out={out_r}",
            )


def convert_capture_tree(spec: ConvertSpec) -> ConvertResult:
    """Walk ``spec.src_root`` and produce a parallel WebP-lossless tree.

    On success, ``spec.out_root`` contains a complete capture tree where
    every RGB and fisheye frame is a lossless WebP and every other file
    is the original byte-for-byte. The output tree can be passed
    directly to :func:`pack_capture_tree` like any other capture tree.
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

    plan = _plan_tree(spec.src_root)
    n_conv = sum(1 for _, a in plan if a == "convert")
    n_link = sum(1 for _, a in plan if a == "link")
    log.info(
        "lossless-convert: scanned %d files (%d convert, %d link) under %s",
        len(plan),
        n_conv,
        n_link,
        spec.src_root,
    )

    if spec.dry_run:
        return ConvertResult(
            files_seen=len(plan),
            files_converted=n_conv,
            files_linked=n_link,
            bytes_before=0,
            bytes_after=0,
        )

    # Split into work items.
    convert_args: List[Tuple[str, str, bool, int]] = []
    link_args: List[Tuple[str, str]] = []
    for src_path, action in plan:
        rel = src_path.relative_to(spec.src_root)
        if action == "convert":
            dst_path = spec.out_root / rel.with_suffix(CONVERTED_SUFFIX)
            convert_args.append((str(src_path), str(dst_path), spec.verify, spec.webp_method))
        else:
            dst_path = spec.out_root / rel
            link_args.append((str(src_path), str(dst_path)))

    bytes_before = 0
    bytes_after = 0
    failures: List[str] = []

    # Conversion pass: parallel, abort on first verification failure.
    log.info("lossless-convert: encoding %d files with %d workers", len(convert_args), spec.workers)
    with ProcessPoolExecutor(max_workers=spec.workers) as ex:
        futures = [ex.submit(_convert_one, *a) for a in convert_args]
        done = 0
        for fut in as_completed(futures):
            _, before_b, after_b, err = fut.result()
            done += 1
            if err:
                ex.shutdown(wait=False, cancel_futures=True)
                raise DatasetError(
                    f"lossless conversion aborted at file {done}/{len(convert_args)}: {err}",
                    hint=(
                        "Inspect the offending PNG; check that libwebp / Pillow "
                        "are healthy on this machine."
                    ),
                )
            bytes_before += before_b
            bytes_after += after_b
            if done % 500 == 0:
                log.info("  ... %d / %d encoded", done, len(convert_args))

    # Linking pass: parallel, individual failures collected (not fatal).
    log.info("lossless-convert: linking %d files", len(link_args))
    with ProcessPoolExecutor(max_workers=spec.workers) as ex:
        futures = [ex.submit(_link_one, *a) for a in link_args]
        for fut in as_completed(futures):
            _, before_b, after_b, err = fut.result()
            if err:
                failures.append(err)
                continue
            bytes_before += before_b
            bytes_after += after_b

    if failures:
        raise DatasetError(
            f"lossless-convert: {len(failures)} link/copy failures; first: {failures[0]}",
        )

    return ConvertResult(
        files_seen=len(plan),
        files_converted=len(convert_args),
        files_linked=len(link_args),
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        failures=failures,
    )

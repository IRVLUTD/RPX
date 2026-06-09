"""Canonical decode helpers with runtime contracts.

The dataset_hub publication pipeline is fault-proof end-to-end —
encoder verification, tar SHA-256, per-file SHA-256, manifest-declared
extensions. The remaining failure mode is a *consumer-side* one: an
adapter that opens files itself, bypasses the loader's contract
assertions, and silently feeds wrong-dtype data to a model.

The defense is to make the canonical decode path easier to use than
the wrong one. Every module that needs a decoded pixel/value array
calls one of the ``safe_load_*`` helpers in this module. The helpers
encode the *only* correct decode pattern per modality:

* :func:`safe_load_rgb`   — ``Image.open(...).convert("RGB")`` →
  ``(H, W, 3) uint8``. Any deviation raises ``ManifestError``.
* :func:`safe_load_depth` — ``Image.open`` with NO ``.convert()`` so
  ``I;16`` mode is preserved → ``(H, W) uint16``. The classic
  silent-truncation bug (``cv2.imread`` without ``IMREAD_UNCHANGED``
  returning ``uint8``) is impossible inside this helper.
* :func:`safe_load_mask`  — palette / ``I;16`` PNG → ``(H, W) int32``
  preserving integer instance IDs even past 255.
* :func:`safe_load_gray`  — 8-bit grayscale (e.g. fisheye) →
  ``(H, W) uint8``. Cannot widen to RGB.
* :func:`safe_load_pose`  — ``.npz`` (keys ``position`` + ``orientation``)
  or ``.npy`` ``(7,) float64`` packing ``[x, y, z, qx, qy, qz, qw]``.

Adapters and per-task scripts that decode files directly *must* go
through these helpers. ``RPXDataset._load_*`` in :mod:`loader`
delegates to them as well, so there is one and only one place per
modality where decode policy lives.

If you find yourself reaching for ``cv2.imread`` or ``Image.open``
inside an adapter, stop — add a ``safe_load_*`` for the new modality
here instead. This module is short by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from .exceptions import ManifestError

__all__ = [
    "safe_load_rgb",
    "safe_load_depth",
    "safe_load_mask",
    "safe_load_gray",
    "safe_load_pose",
]


def safe_load_rgb(path: str | Path) -> np.ndarray:
    """Decode an RGB image to ``(H, W, 3) uint8``.

    Accepts PNG, WebP-lossless, or any other format Pillow recognises.
    The decoded numpy array is bit-identical regardless of container
    format, so adapters never need to branch on file suffix.

    Raises
    ------
    ManifestError
        If the decoded array is not ``(H, W, 3) uint8``.
    """
    p = Path(path)
    with Image.open(p) as im:
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ManifestError(
            f"safe_load_rgb({p}): expected (H, W, 3) uint8, "
            f"got shape {arr.shape} dtype {arr.dtype}",
        )
    return arr


def safe_load_depth(path: str | Path) -> np.ndarray:
    """Decode a 16-bit single-channel depth PNG to ``(H, W) uint16``.

    Does NOT scale or cast to float — that's the caller's job (the
    canonical loader divides by 1000 to convert mm → m). This function's
    only responsibility is decoding without losing bits.

    Critical: never calls ``.convert()``, so PIL's ``I;16`` mode is
    preserved. ``cv2.imread`` without ``IMREAD_UNCHANGED`` would
    silently return ``uint8`` here; this function refuses that path.

    Raises
    ------
    ManifestError
        If the decoded array is not ``(H, W) uint16``.
    """
    p = Path(path)
    with Image.open(p) as im:
        arr = np.asarray(im)
    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ManifestError(
            f"safe_load_depth({p}): expected (H, W) uint16 (16-bit PNG), "
            f"got shape {arr.shape} dtype {arr.dtype}. "
            f"A uint8 result here would silently truncate every depth "
            f"value to 8-bit resolution.",
        )
    return arr


def safe_load_mask(path: str | Path) -> np.ndarray:
    """Decode an instance-segmentation mask to ``(H, W) int32``.

    Accepts palette PNG (mode ``P``), 16-bit grayscale (``I;16``), or
    8-bit grayscale (``L``). The int32 cast preserves IDs >255 that
    palette/I;16 modes can carry.

    Raises
    ------
    ManifestError
        If the decoded array is not ``(H, W) int32``.
    """
    p = Path(path)
    with Image.open(p) as im:
        arr = np.array(im, dtype=np.int32)
    if arr.ndim != 2 or arr.dtype != np.int32:
        raise ManifestError(
            f"safe_load_mask({p}): expected (H, W) int32, got shape {arr.shape} dtype {arr.dtype}",
        )
    return arr


def safe_load_gray(path: str | Path) -> np.ndarray:
    """Decode an 8-bit grayscale image (fisheye) to ``(H, W) uint8``.

    Raises
    ------
    ManifestError
        If the decoded array is not ``(H, W) uint8``.
    """
    p = Path(path)
    with Image.open(p) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ManifestError(
            f"safe_load_gray({p}): expected (H, W) uint8, got shape {arr.shape} dtype {arr.dtype}",
        )
    return arr


def safe_load_pose(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a T265 pose to ``(position[3], orientation[4])`` float64.

    Accepts both on-disk formats:

    * Legacy ``.npz`` with ``position`` and ``orientation`` keys.
    * v2 ``.npy`` carrying a ``(7,) float64`` vector packing
      ``[x, y, z, qx, qy, qz, qw]``.

    Returns
    -------
    (position, orientation)
        ``position`` is ``(3,) float64`` (metres).
        ``orientation`` is ``(4,) float64`` (xyzw quaternion).

    Raises
    ------
    ManifestError
        If the file's shape/dtype does not match the contract.
    """
    p = Path(path)
    data = np.load(p)
    if p.suffix.lower() == ".npy":
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (7,):
            raise ManifestError(
                f"safe_load_pose({p}): .npy must be shape (7,) float64; got {arr.shape}",
            )
        return arr[:3], arr[3:]
    # .npz fallback
    if "position" not in data or "orientation" not in data:
        raise ManifestError(
            f"safe_load_pose({p}): .npz missing required keys 'position' and 'orientation'",
        )
    position = data["position"].astype(np.float64)
    orientation = data["orientation"].astype(np.float64)
    if position.shape != (3,) or orientation.shape != (4,):
        raise ManifestError(
            f"safe_load_pose({p}): unexpected shapes "
            f"position={position.shape}, orientation={orientation.shape}",
        )
    return position, orientation

"""Per-frame alignment of a predicted depth map to GT scale.

Used by both:

* the runner's primary-metric path (``BatchedDepthBenchmarkModel.predict``
  applies the adapter's ``native_alignment`` before the runner sees the
  prediction, so result.json's ``aggregated.absrel`` is meaningful for
  relative-depth models like Marigold / MiDaS); and
* the post-processor's metric basket (``comprehensive_depth_metrics``)
  which iterates alignment modes per-frame.

Single source of truth for the alignment math so a fix in one path
applies everywhere. Modes match the standard mono-depth-evaluation
literature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..exceptions import ConfigError


__all__ = [
    "DEPTH_MIN_M",
    "DEPTH_MAX_M",
    "ALIGNMENT_MODES",
    "align_pred_to_gt",
    "default_valid_mask",
]


#: D435 sensor working range. Pixels outside this range are excluded
#: from the alignment fit (and from every metric — see
#: ``comprehensive_depth_metrics._valid``). Kept here so any future
#: change is one-place.
DEPTH_MIN_M: float = 0.3
DEPTH_MAX_M: float = 5.0

#: Supported alignment mode names.
ALIGNMENT_MODES = frozenset({"none", "median", "ls_affine", "ls_disparity"})


def default_valid_mask(pred: np.ndarray, gt_m: np.ndarray) -> np.ndarray:
    """Standard validity gate used by every alignment + metric step.

    A pixel enters the alignment fit (and the downstream metric) iff:

    * GT is finite (D435 holes encoded as NaN or 0 mm are excluded);
    * GT is in the sensor working range (``DEPTH_MIN_M`` … ``DEPTH_MAX_M``);
    * Prediction is finite;
    * Prediction is positive.
    """
    return (
        np.isfinite(gt_m) & (gt_m > DEPTH_MIN_M) & (gt_m < DEPTH_MAX_M)
        & np.isfinite(pred) & (pred > 0)
    )


def align_pred_to_gt(
    pred: np.ndarray,
    gt_m: np.ndarray,
    mode: str,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit a per-frame transform from raw prediction to GT scale.

    Parameters
    ----------
    pred
        Predicted depth (any 2-D float array; semantics depend on the
        model — for relative models it's up-to-scale, for metric models
        it's already in metres).
    gt_m
        Ground-truth depth in metres.
    mode
        One of:

        * ``"none"``         — identity. Use for native-metric models.
        * ``"median"``       — ``pred *= median(gt) / median(pred)``;
          cheap rescale, single dof.
        * ``"ls_affine"``    — least-squares fit ``a·pred + b`` to GT
          in depth space. Two dof; standard alignment for relative
          and up-to-scale models (DA-V2 relative, MiDaS, Marigold,
          Lotus).
        * ``"ls_disparity"`` — least-squares fit in disparity (1/d)
          space. Useful when the model's native output is disparity
          (some MiDaS variants).
    valid
        Optional pre-computed validity mask. If not provided, uses
        :func:`default_valid_mask`.

    Returns
    -------
    np.ndarray
        Aligned prediction with the same shape and dtype as ``pred``.

    Raises
    ------
    ConfigError
        If ``mode`` isn't in :data:`ALIGNMENT_MODES`.
    """
    if mode == "none" or pred is None or gt_m is None:
        return pred
    if mode not in ALIGNMENT_MODES:
        raise ConfigError(
            f"unknown alignment mode {mode!r}",
            hint=f"Expected one of: {sorted(ALIGNMENT_MODES)}.",
        )
    if valid is None:
        valid = default_valid_mask(pred, gt_m)
    if not valid.any():
        return pred

    p = pred[valid].astype(np.float64)
    g = gt_m[valid].astype(np.float64)

    if mode == "median":
        scale = float(np.median(g) / max(np.median(p), 1e-9))
        return (pred * scale).astype(np.float32)

    if mode == "ls_affine":
        A = np.column_stack([p, np.ones_like(p)])
        coef, *_ = np.linalg.lstsq(A, g, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        return (a * pred + b).astype(np.float32)

    if mode == "ls_disparity":
        eps = 1e-6
        inv_p = 1.0 / np.maximum(p, eps)
        inv_g = 1.0 / np.maximum(g, eps)
        A = np.column_stack([inv_p, np.ones_like(inv_p)])
        coef, *_ = np.linalg.lstsq(A, inv_g, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        inv_pred = 1.0 / np.maximum(pred.astype(np.float64), eps)
        return (1.0 / np.maximum(a * inv_pred + b, eps)).astype(np.float32)

    # Unreachable thanks to the ALIGNMENT_MODES gate above; raises
    # defensively in case the gate is loosened later.
    raise ConfigError(
        f"unknown alignment mode {mode!r}",
        hint="Add the mode to ALIGNMENT_MODES and implement its branch.",
    )

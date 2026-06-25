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
    "align_pred_to_gt_pooled",
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
        np.isfinite(gt_m)
        & (gt_m > DEPTH_MIN_M)
        & (gt_m < DEPTH_MAX_M)
        & np.isfinite(pred)
        & (pred > 0)
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


def align_pred_to_gt_pooled(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    mode: str,
    valid_seq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pooled scale+shift fit over an entire ``(scene, phase)`` cell.

    Paper §3.3 (line 62) prescribes **per-scene** scale-and-shift
    alignment for affine-invariant depth models (Lotus-2, FE2E, etc.).
    "Per-scene" here means one ``(s, t)`` per ``(scene, phase)`` cell,
    pooling every valid pixel across all frames in that cell into a
    single least-squares solve. The same scope applies to:

    * D1-V (paper task ``video_depth``) — pool over ``(T, H, W)``.
    * D1-F (paper task ``monocular_depth``) — pool over the phase's
      ~250 frames, with each ``(H, W)`` slice contributing its valid
      pixels.

    Per-frame alignment would let each frame find its own ``(s, t)``,
    artificially smoothing over any temporal scale drift the model has
    within a phase — hiding a real failure mode. Per-scene preserves
    that drift and reports it through the per-frame metrics (which are
    then averaged into the cell).

    Parameters
    ----------
    pred_seq
        Predicted depth for the cell. Either ``(T, H, W)`` (a phase
        clip) or a list of per-frame arrays stacked along axis 0. Any
        2-D-shape-compatible array works; only the first axis is
        treated as the frame index for masking purposes.
    gt_seq
        Ground-truth depth in metres, same shape as ``pred_seq``.
    mode
        One of :data:`ALIGNMENT_MODES`. ``"none"`` returns ``pred_seq``
        unchanged. ``"median"`` / ``"ls_affine"`` / ``"ls_disparity"``
        all pool valid pixels across frames before solving.
    valid_seq
        Optional pre-computed mask, same shape as ``pred_seq``. If
        absent, computed per-frame with :func:`default_valid_mask` and
        OR'd to form the pooled mask.

    Returns
    -------
    np.ndarray
        Aligned prediction with the same shape and dtype as
        ``pred_seq``. When the pooled valid set is empty, returns
        ``pred_seq`` unchanged.

    Notes
    -----
    A native-metric model gets ``mode="none"`` (or any mode where the
    solver finds ``s ≈ 1, t ≈ 0`` because the prediction already
    matches the GT distribution). An affine-invariant model gets
    ``mode="ls_affine"``. The runner is expected to dispatch on the
    adapter's ``depth_output_kind`` attribute.
    """
    if mode == "none":
        return pred_seq
    if mode not in ALIGNMENT_MODES:
        raise ConfigError(
            f"unknown alignment mode {mode!r}",
            hint=f"Expected one of: {sorted(ALIGNMENT_MODES)}.",
        )
    if pred_seq.shape != gt_seq.shape:
        raise ConfigError(
            f"align_pred_to_gt_pooled: pred_seq.shape={pred_seq.shape} "
            f"does not match gt_seq.shape={gt_seq.shape}",
            hint="Both inputs must be the same shape (typically (T, H, W)).",
        )
    if valid_seq is None:
        # Compute per-frame masks and stack — equivalent to computing
        # one mask over the whole pooled array, but reuses the existing
        # default_valid_mask policy without making it sequence-aware.
        valid_seq = default_valid_mask(pred_seq, gt_seq)
    if valid_seq.shape != pred_seq.shape:
        raise ConfigError(
            f"align_pred_to_gt_pooled: valid_seq.shape={valid_seq.shape} "
            f"does not match pred_seq.shape={pred_seq.shape}",
            hint="The validity mask must be the same shape as the depth tensors.",
        )
    if not valid_seq.any():
        return pred_seq

    # Flatten across (T, H, W) and let the existing per-frame solver
    # do the math. We tag the prediction's dtype so the caller's
    # contract (float32 / float64) is preserved.
    p = pred_seq[valid_seq].astype(np.float64)
    g = gt_seq[valid_seq].astype(np.float64)
    if p.size == 0:
        return pred_seq

    if mode == "median":
        scale = float(np.median(g) / max(np.median(p), 1e-9))
        return (pred_seq * scale).astype(pred_seq.dtype)

    if mode == "ls_affine":
        A = np.column_stack([p, np.ones_like(p)])
        coef, *_ = np.linalg.lstsq(A, g, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        return (a * pred_seq + b).astype(pred_seq.dtype)

    if mode == "ls_disparity":
        eps = 1e-6
        inv_p = 1.0 / np.maximum(p, eps)
        inv_g = 1.0 / np.maximum(g, eps)
        A = np.column_stack([inv_p, np.ones_like(inv_p)])
        coef, *_ = np.linalg.lstsq(A, inv_g, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        inv_pred = 1.0 / np.maximum(pred_seq.astype(np.float64), eps)
        return (1.0 / np.maximum(a * inv_pred + b, eps)).astype(pred_seq.dtype)

    raise ConfigError(  # pragma: no cover — guarded by ALIGNMENT_MODES gate
        f"unknown alignment mode {mode!r}",
        hint="Add the mode to ALIGNMENT_MODES and implement its branch.",
    )

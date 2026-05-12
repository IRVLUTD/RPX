"""Metrics for the RPX active-perception task (Task #11).

The active-perception model proposes a next-best pose ``T̂_next``
given K context views; the oracle picks ``T*`` from the remaining
trajectory. This module scores ``T̂_next`` against ``T*`` and, when
both an NVS adapter and target GT are available, against the
*downstream-utility* PUS-style ratio described in the proposal at
``benchmark/docs/methods/active_perception.md``.

Three primary numbers per sample
--------------------------------

* ``pose_geodesic_deg``     — rotation angle between R̂_next and R*.
* ``translation_l2_m``      — ‖t̂_next − t*‖ in metres.
* ``pus_active``            — optional. Downstream-task accuracy at
  ``T̂_next`` divided by the same at ``T*``. ``None`` when the caller
  doesn't supply the downstream evaluator.

Higher-level aggregations
-------------------------

The runner aggregates these into the standard three-axis layout:

* Axis 1 (task perf):  ``pose_geodesic_deg`` median + IQR,
  ``translation_l2_m`` median + IQR, ``pus_active`` mean when present.
* Axis 2 (robustness): by_sample_type / by_difficulty / by_context_count
  / cross_phase_delta — same stratifiers as the NVS axis.
* Axis 3 (cost):       FLOPs / params / latency to propose one pose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Per-sample primitives
# ─────────────────────────────────────────────────────────────────────────────


def pose_geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Rotation-only geodesic distance in degrees.

    Computes ``θ = arccos((trace(R̂ᵀR*) − 1) / 2)`` with a safe clip,
    returns degrees. Numerically stable for the near-identity case
    (acos of values slightly outside ``[-1, 1]`` due to fp noise).
    """
    R_pred = np.asarray(R_pred, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        from .exceptions import MetricError  # noqa: PLC0415

        raise MetricError(
            f"expected (3, 3) rotation, got {R_pred.shape} vs {R_gt.shape}",
            hint="pose_geodesic_deg consumes the top-left rotation block "
            "of a 4×4 SE(3) pose. If you have a full pose, pass `T[:3, :3]`.",
        )
    rel = R_pred.T @ R_gt
    cos_theta = (np.trace(rel) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def translation_l2_m(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """L2 distance between predicted and oracle camera centres, in metres."""
    t_pred = np.asarray(t_pred, dtype=np.float64).reshape(3)
    t_gt = np.asarray(t_gt, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(t_pred - t_gt))


def pose_distance_se3(T_pred: np.ndarray, T_oracle: np.ndarray) -> Dict[str, float]:
    """Convenience wrapper — gives both rotation and translation errors
    from two 4×4 SE(3) matrices in one call."""
    T_pred = np.asarray(T_pred, dtype=np.float64)
    T_oracle = np.asarray(T_oracle, dtype=np.float64)
    if T_pred.shape != (4, 4) or T_oracle.shape != (4, 4):
        from .exceptions import MetricError  # noqa: PLC0415

        raise MetricError(
            f"expected (4, 4) SE(3) poses, got {T_pred.shape} vs {T_oracle.shape}",
            hint="pose_distance_se3 takes full camera-to-world matrices. "
            "If you have separate (R, t), wrap them into a 4×4 first.",
        )
    return {
        "pose_geodesic_deg": pose_geodesic_deg(T_pred[:3, :3], T_oracle[:3, :3]),
        "translation_l2_m":  translation_l2_m(T_pred[:3, 3], T_oracle[:3, 3]),
    }


def pus_active(
    metric_at_pred: float,
    metric_at_oracle: float,
    higher_is_better: bool = True,
) -> float:
    """Perceptual Utility Score for active perception.

    ``PUS_active = metric(T̂_next) / metric(T*)`` for higher-is-better
    metrics (PSNR, mIoU, δ-acc); inverted ratio for lower-is-better
    (AbsRel, error degrees). Clipped to ``[0, 1]``. A 1.0 means the
    learned NBV reaches oracle-level downstream utility; 0.5 means
    half of it.

    Returns ``nan`` when either metric is non-finite or oracle is zero
    (degenerate — caller should drop those rows from aggregations).
    """
    if not np.isfinite(metric_at_pred) or not np.isfinite(metric_at_oracle):
        return float("nan")
    if metric_at_oracle == 0:
        return float("nan")
    if higher_is_better:
        ratio = metric_at_pred / metric_at_oracle
    else:
        ratio = metric_at_oracle / metric_at_pred  # smaller error = better
    return float(np.clip(ratio, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample evaluation (call-once contract for the runner)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_active_perception_sample(
    T_pred: np.ndarray,
    T_oracle: np.ndarray,
    metric_at_pred: Optional[float] = None,
    metric_at_oracle: Optional[float] = None,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """Score one active-perception sample. Mirrors
    ``rpx_benchmark.nvs_eval.evaluate_single_sample`` in shape."""
    out: Dict[str, float] = dict(pose_distance_se3(T_pred, T_oracle))
    if metric_at_pred is not None and metric_at_oracle is not None:
        out["pus_active"] = pus_active(
            metric_at_pred, metric_at_oracle, higher_is_better=higher_is_better
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sweep aggregation — three-axis output
# ─────────────────────────────────────────────────────────────────────────────


def _median_iqr(values: Sequence[float]) -> Dict[str, float]:
    """Median + interquartile range for a 1-D sample. Used as the
    primary location/scale stats for pose-geodesic distributions,
    which are typically heavy-tailed (mean is misleading)."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "q25":    float(np.quantile(arr, 0.25)),
        "q75":    float(np.quantile(arr, 0.75)),
        "n":      float(len(arr)),
    }


def evaluate_active_perception(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample rows into the three-axis report.

    Stratifies by ``sample_type``, ``difficulty`` and ``n_context`` —
    same stratifiers the NVS axis uses, so a paper figure can plot
    both axes on the same x-axis without re-shaping.
    """
    if not per_sample:
        return {}

    pose_geo = [s["pose_geodesic_deg"] for s in per_sample if "pose_geodesic_deg" in s]
    trans_l2 = [s["translation_l2_m"] for s in per_sample if "translation_l2_m" in s]
    pus_vals = [s["pus_active"] for s in per_sample if "pus_active" in s and np.isfinite(s["pus_active"])]

    aggregated: Dict[str, Any] = {
        "pose_geodesic_deg": _median_iqr(pose_geo),
        "translation_l2_m":  _median_iqr(trans_l2),
    }
    if pus_vals:
        aggregated["pus_active"] = {
            "mean":   float(np.mean(pus_vals)),
            "median": float(np.median(pus_vals)),
            "n":      float(len(pus_vals)),
        }

    def _bucket(field: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        keys = sorted({s.get(field) for s in per_sample if s.get(field) is not None})
        for k in keys:
            subset = [s for s in per_sample if s.get(field) == k]
            geos = [s["pose_geodesic_deg"] for s in subset if "pose_geodesic_deg" in s]
            tls = [s["translation_l2_m"] for s in subset if "translation_l2_m" in s]
            out[str(k)] = {
                "n_samples":         float(len(subset)),
                "pose_geodesic_deg": _median_iqr(geos),
                "translation_l2_m":  _median_iqr(tls),
            }
        return out

    return {
        "n_samples":         len(per_sample),
        "aggregated":        aggregated,
        "by_sample_type":    _bucket("sample_type"),
        "by_difficulty":     _bucket("difficulty"),
        "by_context_count":  _bucket("n_context"),
    }


__all__ = [
    "pose_geodesic_deg",
    "translation_l2_m",
    "pose_distance_se3",
    "pus_active",
    "evaluate_active_perception_sample",
    "evaluate_active_perception",
]

"""RPX-RCPE evaluation metrics — novel + standard.

Consumes per-pair results (from the runner) enriched with the
metadata fields that :class:`PosePairGenerator` attaches
(``pair_type``, ``rotation_bin``, ``chain_id``, ``chain_position``).

Standard
--------
- ``rotation_error_deg``, ``translation_error_m`` — already computed
  per-sample by ``rpx_benchmark.metrics.pose``.
- ``translation_angular_deg`` — scale-invariant direction error.
- ``pose_error_max_deg`` — max(rot, trans_angular).
- ``AUC@5° / @10° / @20°`` — standard pose AUC.

Novel (unique to RPX-RCPE)
--------------------------
- **Metric AUC@(θ°, d cm)** — joint threshold on rotation AND metric
  translation L2. Only meaningful with metric-scale GT (MegaDepth
  can't do this).
- **Cross-phase Δ** — performance drop from intra-phase to cross-phase
  pairs at matched rotation bins. Measures scene-change robustness.
- **Temporal drift** — accumulated rotation/translation error over a
  chain. Measures stability for SLAM / trajectory integration.
- **Per-bin / per-type breakdowns** — full disaggregation by pair type
  and rotation difficulty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

CANONICAL_POSE_METRICS = ("auc_5deg", "auc_10deg", "auc_20deg")
PHASE_LABELS = {0: "clutter", 1: "interaction", 2: "clean"}
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

# ─────────────────────────────────────────────────────────────────────────────
# Per-pair error functions (pure numpy, no framework deps)
# ─────────────────────────────────────────────────────────────────────────────

def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_rel = R_pred @ R_gt.T
    cos_t = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def translation_l2(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_pred - t_gt))


def translation_angular_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    n_pred = np.linalg.norm(t_pred)
    n_gt = np.linalg.norm(t_gt)
    if n_pred < 1e-9 or n_gt < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(t_pred, t_gt) / (n_pred * n_gt), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ─────────────────────────────────────────────────────────────────────────────
# AUC computation
# ─────────────────────────────────────────────────────────────────────────────

def auc_at_thresholds(
    errors: np.ndarray,
    thresholds: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict[str, float]:
    """Fraction-of-pairs ≤ threshold, integrated (AUC) up to each threshold.

    Returns ``{"auc@5": ..., "auc@10": ..., "auc@20": ...}`` etc.
    """
    errors = np.sort(errors)
    n = len(errors)
    out: Dict[str, float] = {}
    for th in thresholds:
        # Trapezoidal AUC of the recall curve from 0 to th
        bins = np.linspace(0, th, num=100)
        recall = np.array([(errors <= b).sum() / max(n, 1) for b in bins])
        auc = float(_trapezoid(recall, bins) / th)
        out[f"auc@{th:.0f}"] = auc
    return out


def metric_auc(
    rot_errors: np.ndarray,
    trans_errors_m: np.ndarray,
    rot_thresholds: Sequence[float] = (5.0, 10.0, 20.0),
    trans_thresholds_cm: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict[str, float]:
    """Joint AUC: fraction of pairs with rot ≤ θ AND trans ≤ d.

    Novel metric — only possible with metric-scale GT translation.
    Returns e.g. ``{"metric_auc@(5deg,5cm)": 0.32, ...}``.
    """
    out: Dict[str, float] = {}
    for th_r in rot_thresholds:
        for th_t in trans_thresholds_cm:
            th_t_m = th_t / 100.0
            frac = float(np.mean(
                (rot_errors <= th_r) & (trans_errors_m <= th_t_m)
            ))
            out[f"metric_auc@({th_r:.0f}deg,{th_t:.0f}cm)"] = frac
    return out


def metric_auc_curve(
    rot_errors_deg: np.ndarray,
    trans_errors_m: np.ndarray,
    max_rotation_deg: float = 20.0,
    max_translation_cm: float = 20.0,
) -> float:
    """Joint rotation/metric-translation AUC used as the paper's M-AUC.

    At curve position ``u`` a pair succeeds only when both
    ``rotation <= u*max_rotation_deg`` and
    ``translation <= u*max_translation_cm``. Integrating joint recall over
    ``u in [0, 1]`` produces one bounded higher-is-better number.
    """
    rot = np.asarray(rot_errors_deg, dtype=np.float64)
    trans_cm = np.asarray(trans_errors_m, dtype=np.float64) * 100.0
    valid = np.isfinite(rot) & np.isfinite(trans_cm)
    if not np.any(valid):
        return 0.0
    u = np.linspace(0.0, 1.0, 101)
    recall = np.array([
        np.mean(
            (rot[valid] <= value * max_rotation_deg)
            & (trans_cm[valid] <= value * max_translation_cm)
        )
        for value in u
    ])
    return float(_trapezoid(recall, u))


def canonical_metric_vector(
    per_pair: List[Dict[str, Any]],
    *,
    metric_translation_available: bool = True,
) -> Dict[str, float | None]:
    """Return the active D6 K=3 angular vector for one pose cell.

    The three angular AUCs use ``max(rotation error, translation-direction
    error)``. Metric-translation M-AUC is intentionally excluded from the
    active protocol so metric- and up-to-scale adapters remain comparable.
    """
    valid = [
        row for row in per_pair
        if np.isfinite(row.get("pose_error_max_deg", np.nan))
    ]
    if not valid:
        return {key: None for key in CANONICAL_POSE_METRICS}
    pose_error = np.asarray(
        [row["pose_error_max_deg"] for row in valid], dtype=np.float64
    )
    auc = auc_at_thresholds(pose_error, (5.0, 10.0, 20.0))
    values: Dict[str, float | None] = {
        "auc_5deg": auc["auc@5"],
        "auc_10deg": auc["auc@10"],
        "auc_20deg": auc["auc@20"],
    }
    return values


def build_rcpe_cells(
    per_pair: List[Dict[str, Any]],
    *,
    model_name: str,
    difficulty: str,
    metric_translation_available: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build canonical per-(scene, phase) D6 cells from intra-phase pairs.

    Cross-phase pairs and temporal chains are diagnostics and intentionally
    do not enter the phase MANOVA/JEDI cells. Their results remain in
    ``cross_phase_delta`` and ``temporal_drift`` respectively.
    """
    from .cell_log import cell_from_metrics

    buckets: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for row in per_pair:
        if row.get("pair_type") != "intra_phase":
            continue
        try:
            phase = int(row.get("phase"))
        except (TypeError, ValueError):
            continue
        if phase not in PHASE_LABELS:
            continue
        scene = str(row.get("scene_id") or "")
        if not scene:
            continue
        buckets.setdefault((scene, phase), []).append(row)

    cells: List[Dict[str, Any]] = []
    analysis_rows: List[Dict[str, Any]] = []
    for (scene, phase), rows in sorted(buckets.items()):
        metrics = canonical_metric_vector(
            rows,
            metric_translation_available=metric_translation_available,
        )
        phase_label = PHASE_LABELS[phase]
        cells.append(cell_from_metrics(
            metrics,
            model_name=model_name,
            task="relative_camera_pose",
            scene_id=scene,
            phase=phase_label,
            n_samples=len(rows),
            difficulty=difficulty,
            metric_keys=CANONICAL_POSE_METRICS,
        ))
        analysis_rows.append({
            "id": f"{scene}_{phase_label}",
            "scene": scene,
            "phase": phase_label,
            "difficulty": difficulty,
            **metrics,
        })
    return cells, analysis_rows


# ─────────────────────────────────────────────────────────────────────────────
# Disaggregated breakdowns
# ─────────────────────────────────────────────────────────────────────────────

def _group_mean(
    per_pair: List[Dict[str, Any]],
    group_key: str,
    metric_keys: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Group per-pair results by ``group_key`` and compute means."""
    groups: Dict[str, List[Dict]] = {}
    for p in per_pair:
        g = str(p.get(group_key, "unknown"))
        groups.setdefault(g, []).append(p)

    out: Dict[str, Dict[str, float]] = {}
    for g, items in sorted(groups.items()):
        means: Dict[str, float] = {"n_pairs": float(len(items))}
        for k in metric_keys:
            vals = [item[k] for item in items if k in item]
            if vals:
                means[k] = float(np.mean(vals))
        out[g] = means
    return out


def per_bin_breakdown(per_pair: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate by rotation_bin (easy / medium / hard / extreme)."""
    return _group_mean(per_pair, "rotation_bin", [
        "rotation_error_deg", "translation_error_m",
        "translation_angular_deg", "pose_error_max_deg",
    ])


def per_type_breakdown(per_pair: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate by pair_type (intra_phase / cross_phase / temporal_chain)."""
    return _group_mean(per_pair, "pair_type", [
        "rotation_error_deg", "translation_error_m",
        "translation_angular_deg", "pose_error_max_deg",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-phase Δ
# ─────────────────────────────────────────────────────────────────────────────

def cross_phase_delta(
    per_pair: List[Dict[str, Any]],
    metric_key: str = "rotation_error_deg",
) -> Dict[str, float]:
    """Performance drop from intra-phase to cross-phase at matched bins.

    Returns ``{"delta_easy": ..., "delta_medium": ..., ...}`` where
    delta = mean(cross) - mean(intra) for the same rotation bin.
    Positive = cross-phase is harder (expected).
    """
    intra: Dict[str, List[float]] = {}
    cross: Dict[str, List[float]] = {}
    for p in per_pair:
        pt = p.get("pair_type")
        rb = p.get("rotation_bin")
        if rb is None or metric_key not in p:
            continue
        val = p[metric_key]
        if pt == "intra_phase":
            intra.setdefault(rb, []).append(val)
        elif pt == "cross_phase":
            cross.setdefault(rb, []).append(val)

    out: Dict[str, float] = {}
    for bin_name in ("easy", "medium", "hard", "extreme"):
        if bin_name in intra and bin_name in cross:
            delta = float(np.mean(cross[bin_name]) - np.mean(intra[bin_name]))
            out[f"delta_{bin_name}"] = delta
    if intra and cross:
        all_intra = [v for vs in intra.values() for v in vs]
        all_cross = [v for vs in cross.values() for v in vs]
        out["delta_overall"] = float(np.mean(all_cross) - np.mean(all_intra))
    return out


def cross_phase_auc_delta(
    per_pair: List[Dict[str, Any]],
    thresholds: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict[str, float]:
    """Matched-bin AUC loss from intra-phase to Clutter/Clean pairs.

    Positive means cross-phase is worse. The overall value is the mean of
    available bin/threshold deltas, so lower is better as in the paper table.
    """
    deltas: Dict[str, float] = {}
    all_values = []
    for bin_name in ("easy", "medium", "hard", "extreme"):
        intra = [
            float(row["pose_error_max_deg"])
            for row in per_pair
            if row.get("pair_type") == "intra_phase"
            and row.get("rotation_bin") == bin_name
            and np.isfinite(row.get("pose_error_max_deg", np.nan))
        ]
        cross = [
            float(row["pose_error_max_deg"])
            for row in per_pair
            if row.get("pair_type") == "cross_phase"
            and row.get("rotation_bin") == bin_name
            and np.isfinite(row.get("pose_error_max_deg", np.nan))
        ]
        if not intra or not cross:
            continue
        intra_auc = auc_at_thresholds(np.asarray(intra), thresholds)
        cross_auc = auc_at_thresholds(np.asarray(cross), thresholds)
        for threshold in thresholds:
            key = f"auc@{threshold:.0f}"
            value = float(intra_auc[key] - cross_auc[key])
            deltas[f"delta_{bin_name}_{key}"] = value
            all_values.append(value)
    if all_values:
        deltas["delta_overall"] = float(np.mean(all_values))
    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# Temporal drift
# ─────────────────────────────────────────────────────────────────────────────

def temporal_drift(
    per_pair: List[Dict[str, Any]], *, metric_translation_available: bool = True
) -> Dict[str, Any]:
    """Accumulated error along temporal chains.

    Groups chain pairs by ``chain_id``, orders by ``chain_position``,
    and computes cumulative rotation + translation error.

    Returns ``{"chains": [...], "mean_drift_rot_deg": ...,
    "mean_drift_trans_m": ...}``.
    """
    chains: Dict[str, List[Dict]] = {}
    for p in per_pair:
        if p.get("pair_type") != "temporal_chain":
            continue
        cid = (p.get("metadata") or p).get("chain_id")
        if cid is None:
            continue
        chains.setdefault(cid, []).append(p)

    chain_results = []
    for cid, pairs in sorted(chains.items()):
        pairs.sort(key=lambda x: (x.get("metadata") or x).get("chain_position", 0))
        pred_total = np.eye(4, dtype=np.float64)
        gt_total = np.eye(4, dtype=np.float64)
        legacy_cum_rot = 0.0
        legacy_cum_trans = 0.0
        steps = []
        for p in pairs:
            if all(
                key in p
                for key in (
                    "pred_rotation", "pred_translation",
                    "gt_rotation", "gt_translation",
                )
            ):
                pred_step = np.eye(4, dtype=np.float64)
                gt_step = np.eye(4, dtype=np.float64)
                pred_step[:3, :3] = np.asarray(p["pred_rotation"])
                pred_step[:3, 3] = np.asarray(p["pred_translation"])
                gt_step[:3, :3] = np.asarray(p["gt_rotation"])
                gt_step[:3, 3] = np.asarray(p["gt_translation"])
                pred_total = pred_total @ pred_step
                gt_total = gt_total @ gt_step
                cum_rot = rotation_error_deg(pred_total[:3, :3], gt_total[:3, :3])
                cum_trans = translation_l2(pred_total[:3, 3], gt_total[:3, 3])
            else:
                legacy_cum_rot += p.get("rotation_error_deg", 0.0)
                legacy_cum_trans += p.get("translation_error_m", 0.0)
                cum_rot = legacy_cum_rot
                cum_trans = legacy_cum_trans
            steps.append({
                "position": (p.get("metadata") or p).get("chain_position", 0),
                "rotation_error_deg": p.get("rotation_error_deg", 0.0),
                "translation_error_m": (
                    p.get("translation_error_m", 0.0)
                    if metric_translation_available else None
                ),
                "cumulative_rotation_error": cum_rot,
                "cumulative_translation_error": (
                    cum_trans if metric_translation_available else None
                ),
            })
        chain_results.append({
            "chain_id": cid,
            "length": len(pairs),
            "total_drift_rot_deg": cum_rot,
            "total_drift_trans_m": cum_trans if metric_translation_available else None,
            "steps": steps,
        })

    if not chain_results:
        return {
            "chains": [],
            "mean_drift_rot_deg": 0.0,
            "mean_drift_trans_m": 0.0 if metric_translation_available else None,
        }

    return {
        "chains": chain_results,
        "mean_drift_rot_deg": float(np.mean([c["total_drift_rot_deg"] for c in chain_results])),
        "mean_drift_trans_m": (
            float(np.mean([c["total_drift_trans_m"] for c in chain_results]))
            if metric_translation_available else None
        ),
        "n_chains": len(chain_results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation — the one function users call
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rcpe(
    per_pair: List[Dict[str, Any]],
    rot_auc_thresholds: Sequence[float] = (5.0, 10.0, 20.0),
    metric_rot_thresholds: Sequence[float] = (5.0, 10.0, 20.0),
    metric_trans_thresholds_cm: Sequence[float] = (5.0, 10.0, 20.0),
    metric_translation_available: bool = True,
) -> Dict[str, Any]:
    """Compute the full RPX-RCPE metric basket from per-pair results.

    Parameters
    ----------
    per_pair : list of dict
        Each dict must contain at minimum ``rotation_error_deg`` and
        ``translation_error_m``.  Should also carry ``pair_type``,
        ``rotation_bin``, ``chain_id``, ``chain_position`` from the
        PosePairGenerator metadata for the novel metrics to fire.

    Returns
    -------
    dict with keys:
        ``aggregated`` — overall means.
        ``standard_auc`` — AUC@5°/10°/20° on pose_error_max_deg.
        ``metric_auc`` — null with status ``skipped_by_protocol``.
        ``per_bin`` — breakdown by rotation difficulty.
        ``per_type`` — breakdown by pair type (intra/cross/temporal).
        ``cross_phase_delta`` — perf drop intra→cross (novel).
        ``temporal_drift`` — accumulated error over chains (novel).
    """
    if not per_pair:
        return {}

    # Ensure derived per-pair fields exist.  The runner now emits
    # translation_angular_deg and pose_error_max_deg directly; this
    # fallback handles legacy per-pair dicts that only carry the
    # two original scalar errors.
    for p in per_pair:
        if "translation_angular_deg" not in p:
            p["translation_angular_deg"] = float("nan")
        if "pose_error_max_deg" not in p:
            ta = p.get("translation_angular_deg", 0.0)
            if np.isnan(ta):
                ta = 0.0
            p["pose_error_max_deg"] = max(
                p.get("rotation_error_deg", 0.0), ta,
            )

    # Headline angular AUC values are defined on exact-gap intra-phase pairs.
    # Cross-phase pairs and temporal chains have separate diagnostics.
    headline_rows = [p for p in per_pair if p.get("pair_type") == "intra_phase"]
    if not headline_rows:  # backwards compatibility for legacy manifests
        headline_rows = per_pair
    rot_errs = np.array([p["rotation_error_deg"] for p in headline_rows])
    trans_errs = np.array([p["translation_error_m"] for p in headline_rows])
    pose_max = np.array([p["pose_error_max_deg"] for p in headline_rows])
    canonical = canonical_metric_vector(
        headline_rows,
        metric_translation_available=metric_translation_available,
    )

    per_bin = per_bin_breakdown(per_pair)
    per_type = per_type_breakdown(per_pair)
    if not metric_translation_available:
        for breakdown in (per_bin, per_type):
            for values in breakdown.values():
                values.pop("translation_error_m", None)

    return {
        "n_pairs": len(per_pair),
        "n_headline_intra_pairs": len(headline_rows),
        "metric_translation_available": metric_translation_available,
        "aggregated": {
            "rotation_error_deg": float(np.mean(rot_errs)),
            "rotation_error_deg_median": float(np.median(rot_errs)),
            "translation_error_m": (
                float(np.mean(trans_errs)) if metric_translation_available else None
            ),
            "translation_error_m_median": (
                float(np.median(trans_errs)) if metric_translation_available else None
            ),
            "pose_error_max_deg": float(np.mean(pose_max)),
        },
        "standard_auc": auc_at_thresholds(pose_max, rot_auc_thresholds),
        "canonical": canonical,
        "m_auc": None,
        "metric_auc": None,
        "metric_auc_status": "skipped_by_protocol",
        "per_bin": per_bin,
        "per_type": per_type,
        "cross_phase_delta": cross_phase_auc_delta(per_pair, rot_auc_thresholds),
        "cross_phase_error_delta": cross_phase_delta(per_pair, "rotation_error_deg"),
        "cross_phase_delta_trans": (
            cross_phase_delta(per_pair, "translation_error_m")
            if metric_translation_available else {}
        ),
        "temporal_drift": temporal_drift(
            per_pair, metric_translation_available=metric_translation_available
        ),
    }

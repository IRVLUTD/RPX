"""Comprehensive metrics for two-view relative camera pose.

Sister of ``comprehensive_depth_metrics.py``. Reads a per-pair CSV
log written by ``BatchedRelativePoseBenchmarkModel`` (one row per
prediction, columns ``sample_id, scene_id, phase, phase_b, frame_a, frame_b, R00..R22,
tx, ty, tz``) plus the matching split manifest (carries GT poses
``pose_a`` / ``pose_b`` per pair) and emits the full pose-error
basket with 95% CIs.

Metrics (per pair)
------------------
* **rotation_error_deg** — geodesic angle on SO(3) between predicted
  and GT relative rotations: ``acos((trace(R_pred·R_gt^T) - 1) / 2)``.
* **translation_l2** — Euclidean distance ``||t_pred - t_gt||₂``,
  metres. Note: many pose models predict up-to-scale translation, so
  this is most informative paired with the angular error.
* **translation_angular_deg** — angle between unit-translation
  vectors. Scale-invariant; the standard pose-evaluation literature
  reports this as the "translation error".
* **pose_error_max_deg** — ``max(rotation_error_deg, translation_angular_deg)``,
  used as the threshold variable for AUC.

Aggregates (across the split)
-----------------------------
* Means (with 95% CI bootstrap + t-CI) of every per-pair metric.
* **AUC@5° / @10° / @20°** — area under the curve of fraction-of-pairs
  with ``pose_error_max_deg`` ≤ threshold, integrated up to that
  threshold. The standard pose AUC reported in MapFree, ScanNet, and
  the FAR / Reloc3r / DUSt3R papers.
* Per-pair-stride breakdown (the manifest writer's default stride is
  5 frames, but if it varies across (scene, phase) the breakdown
  shows which strides were harder).
* Phase / difficulty stratification so the per-phase trajectory falls
  out for free.

Usage
-----
    from pose_comprehensive_metrics import compute_run
    extras = compute_run(predictions_csv, manifest_path)
    # extras = {"per_pair": [...], "aggregated": {...},
    #           "aggregated_with_ci": {...}, "auc": {...}}
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# NumPy renamed ``trapz`` → ``trapezoid`` in 2.0 and removed the old name
# entirely. Pick whichever exists so we work across both lines.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


# ─────────────────────────  thresholds  ────────────────────────────────────

#: AUC thresholds in degrees. Standard MapFree / ScanNet pose AUC reports
#: at these three. Configurable via :func:`compute_run`'s ``auc_thresholds``.
DEFAULT_AUC_DEG: Tuple[float, ...] = (5.0, 10.0, 20.0)


# ─────────────────────────  per-pair errors  ───────────────────────────────


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle on SO(3), in degrees, between two rotation matrices."""
    R_rel = R_pred @ R_gt.T
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def translation_l2(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Euclidean distance, in metres (or whatever GT units)."""
    return float(np.linalg.norm(t_pred - t_gt))


def translation_angular_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle between two translation vectors, in degrees.

    Scale-invariant — the standard pose-eval literature reports this as
    "translation error". Returns 0 if either vector has near-zero norm
    (rare for paired-frame pose where the camera always moves a bit).
    """
    eps = 1e-9
    n_pred = np.linalg.norm(t_pred)
    n_gt = np.linalg.norm(t_gt)
    if n_pred < eps or n_gt < eps:
        return 0.0
    cos_theta = float(np.dot(t_pred, t_gt) / (n_pred * n_gt))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


# ─────────────────────────  AUC  ───────────────────────────────────────────


def auc_pose_error(errors_deg: np.ndarray, thresholds: Tuple[float, ...]) -> Dict[str, float]:
    """Standard pose-AUC: area under the empirical CDF of the per-pair
    pose error (max(rot_err, trans_ang_err)), integrated up to each
    threshold and normalised to [0, 1].

    Matches the implementation common across MapFree, ScanNet, FAR,
    Reloc3r, DUSt3R, MASt3R: histogram of errors, cumulative fraction
    sampled on a fine grid, trapezoidal integration to the threshold,
    divided by the threshold.
    """
    out: Dict[str, float] = {}
    if errors_deg.size == 0:
        for thr in thresholds:
            out[f"auc_{thr:g}deg"] = 0.0
        return out

    e = np.asarray(errors_deg, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        for thr in thresholds:
            out[f"auc_{thr:g}deg"] = 0.0
        return out

    for thr in thresholds:
        # 100 bins from 0 to thr; cumulative fraction at each bin edge.
        bins = np.linspace(0.0, float(thr), 101)
        cumulative = (e[None, :] <= bins[:, None]).sum(axis=1).astype(np.float64) / float(e.size)
        # Trapezoidal AUC under the curve, normalised to [0, 1] by dividing by thr.
        auc = float(_trapezoid(cumulative, bins) / float(thr))
        out[f"auc_{thr:g}deg"] = auc
    return out


# ─────────────────────────  per-pair compute  ──────────────────────────────


def _relative_pose_from_world(
    pose_a: np.ndarray, pose_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose camera-to-world poses → relative SE(3) from a→b.

    ``T_rel = T_a^{-1} · T_b``. Returns ``(R_rel, t_rel)``. Same convention
    as :class:`rpx_benchmark.api.RelativePoseGroundTruth`.
    """
    T_rel = np.linalg.inv(pose_a) @ pose_b
    return T_rel[:3, :3], T_rel[:3, 3]


def _pair_metrics(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Dict[str, float]:
    rot_err = rotation_error_deg(R_pred, R_gt)
    t_l2 = translation_l2(t_pred, t_gt)
    t_ang = translation_angular_deg(t_pred, t_gt)
    return {
        "rotation_error_deg": rot_err,
        # Canonical rpx_benchmark pose-metric name. Keep translation_l2 as a
        # compatibility alias in the comprehensive report.
        "translation_error_m": t_l2,
        "translation_l2": t_l2,
        "translation_angular_deg": t_ang,
        "pose_error_max_deg": max(rot_err, t_ang),
    }


# ─────────────────────────  CSV / manifest IO  ─────────────────────────────


def _read_predictions_csv(
    path: Path,
) -> Dict[tuple, Tuple[np.ndarray, np.ndarray]]:
    """Read predictions.csv into canonical and legacy lookup keys.

    Tolerates the CSV's exact column order via DictReader; rows with
    non-finite numeric entries are skipped with a warning. New logs are keyed
    by the manifest sample ID and by both endpoint phases. The four-field key
    remains as a read-only fallback for pre-existing intra-phase logs.
    """
    out: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                phase_b = row.get("phase_b") or row["phase"]
                R = np.array(
                    [
                        float(row["R00"]),
                        float(row["R01"]),
                        float(row["R02"]),
                        float(row["R10"]),
                        float(row["R11"]),
                        float(row["R12"]),
                        float(row["R20"]),
                        float(row["R21"]),
                        float(row["R22"]),
                    ],
                    dtype=np.float64,
                ).reshape(3, 3)
                t = np.array(
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64
                )
            except (KeyError, ValueError) as e:
                log.warning("skipping malformed row %s: %s", row, e)
                continue
            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                log.warning("skipping non-finite row %s", row)
                continue
            value = (R, t)
            sample_id = row.get("sample_id")
            if sample_id:
                out[("id", sample_id)] = value
            out[(
                "pair",
                row["scene_id"], row["phase"], phase_b,
                row["frame_a"], row["frame_b"],
            )] = value
            # Legacy logs did not distinguish phase_b. Only expose their old
            # key when the row truly lacks the new phase_b column.
            if "phase_b" not in row:
                out[(
                    "legacy", row["scene_id"], row["phase"],
                    row["frame_a"], row["frame_b"],
                )] = value
    return out


def _load_pose_npz(path: Path) -> np.ndarray:
    """Load a 4×4 SE(3) from a published NPY or legacy NPZ pose.

    Published RPX snapshots use a ``(7,)`` NPY vector ordered as
    ``[x, y, z, qx, qy, qz, qw]``. Legacy trees store the same values in
    named NPZ arrays. Keep this wrapper name for compatibility with callers.
    """
    from rpx_benchmark.pose_pairs import _load_pose_file

    R, pos = _load_pose_file(path)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


# ─────────────────────────  public entry  ──────────────────────────────────


def compute_run(
    predictions_csv: Path,
    manifest_path: Path,
    *,
    auc_thresholds: Tuple[float, ...] = DEFAULT_AUC_DEG,
    snapshot_root: Path | None = None,
) -> Dict:
    """Walk the prediction CSV against the manifest, compute the full
    pose-error basket with 95% CIs.

    Returns ``{"per_pair", "aggregated", "aggregated_with_ci", "auc",
    "by_phase", "by_stride"}``. Empty if the CSV is empty or the
    manifest doesn't carry pose entries.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    extracted_root = Path(manifest.get("root") or (snapshot_root or "."))
    preds = _read_predictions_csv(Path(predictions_csv))

    per_pair: List[Dict] = []
    for s in manifest["samples"]:
        # Manifest entries for relative_pose carry pose_a + pose_b paths
        # alongside the rgb / rgb_b. Skip rows the writer doesn't carry.
        if "pose_a" not in s or "pose_b" not in s:
            continue
        scene = str(s.get("scene_id") or "")
        phase = str(s.get("phase") if s.get("phase") is not None else "")
        # frame_a / frame_b are stored in the `metadata` block by
        # split_manifests' _RelativePoseSpec; fall back to id parsing.
        meta = s.get("metadata") or {}
        frame_a = str(meta.get("frame") or "")
        frame_b = str(meta.get("frame_b") or "")
        phase_b = str(meta.get("phase_idx_b") if meta.get("phase_idx_b") is not None else phase)
        lookup_keys = [
            ("id", str(s.get("id") or "")),
            ("pair", scene, phase, phase_b, frame_a, frame_b),
            ("legacy", scene, phase, frame_a, frame_b),
        ]
        prediction = next((preds[k] for k in lookup_keys if k in preds), None)
        if prediction is None:
            continue
        key = lookup_keys[0]

        try:
            T_a = _load_pose_npz(extracted_root / s["pose_a"])
            T_b = _load_pose_npz(extracted_root / s["pose_b"])
        except (FileNotFoundError, KeyError, ValueError) as e:
            log.warning("skipping pair %s — could not load GT pose: %s", key, e)
            continue
        R_gt, t_gt = _relative_pose_from_world(T_a, T_b)
        R_pred, t_pred = prediction
        m = _pair_metrics(R_pred, t_pred, R_gt, t_gt)
        m.update(
            {
                "id": s.get("id"),
                "scene_id": scene,
                "phase": int(phase) if phase.isdigit() else phase,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "stride": int(meta.get("pair_stride") or 0) or _infer_stride(frame_a, frame_b),
                "pair_type": meta.get("pair_type", s.get("pair_type", "unknown")),
                "rotation_bin": meta.get("rotation_bin", s.get("rotation_bin")),
                "chain_id": meta.get("chain_id"),
                "chain_position": meta.get("chain_position"),
                "metadata": meta,
                "pred_rotation": R_pred.tolist(),
                "pred_translation": t_pred.tolist(),
                "gt_rotation": R_gt.tolist(),
                "gt_translation": t_gt.tolist(),
            }
        )
        per_pair.append(m)

    if not per_pair:
        return {
            "per_pair": [],
            "aggregated": {},
            "aggregated_with_ci": {},
            "auc": {},
            "by_phase": {},
            "by_stride": {},
        }

    # Plain means (backwards-compat with the depth-side comprehensive shape).
    metric_keys = (
        "rotation_error_deg",
        "translation_error_m",
        "translation_l2",
        "translation_angular_deg",
        "pose_error_max_deg",
    )
    aggregated = {k: float(np.mean([r[k] for r in per_pair])) for k in metric_keys}

    # AUC over the *whole-split* pose_error_max_deg distribution.
    auc = auc_pose_error(
        np.asarray([r["pose_error_max_deg"] for r in per_pair], dtype=np.float64),
        thresholds=auc_thresholds,
    )
    aggregated.update(auc)

    # 95% CIs via the shared stat_utils helper. We import lazily so this
    # module is importable in environments without the scripts/ entry
    # on PYTHONPATH.
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from stat_utils import aggregate_per_sample_with_ci  # noqa: PLC0415

        aggregated_with_ci = aggregate_per_sample_with_ci(
            per_pair,
            drop_keys=("id", "scene_id", "phase", "frame_a", "frame_b", "stride"),
        )
    except ImportError:
        aggregated_with_ci = {}

    # Phase + stride stratification.
    by_phase: Dict[str, Dict[str, float]] = {}
    for ph in {r["phase"] for r in per_pair}:
        rows = [r for r in per_pair if r["phase"] == ph]
        by_phase[str(ph)] = {
            **{k: float(np.mean([r[k] for r in rows])) for k in metric_keys},
            **auc_pose_error(
                np.asarray([r["pose_error_max_deg"] for r in rows], dtype=np.float64),
                thresholds=auc_thresholds,
            ),
            "n_pairs": len(rows),
        }

    by_stride: Dict[str, Dict[str, float]] = {}
    for st in {r["stride"] for r in per_pair if r["stride"]}:
        rows = [r for r in per_pair if r["stride"] == st]
        by_stride[str(st)] = {
            **{k: float(np.mean([r[k] for r in rows])) for k in metric_keys},
            **auc_pose_error(
                np.asarray([r["pose_error_max_deg"] for r in rows], dtype=np.float64),
                thresholds=auc_thresholds,
            ),
            "n_pairs": len(rows),
        }

    return {
        "per_pair": per_pair,
        "aggregated": aggregated,
        "aggregated_with_ci": aggregated_with_ci,
        "auc": auc,
        "by_phase": by_phase,
        "by_stride": by_stride,
    }


def _infer_stride(frame_a: str, frame_b: str) -> int:
    try:
        return abs(int(frame_b) - int(frame_a))
    except ValueError:
        return 0


# ─────────────────────────  CLI  ───────────────────────────────────────────


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--predictions-csv", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--auc-thresholds", type=float, nargs="+", default=list(DEFAULT_AUC_DEG))
    args = ap.parse_args()

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "pose_comprehensive_metrics — full RCPE metric basket",
        "rotation + translation L2/angular + AUC@5°/10°/20° (with 95% CIs, by-phase, by-stride)",
    )
    cli_ux.config(vars(args))

    extras = compute_run(
        args.predictions_csv,
        args.manifest,
        auc_thresholds=tuple(args.auc_thresholds),
    )
    out = args.out or args.predictions_csv.parent / "pose_comprehensive_metrics.json"
    out.write_text(json.dumps(extras, indent=2))
    print(f"wrote {out}  ({len(extras['per_pair'])} pairs)")
    if extras["aggregated"]:
        print("--- aggregated ---")
        for k in sorted(extras["aggregated"]):
            print(f"  {k:>32}: {extras['aggregated'][k]:.4f}")


if __name__ == "__main__":
    _cli()

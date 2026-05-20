"""Comprehensive monocular-depth metrics per
``docs/methods/comprehensive_metrics.md`` spec.

Runs **after** BenchmarkRunner finishes, reads the per-frame ``.npz`` predictions
(saved by ``run_depth.py --save-predictions``) and the GT depth + (optional)
mask paths from the local manifest. Computes:

- Error metrics (9): AbsRel, SqRel, RMSE, RMSElog, SIlog, log10, MAE, iRMSE, iMAE
- Accuracy thresholds (3): δ1, δ2, δ3
- Alignment variants: none | median | ls_affine | ls_disparity
- Stratification:
  - depth bands: near (0.3–1 m), mid (1–3 m), far (3 m+)
  - mask region: in-mask (object) vs out-of-mask (background) — when masks present
- Boundary metrics: edge F-score, edge accuracy, edge completeness — when masks present
- Object-relative depth ordering (ORD): % object-pair depth-rankings preserved

Skipped in v1 (will land later):
- Planarity error on table surface (needs explicit plane-prior selection)
- Depth-discontinuity recall (needs GT depth-edge labelling)

Usage
-----
    from comprehensive_depth_metrics import compute_run
    extras = compute_run(predictions_dir, manifest_path, alignment="none")
    # extras = {"per_sample": [...], "aggregated": {...}}

Or via run_depth.py: ``--comprehensive-metrics`` (auto-enables --save-predictions).
"""

from __future__ import annotations

import json
import logging
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# Valid-pixel range for evaluation (D435 working range).
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 5.0

# Below this many valid pixels, a (region | object | band) is reported as
# "no data" instead of producing a tiny-sample point estimate. 100 is loose
# enough to keep small objects, tight enough to avoid 1-pixel artefacts.
MIN_VALID_PIXELS = 100

# Per-object: drop instances smaller than this many pixels — they are
# almost always over-segmentation noise and dominate the per-object mean
# with high-variance estimates.
MIN_OBJECT_PIXELS = 200


def _valid(pred: np.ndarray, gt_m: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels usable for metric computation.

    A pixel is *valid* iff:
    - GT is finite (not NaN / not inf)  — the D435 emits NaN/0 in holes;
    - GT is in the sensor's working range (``DEPTH_MIN_M``…``DEPTH_MAX_M`` m);
    - prediction is finite (not NaN / not inf);
    - prediction is positive (depth must be > 0).

    Holes (GT NaN or zero) are *excluded*; they are reported separately by
    :func:`_hole_stats` so a model that hallucinates depth in a sensor
    hole isn't penalised, and a per-frame coverage number is available.
    """
    finite_gt = np.isfinite(gt_m)
    in_range = (gt_m > DEPTH_MIN_M) & (gt_m < DEPTH_MAX_M)
    finite_pr = np.isfinite(pred)
    pos_pr = pred > 0
    return finite_gt & in_range & finite_pr & pos_pr


# ───────────────────────────  Alignment  ────────────────────────────────────

# Single source of truth lives in rpx_benchmark.metrics.depth_alignment so
# the runner-side alignment (BatchedDepthBenchmarkModel.predict) and the
# post-processor here stay bit-identical.
from rpx_benchmark.metrics.depth_alignment import align_pred_to_gt as _align_canonical


def _align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    """Thin shim with the historical (pred, gt, valid, mode) signature.

    Delegates to the canonical aligner; the precomputed valid mask is
    threaded through so we don't recompute it.
    """
    return _align_canonical(pred, gt, mode, valid=valid)


# ───────────────────────────  Error / accuracy  ─────────────────────────────


def _errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p, g = pred[valid], gt[valid]
    eps = 1e-6
    log_p, log_g = np.log(np.maximum(p, eps)), np.log(np.maximum(g, eps))
    e = log_p - log_g
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean(e**2))),
        "silog": float(100.0 * np.sqrt(max(np.mean(e**2) - np.mean(e) ** 2, 0.0))),
        "log10": float(
            np.mean(np.abs(np.log10(np.maximum(p, eps)) - np.log10(np.maximum(g, eps))))
        ),
        "mae": float(np.mean(np.abs(p - g))),
        "irmse": float(np.sqrt(np.mean((1.0 / p - 1.0 / g) ** 2))),
        "imae": float(np.mean(np.abs(1.0 / p - 1.0 / g))),
    }


def _accuracy(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p, g = pred[valid], gt[valid]
    ratio = np.maximum(p / g, g / p)
    return {
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def _basket(pred, gt, valid) -> dict:
    if valid.sum() < MIN_VALID_PIXELS:
        return {}
    out = _errors(pred, gt, valid)
    out.update(_accuracy(pred, gt, valid))
    return out


# ───────────────────────────  Per-object metrics  ──────────────────────────
#
# Robotics-relevant: how well does the model predict depth on each
# manipulable object? An object is one connected instance in the SAM2 mask
# (mask values > 0 are instance ids; 0 is background). For each instance
# we also report `coverage` — the fraction of object pixels where GT is
# valid (i.e. not in a sensor hole) — so a low number flags "the model's
# prediction here is nominally OK but the sensor only saw 30% of the
# object".


def _per_object_rows(pred, gt, valid, mask) -> list[dict]:
    """Return a list of per-instance metric rows.

    Each row carries:
        ``instance_id`` (int), ``n_pixels`` (object pixels in the frame),
        ``n_valid`` (object pixels where GT is valid), ``coverage``
        (n_valid / n_pixels), plus the full error+accuracy basket
        computed on the object's valid-pixel subset.

    Instances with fewer than ``MIN_OBJECT_PIXELS`` pixels are dropped to
    keep the per-object mean from being dominated by 5-pixel splinters.
    """
    if mask is None or mask.size == 0:
        return []
    ids = np.unique(mask)
    ids = ids[ids > 0]
    rows: list[dict] = []
    for inst in ids:
        obj = mask == inst
        n_pixels = int(obj.sum())
        if n_pixels < MIN_OBJECT_PIXELS:
            continue
        obj_valid = obj & valid
        n_valid = int(obj_valid.sum())
        coverage = float(n_valid / n_pixels) if n_pixels else 0.0
        row: dict = {
            "instance_id": int(inst),
            "n_pixels": n_pixels,
            "n_valid": n_valid,
            "coverage": coverage,
        }
        if n_valid >= MIN_VALID_PIXELS:
            row.update(_errors(pred, gt, obj_valid))
            row.update(_accuracy(pred, gt, obj_valid))
        rows.append(row)
    return rows


def _per_object_aggregate(rows: list[dict]) -> dict:
    """Mean across object instances (instance-weighted, *not* pixel-weighted).

    Robotics framing: each object counts the same regardless of how many
    pixels it occupies, because a small graspable object is just as
    important to perceive as a large one.
    """
    if not rows:
        return {}
    metric_keys = sorted(
        {
            k
            for r in rows
            for k in r
            if k not in {"instance_id", "n_pixels", "n_valid", "coverage"}
            and isinstance(r[k], (int, float))
        }
    )
    out: dict = {f"per_object/n_objects": len(rows)}
    if rows:
        out["per_object/mean_coverage"] = float(np.mean([r["coverage"] for r in rows]))
        out["per_object/mean_n_pixels"] = float(np.mean([r["n_pixels"] for r in rows]))
    for k in metric_keys:
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        if vals:
            out[f"per_object/{k}"] = float(np.mean(vals))
    return out


# ───────────────────────────  Hole / coverage stats  ───────────────────────


def _hole_stats(gt: np.ndarray, mask: np.ndarray | None) -> dict:
    """Per-frame hole statistics — what fraction of the GT is missing.

    Reports overall hole fraction, plus in-mask and out-of-mask hole
    fractions when a SAM2 mask is provided. NaN, ±inf, and pixels at or
    below ``DEPTH_MIN_M`` are all counted as holes (the D435 reports
    holes as 0 mm; some downstream code converts to NaN — both are
    captured by ``~np.isfinite | <= DEPTH_MIN_M``).
    """
    holes = ~np.isfinite(gt) | (gt <= DEPTH_MIN_M)
    n = gt.size
    out: dict = {"holes/overall_fraction": float(holes.sum() / n) if n else 0.0}
    if mask is not None and mask.size == gt.size:
        in_obj = mask > 0
        if in_obj.any():
            out["holes/in_mask_fraction"] = float((holes & in_obj).sum() / max(in_obj.sum(), 1))
        out_obj = mask == 0
        if out_obj.any():
            out["holes/out_mask_fraction"] = float((holes & out_obj).sum() / max(out_obj.sum(), 1))
    return out


# ───────────────────────────  Stratification  ───────────────────────────────


def _by_depth_band(pred, gt, valid) -> dict:
    bands = {"near": (0.3, 1.0), "mid": (1.0, 3.0), "far": (3.0, DEPTH_MAX_M)}
    out = {}
    for name, (lo, hi) in bands.items():
        m = valid & (gt >= lo) & (gt < hi)
        b = _basket(pred, gt, m)
        for k, v in b.items():
            out[f"{name}/{k}"] = v
    return out


def _by_mask(pred, gt, valid, mask) -> dict:
    out = {}
    in_m = valid & (mask > 0)
    out_m = valid & (mask == 0)
    for k, v in _basket(pred, gt, in_m).items():
        out[f"in_mask/{k}"] = v
    for k, v in _basket(pred, gt, out_m).items():
        out[f"out_mask/{k}"] = v
    return out


# ───────────────────────────  Boundary / ORD  ───────────────────────────────


def _edge_pixels(mask: np.ndarray) -> np.ndarray:
    """3-pixel-thick boundary set of a labelled instance mask."""
    if mask.size == 0:
        return np.zeros_like(mask, dtype=bool)
    # Differences with neighbours; any neighbour with a different label = edge.
    e = np.zeros_like(mask, dtype=bool)
    e[:-1] |= mask[:-1] != mask[1:]
    e[1:] |= mask[1:] != mask[:-1]
    e[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    e[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    # Slight dilation (3x3 OR) so the band is not a single pixel wide.
    out = e.copy()
    out[1:] |= e[:-1]
    out[:-1] |= e[1:]
    out[:, 1:] |= e[:, :-1]
    out[:, :-1] |= e[:, 1:]
    return out


def _boundary_metrics(pred, gt, valid, mask, depth_jump_m: float = 0.05) -> dict:
    """Edge F-score / accuracy / completeness on instance-mask boundaries."""
    edges = _edge_pixels(mask) & valid
    if edges.sum() < 50:
        return {}
    # GT depth jump along edges (max neighbour difference).
    gt_jump = np.zeros_like(gt)
    gt_jump[:-1] = np.maximum(gt_jump[:-1], np.abs(gt[:-1] - gt[1:]))
    gt_jump[1:] = np.maximum(gt_jump[1:], np.abs(gt[1:] - gt[:-1]))
    gt_jump[:, :-1] = np.maximum(gt_jump[:, :-1], np.abs(gt[:, :-1] - gt[:, 1:]))
    gt_jump[:, 1:] = np.maximum(gt_jump[:, 1:], np.abs(gt[:, 1:] - gt[:, :-1]))
    pred_jump = np.zeros_like(pred)
    pred_jump[:-1] = np.maximum(pred_jump[:-1], np.abs(pred[:-1] - pred[1:]))
    pred_jump[1:] = np.maximum(pred_jump[1:], np.abs(pred[1:] - pred[:-1]))
    pred_jump[:, :-1] = np.maximum(pred_jump[:, :-1], np.abs(pred[:, :-1] - pred[:, 1:]))
    pred_jump[:, 1:] = np.maximum(pred_jump[:, 1:], np.abs(pred[:, 1:] - pred[:, :-1]))

    is_gt_edge = edges & (gt_jump > depth_jump_m)
    is_pred_edge_at_gt = edges & (pred_jump > depth_jump_m)

    if is_gt_edge.sum() < 50:
        return {}
    accuracy = float(np.mean(is_pred_edge_at_gt[edges]))  # pred edges that are real
    completeness = float(np.mean(is_pred_edge_at_gt[is_gt_edge]))  # GT edges recovered
    f_score = 2 * accuracy * completeness / max(accuracy + completeness, 1e-9)
    return {
        "boundary/accuracy": accuracy,
        "boundary/completeness": completeness,
        "boundary/f_score": float(f_score),
    }


def _ord(pred, gt, valid, mask, max_pairs: int = 4000) -> dict:
    """Object-relative depth-ordering accuracy: sample pairs of pixels from
    different instances; check if pred preserves their gt depth ordering."""
    ids = np.unique(mask[valid])
    ids = ids[ids > 0]
    if len(ids) < 2:
        return {}
    # Project-wide canonical seed: MMDDYYYY 05/06/2026 → 5_062_026.
    from rpx_benchmark.determinism import RPX_SEED  # noqa: PLC0415

    rng = np.random.default_rng(RPX_SEED)
    correct = 0
    total = 0
    for _ in range(max_pairs):
        a, b = rng.choice(ids, size=2, replace=False)
        ya, xa = np.where((mask == a) & valid)
        yb, xb = np.where((mask == b) & valid)
        if len(ya) == 0 or len(yb) == 0:
            continue
        i = rng.integers(len(ya))
        j = rng.integers(len(yb))
        ga, gb = float(gt[ya[i], xa[i]]), float(gt[yb[j], xb[j]])
        pa, pb = float(pred[ya[i], xa[i]]), float(pred[yb[j], xb[j]])
        if abs(ga - gb) < 0.05:  # ambiguous gt ordering
            continue
        if (ga < gb) == (pa < pb):
            correct += 1
        total += 1
    if total < 50:
        return {}
    return {"ord/pair_accuracy": correct / total, "ord/n_pairs_used": total}


# ───────────────────────────  Mask loading from tars  ──────────────────────


def _load_mask_from_cache(
    snapshot_root: Path, scene: str, phase: int, frame_filename: str
) -> Optional[np.ndarray]:
    """Read a single instance mask from the cached ``masks/v1.tar``.

    Tar layout: ``sam2/masks/<frame>.png`` (the SAM2 outputs sit under a
    ``sam2/`` prefix inside the tar — earlier ``masks/<frame>`` was the
    bug). Falls back to a few legacy paths so older shards still resolve.
    """
    tar_path = snapshot_root / f"scenes/{scene}/{phase}/labels/masks/v1.tar"
    if not tar_path.exists():
        return None
    candidates = (
        f"sam2/masks/{frame_filename}",
        f"masks/{frame_filename}",
        f"sam2/{frame_filename}",
    )
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in candidates:
                f = tf.extractfile(member)
                if f is not None:
                    return np.array(Image.open(BytesIO(f.read())))
    except Exception:
        return None
    return None


def _hf_snapshot_root(repo_id: str = "itaykadosh/RPX") -> Path:
    import os as _os

    cache = Path(_os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = sorted(
        (cache / f"datasets--{repo_id.replace('/', '--')}" / "snapshots").iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snaps[0]


# ───────────────────────────  Per-run aggregation  ─────────────────────────


def compute_run(
    predictions_dir: Path,
    manifest_path: Path,
    *,
    alignment: str = "none",
    snapshot_root: Optional[Path] = None,
) -> dict:
    """Walk per-frame predictions and aggregate the comprehensive basket.

    Returns ``{"alignment": ..., "per_sample": [...], "aggregated": {...}}``.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    extracted_root = Path(manifest["root"])
    snapshot = snapshot_root or _hf_snapshot_root()

    per_sample = []
    per_object_rows_all: list[dict] = []
    for s in manifest["samples"]:
        sid = s["id"]
        # Predictions can sit in either the legacy flat layout
        # (``<dir>/<id>.npz``) or the scene/phase/frame layout
        # (``<dir>/<scene>/<phase>/<frame>.npz``). Try the scene/phase
        # path first since it's the new default; fall back to flat.
        scene = str(s.get("scene_id") or "")
        phase = str(s.get("phase") if s.get("phase") is not None else "")
        frame = Path(s.get("rgb", f"{sid}.png")).stem  # frame number, no ext
        pred_path = Path(predictions_dir) / scene / phase / f"{frame}.npz"
        if not pred_path.exists():
            pred_path = Path(predictions_dir) / f"{sid}.npz"
        if not pred_path.exists():
            continue
        pred = np.load(pred_path)["depth"].astype(np.float32)
        gt_path = extracted_root / s["depth"]
        gt_mm = np.array(Image.open(gt_path))
        gt_m = gt_mm.astype(np.float32) / 1000.0
        valid = _valid(pred, gt_m)

        pred_aligned = _align(pred, gt_m, valid, alignment) if valid.any() else pred

        row: dict = {
            "id": sid,
            "scene_id": s.get("scene_id"),
            "phase": s.get("phase"),
            "alignment": alignment,
        }
        row.update(_basket(pred_aligned, gt_m, valid))
        row.update(_by_depth_band(pred_aligned, gt_m, valid))

        # Mask-dependent metrics, if a mask is reachable from the cache.
        mask = (
            _load_mask_from_cache(
                snapshot, str(s.get("scene_id", "")), int(s.get("phase", 0)), Path(s["rgb"]).name
            )
            if s.get("scene_id") is not None
            else None
        )
        # Hole statistics — independent of model quality. Reported even
        # when there's no mask (just the overall hole fraction then).
        row.update(_hole_stats(gt_m, mask))
        if mask is not None:
            row.update(_by_mask(pred_aligned, gt_m, valid, mask))
            row.update(_boundary_metrics(pred_aligned, gt_m, valid, mask))
            row.update(_ord(pred_aligned, gt_m, valid, mask))

            # Per-object detail and per-frame aggregate.
            obj_rows = _per_object_rows(pred_aligned, gt_m, valid, mask)
            for obj in obj_rows:
                per_object_rows_all.append(
                    {
                        "id": sid,
                        "scene_id": s.get("scene_id"),
                        "phase": s.get("phase"),
                        **obj,
                    }
                )
            row.update(_per_object_aggregate(obj_rows))
        per_sample.append(row)

    if not per_sample:
        return {
            "alignment": alignment,
            "per_sample": [],
            "per_object": [],
            "aggregated": {},
            "aggregated_with_ci": {},
            "per_object_aggregated": {},
            "per_object_aggregated_with_ci": {},
        }

    # Plain mean (preserved for backward compat)…
    keys = sorted(
        {k for r in per_sample for k in r if k not in {"id", "scene_id", "phase", "alignment"}}
    )
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in per_sample if k in r and isinstance(r[k], (int, float))]
        if vals:
            aggregated[k] = float(np.mean(vals))

    # …and the same basket with 95% CIs (t-based + bootstrap), std, p5/p95.
    # Used by the paper tables and any downstream significance test.
    from stat_utils import aggregate_per_sample_with_ci  # noqa: PLC0415

    aggregated_with_ci = aggregate_per_sample_with_ci(per_sample)

    # Per-object aggregation: every object instance is one observation
    # (instance-weighted mean), so a small graspable bottle counts the
    # same as a large monitor. Reported with the same CI shape so paper
    # tables can show "AbsRel = 0.085 [0.082, 0.088] across 142 objects".
    per_object_aggregated = {}
    per_object_aggregated_with_ci = {}
    if per_object_rows_all:
        obj_keys = sorted(
            {
                k
                for r in per_object_rows_all
                for k in r
                if k not in {"id", "scene_id", "phase", "instance_id"}
                and isinstance(r[k], (int, float))
            }
        )
        for k in obj_keys:
            vals = [r[k] for r in per_object_rows_all if k in r and isinstance(r[k], (int, float))]
            if vals:
                per_object_aggregated[k] = float(np.mean(vals))
        per_object_aggregated_with_ci = aggregate_per_sample_with_ci(
            per_object_rows_all,
            drop_keys=("id", "scene_id", "phase", "instance_id"),
        )

    return {
        "alignment": alignment,
        "per_sample": per_sample,
        "aggregated": aggregated,
        "aggregated_with_ci": aggregated_with_ci,
        "per_object": per_object_rows_all,
        "per_object_aggregated": per_object_aggregated,
        "per_object_aggregated_with_ci": per_object_aggregated_with_ci,
    }


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--alignment", default="none", choices=["none", "median", "ls_affine", "ls_disparity"]
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from rpx_benchmark import cli_ux

    cli_ux.banner(
        "comprehensive_depth_metrics — full metric basket",
        "9 errors × 4 alignment modes × 3 depth bands × in/out-of-mask + per-object",
    )
    cli_ux.config(vars(args))

    extras = compute_run(args.predictions_dir, args.manifest, alignment=args.alignment)
    out = args.out or args.predictions_dir.parent / "comprehensive_metrics.json"
    out.write_text(json.dumps(extras, indent=2))
    print(f"wrote {out}  ({len(extras['per_sample'])} samples)")
    if extras["aggregated"]:
        print("--- aggregated ---")
        for k in sorted(extras["aggregated"]):
            print(f"  {k:>34}: {extras['aggregated'][k]:.4f}")


if __name__ == "__main__":
    _cli()

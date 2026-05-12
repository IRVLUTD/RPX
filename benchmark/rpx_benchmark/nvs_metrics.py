"""RPX-NVS evaluation metrics — standard + Perceptual Utility Score.

Standard
--------
- PSNR, SSIM, LPIPS — rendering quality of synthesized RGB vs GT.
- Depth AbsRel / RMSE / δ<1.25 — geometric accuracy of rendered depth
  vs D435 sensor GT.  Novel for NVS benchmarks (no existing benchmark
  evaluates rendered depth against real sensor measurements).

Novel: Perceptual Utility Score (PUS)
-------------------------------------
Measures whether a rendered view is *useful* for downstream robot
perception — not just whether it *looks good*.

    PUS(task) = metric(rendered_view) / metric(real_view)

PUS = 1.0 means the NVS output is perceptually indistinguishable from
real for the downstream task.  PUS = 0.5 means the downstream model
loses half its performance on rendered views.

Motivation: GNFactor (CoRL 2023) showed PSNR drops when action loss is
active but task success improves — PSNR is a poor proxy for robotic
utility.  PUS directly measures what robots need.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Standard rendering quality
# ─────────────────────────────────────────────────────────────────────────────

def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB.  uint8 range [0, 255]."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(
    pred: np.ndarray, gt: np.ndarray,
    k1: float = 0.01, k2: float = 0.03, L: float = 255.0,
) -> float:
    """Simplified global SSIM (no sliding window).

    For publication, recompute with a full sliding-window implementation
    (e.g., skimage.metrics.structural_similarity).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    mu_p, mu_g = pred.mean(), gt.mean()
    sig_p, sig_g = pred.std(), gt.std()
    sig_pg = float(np.mean((pred - mu_p) * (gt - mu_g)))
    num = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)
    den = (mu_p ** 2 + mu_g ** 2 + c1) * (sig_p ** 2 + sig_g ** 2 + c2)
    return float(num / den) if den > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Depth rendering quality (novel for NVS benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

def depth_metrics(
    pred_depth: np.ndarray, gt_depth: np.ndarray,
) -> Dict[str, float]:
    """AbsRel / RMSE / δ<1.25 of rendered depth vs sensor GT.

    Evaluated only at pixels where GT depth > 0 (valid D435 measurement).
    """
    pred = np.asarray(pred_depth, dtype=np.float32)
    gt = np.asarray(gt_depth, dtype=np.float32)

    valid = gt > 0
    if not valid.any():
        return {"depth_absrel": 0.0, "depth_rmse": 0.0, "depth_delta1": 1.0}

    p, g = np.clip(pred[valid], 1e-6, None), gt[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    thresh = np.maximum(p / g, g / p)
    delta1 = float(np.mean(thresh < 1.25))

    return {"depth_absrel": absrel, "depth_rmse": rmse, "depth_delta1": delta1}


# ─────────────────────────────────────────────────────────────────────────────
# Per-object rendering quality (novel — uses instance masks)
# ─────────────────────────────────────────────────────────────────────────────

def per_object_psnr(
    pred_rgb: np.ndarray, gt_rgb: np.ndarray, mask: np.ndarray,
) -> Dict[int, float]:
    """PSNR per instance ID in the mask.

    Returns ``{instance_id: psnr_value}``.  Background (ID ≤ 0) excluded.
    """
    mask = np.asarray(mask, dtype=np.int32)
    ids = np.unique(mask)
    ids = ids[ids > 0]  # skip background

    out: Dict[int, float] = {}
    for obj_id in ids:
        obj_mask = mask == obj_id
        if obj_mask.sum() < 10:  # skip tiny regions
            continue
        pred_roi = np.asarray(pred_rgb, dtype=np.float64)[obj_mask]
        gt_roi = np.asarray(gt_rgb, dtype=np.float64)[obj_mask]
        mse = np.mean((pred_roi - gt_roi) ** 2)
        out[int(obj_id)] = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 1e-10 else 100.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Utility Score (PUS) — the novel contribution
# ─────────────────────────────────────────────────────────────────────────────

def perceptual_utility_score(
    metric_on_rendered: float,
    metric_on_real: float,
    higher_is_better: bool = True,
) -> float:
    """Compute PUS for a single (rendered, real) metric pair.

    PUS = metric(rendered) / metric(real)  if higher_is_better
    PUS = metric(real) / metric(rendered)  if lower_is_better

    Clamped to [0, 2] to avoid degenerate ratios.
    Returns 1.0 when rendered matches real.
    """
    if higher_is_better:
        if metric_on_real < 1e-10:
            return 1.0 if metric_on_rendered < 1e-10 else 2.0
        return float(np.clip(metric_on_rendered / metric_on_real, 0.0, 2.0))
    else:
        if metric_on_rendered < 1e-10:
            return 2.0 if metric_on_real > 1e-10 else 1.0
        return float(np.clip(metric_on_real / metric_on_rendered, 0.0, 2.0))


def compute_pus_batch(
    rendered_results: List[Dict[str, float]],
    real_results: List[Dict[str, float]],
    metric_directions: Dict[str, bool],
) -> Dict[str, float]:
    """Compute PUS across a batch of (rendered, real) evaluation pairs.

    Parameters
    ----------
    rendered_results : list of dict
        Per-sample metrics computed on rendered views.
    real_results : list of dict
        Per-sample metrics computed on real views (same viewpoints).
    metric_directions : dict
        Maps metric name → True if higher is better.
        E.g. ``{"ap50": True, "absrel": False, "miou": True}``.

    Returns
    -------
    dict mapping ``"pus_{metric_name}"`` → mean PUS across all samples.
    """
    if len(rendered_results) != len(real_results):
        from .exceptions import MetricError
        raise MetricError(
            f"Mismatched batch sizes: {len(rendered_results)} rendered "
            f"vs {len(real_results)} real",
            hint="Ensure rendered_results and real_results have the same length.",
        )

    pus_accum: Dict[str, List[float]] = {}
    for rend, real in zip(rendered_results, real_results, strict=False):
        for metric_name, higher in metric_directions.items():
            if metric_name in rend and metric_name in real:
                score = perceptual_utility_score(
                    rend[metric_name], real[metric_name], higher,
                )
                pus_accum.setdefault(metric_name, []).append(score)

    out: Dict[str, float] = {}
    for name, scores in pus_accum.items():
        out[f"pus_{name}"] = float(np.mean(scores))
    if pus_accum:
        all_scores = [s for scores in pus_accum.values() for s in scores]
        out["pus_mean"] = float(np.mean(all_scores))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full NVS evaluation — one function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_nvs(
    per_sample: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the full RPX-NVS metric basket from per-sample results.

    Each sample dict should contain at minimum:
        ``psnr``, ``ssim`` (standard rendering quality).
    Optional:
        ``depth_absrel``, ``depth_rmse``, ``depth_delta1`` (depth rendering).
        ``sample_type`` (interpolation / extrapolation / cross_phase).
        ``n_context`` (number of context views).
        ``pus_*`` fields (perceptual utility scores).
    """
    if not per_sample:
        return {}

    # Aggregate standard metrics
    metric_keys = [
        "psnr", "ssim", "lpips",
        "depth_absrel", "depth_rmse", "depth_delta1",
    ]
    aggregated: Dict[str, float] = {}
    for k in metric_keys:
        vals = [s[k] for s in per_sample if k in s and s[k] is not None]
        if vals:
            aggregated[k] = float(np.mean(vals))

    # Aggregate PUS metrics
    pus_keys = [k for k in per_sample[0] if k.startswith("pus_")]
    for k in pus_keys:
        vals = [s[k] for s in per_sample if k in s and s[k] is not None]
        if vals:
            aggregated[k] = float(np.mean(vals))

    # Per sample_type breakdown
    by_type: Dict[str, Dict[str, float]] = {}
    types = set(s.get("sample_type", "unknown") for s in per_sample)
    for stype in sorted(types):
        subset = [s for s in per_sample if s.get("sample_type") == stype]
        type_agg: Dict[str, float] = {"n_samples": float(len(subset))}
        for k in metric_keys + pus_keys:
            vals = [s[k] for s in subset if k in s and s[k] is not None]
            if vals:
                type_agg[k] = float(np.mean(vals))
        by_type[stype] = type_agg

    # Per n_context breakdown
    by_context: Dict[str, Dict[str, float]] = {}
    ctx_counts = set(s.get("n_context", 0) for s in per_sample)
    for n_ctx in sorted(ctx_counts):
        subset = [s for s in per_sample if s.get("n_context") == n_ctx]
        ctx_agg: Dict[str, float] = {"n_samples": float(len(subset))}
        for k in metric_keys + pus_keys:
            vals = [s[k] for s in subset if k in s and s[k] is not None]
            if vals:
                ctx_agg[k] = float(np.mean(vals))
        by_context[f"ctx_{n_ctx}"] = ctx_agg

    # Cross-phase degradation
    intra = [s for s in per_sample if s.get("sample_type") != "cross_phase"]
    cross = [s for s in per_sample if s.get("sample_type") == "cross_phase"]
    cross_delta: Dict[str, float] = {}
    if intra and cross:
        for k in metric_keys:
            intra_vals = [s[k] for s in intra if k in s and s[k] is not None]
            cross_vals = [s[k] for s in cross if k in s and s[k] is not None]
            if intra_vals and cross_vals:
                cross_delta[f"delta_{k}"] = float(
                    np.mean(cross_vals) - np.mean(intra_vals)
                )

    return {
        "n_samples": len(per_sample),
        "aggregated": aggregated,
        "by_sample_type": by_type,
        "by_context_count": by_context,
        "cross_phase_delta": cross_delta,
    }

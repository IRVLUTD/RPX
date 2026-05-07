"""DRS sensitivity analysis: exponent, efficiency function, and anchor perturbation.

Run after a full sweep to verify ranking stability.  Produces the
tables referenced in Appendix D of the paper.

Usage::

    from rpx_benchmark.drs_sensitivity import run_sensitivity
    from rpx_benchmark.deployment import OperatingPoint

    models = {
        "DA-V2-Indoor": [OperatingPoint(...)],
        "UniDepth-V2":  [OperatingPoint(...)],
        ...
    }
    report = run_sensitivity(models)
    print(report)  # Markdown tables
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .deployment import OperatingPoint, _efficiency_score


@dataclass
class SensitivityReport:
    """Result of all three sensitivity tests."""

    exponent_taus: Dict[str, float]      # {(α,β,γ) str: τ}
    efficiency_taus: Dict[str, float]    # {func_name: τ}
    anchor_taus: Dict[float, float]      # {multiplier: τ}
    baseline_ranking: List[str]          # model names in DRS order

    def to_markdown(self) -> str:
        lines = ["# DRS Sensitivity Analysis\n"]

        lines.append("## Exponent perturbation (27 combinations)\n")
        lines.append("| (α, β, γ) | Kendall τ |")
        lines.append("|-----------|----------|")
        for key in sorted(self.exponent_taus, key=lambda k: -self.exponent_taus[k]):
            lines.append(f"| {key} | {self.exponent_taus[key]:.3f} |")
        taus = list(self.exponent_taus.values())
        lines.append(f"\nMin τ = {min(taus):.3f}, Mean τ = {np.mean(taus):.3f}, "
                      f"Max τ = {max(taus):.3f}\n")

        lines.append("## Alternative efficiency functions\n")
        lines.append("| Function | Kendall τ |")
        lines.append("|----------|----------|")
        for name, tau in sorted(self.efficiency_taus.items(), key=lambda x: -x[1]):
            lines.append(f"| {name} | {tau:.3f} |")

        lines.append("\n## Anchor perturbation\n")
        lines.append("| F_med multiplier | Kendall τ |")
        lines.append("|-----------------|----------|")
        for mult, tau in sorted(self.anchor_taus.items()):
            lines.append(f"| {mult:.2f}× | {tau:.3f} |")

        return "\n".join(lines)


def _kendall_tau(rank_a: List[str], rank_b: List[str]) -> float:
    """Kendall τ between two rankings (lists of model names)."""
    if len(rank_a) <= 1:
        return 1.0
    # Build index maps
    idx_b = {name: i for i, name in enumerate(rank_b)}
    n = len(rank_a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = i, j  # rank_a order
            b_i = idx_b.get(rank_a[i])
            b_j = idx_b.get(rank_a[j])
            if b_i is None or b_j is None:
                continue
            if (a_i - a_j) * (b_i - b_j) > 0:
                concordant += 1
            elif (a_i - a_j) * (b_i - b_j) < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def _compute_drs_custom(
    ops: List[OperatingPoint],
    f_median: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    efficiency_fn=None,
) -> float:
    """Compute DRS for one model with custom exponents / efficiency function."""
    if not ops:
        return 0.0
    best = 0.0
    for op in ops:
        tp = op.task_performance() ** alpha
        r = op.robustness() ** beta
        if efficiency_fn is not None:
            e = efficiency_fn(op.flops_g, f_median)
        else:
            e = _efficiency_score(op.flops_g, f_median)
        e = max(e, 0.0) ** gamma
        drs = tp * r * e
        best = max(best, drs)
    return best


def _rank_models(
    models: Dict[str, List[OperatingPoint]],
    f_median: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    efficiency_fn=None,
) -> List[str]:
    """Rank models by DRS (descending), return list of names."""
    scores = {}
    for name, ops in models.items():
        scores[name] = _compute_drs_custom(
            ops, f_median, alpha, beta, gamma, efficiency_fn,
        )
    return sorted(scores, key=lambda n: -scores[n])


# ── Alternative efficiency functions ──────────────────────────────

def _e_exponential(f: float, f_med: float) -> float:
    if f_med <= 0 or f <= 0:
        return 0.0
    return float(np.exp(-f / f_med))


def _e_inverse(f: float, f_med: float) -> float:
    if f <= 0:
        return 0.0
    return float(min(f_med / f, 1.0))


def _e_log(f: float, f_med: float) -> float:
    if f_med <= 0 or f <= 0:
        return 0.0
    ratio = f / f_med
    if ratio <= 1.0:
        return 1.0
    val = 1.0 / (1.0 + np.log2(ratio))
    return float(max(val, 0.0))


def _e_linear(f: float, f_med: float, f_max: float = None) -> float:
    """Linear budget: E = max(0, 1 - F/F_max)."""
    if f_max is None:
        f_max = f_med * 10  # default: 10× median
    return float(max(0.0, 1.0 - f / f_max))


# ── Main entry point ─────────────────────────────────────────────

def run_sensitivity(
    models: Dict[str, List[OperatingPoint]],
) -> SensitivityReport:
    """Run all three sensitivity tests.

    Parameters
    ----------
    models : dict
        ``{model_name: [OperatingPoint, ...]}`` — the full sweep.

    Returns
    -------
    SensitivityReport
    """
    # Compute F_median
    all_flops = [op.flops_g for ops in models.values() for op in ops]
    f_median = float(np.median(all_flops)) if all_flops else 1.0
    f_max = float(np.max(all_flops)) if all_flops else 1.0

    # Baseline ranking (equal-weight, default E)
    baseline = _rank_models(models, f_median)

    # ── Test 1: Exponent perturbation ───────────────────────────
    exponents = [0.5, 1.0, 2.0]
    exponent_taus = {}
    for a, b, g in itertools.product(exponents, repeat=3):
        ranking = _rank_models(models, f_median, alpha=a, beta=b, gamma=g)
        tau = _kendall_tau(baseline, ranking)
        exponent_taus[f"({a}, {b}, {g})"] = tau

    # ── Test 2: Alternative efficiency functions ────────────────
    efficiency_taus = {}
    for name, fn in [
        ("exponential", _e_exponential),
        ("inverse", _e_inverse),
        ("log-scaled", _e_log),
        ("linear (10× budget)", lambda f, fm: _e_linear(f, fm, f_max)),
    ]:
        ranking = _rank_models(models, f_median, efficiency_fn=fn)
        tau = _kendall_tau(baseline, ranking)
        efficiency_taus[name] = tau

    # ── Test 3: Anchor perturbation ─────────────────────────────
    anchor_taus = {}
    for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
        ranking = _rank_models(models, f_median * mult)
        tau = _kendall_tau(baseline, ranking)
        anchor_taus[mult] = tau

    return SensitivityReport(
        exponent_taus=exponent_taus,
        efficiency_taus=efficiency_taus,
        anchor_taus=anchor_taus,
        baseline_ranking=baseline,
    )

"""JEDI: Joint Empirical Desirability Index.

Implements the composite desirability score from paper §3.3. JEDI is a
direction-aware geometric mean over per-metric individual desirabilities
:math:`d_k \\in [0, 1]`. Failing any one metric drives the composite toward
zero unless the caller opts into an :math:`\\varepsilon`-floor.

Pipeline (paper Eqs. 4–6)
-------------------------

1. Look up each metric's :class:`MetricSpec` (direction, best, worst).
2. Map raw score :math:`x_k` to individual desirability::

       d_k = clip( (x_k - x_k^worst) / (x_k^best - x_k^worst), epsilon, 1 )

   The numerator and denominator share the *same sign* by construction
   (paper §3.3), so :math:`d_k \\in [0, 1]` without ad-hoc negation. The
   ``clip(·, eps, 1)`` guards out-of-bounds values.

3. Take the geometric mean::

       J = ( prod_k d_k ) ** (1/K)

4. For per-(scene, phase) aggregation, average :math:`J_{s,p}` across the
   :math:`3N` cells to get :math:`J_{model}`.

Default :math:`\\varepsilon = 0` keeps the strict "any failure tanks J"
semantics the paper argues for; practitioners who want softer behaviour
can pass a small positive ``epsilon`` (e.g. 0.01).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .exceptions import MetricError
from .logging_utils import get_logger
from .metrics.specs import MetricSpec, get_spec

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Per-cell JEDI
# --------------------------------------------------------------------------- #


@dataclass
class JEDIResult:
    """Outcome of a single (scene, phase) JEDI computation.

    Attributes
    ----------
    score : float
        The composite J = ( ∏ d_k )^(1/K) ∈ [eps^1, 1].
    individual : Dict[str, float]
        Per-metric individual desirabilities d_k.
    out_of_bounds : List[str]
        Metric names whose raw value fell outside the registered
        (worst, best) window and were clipped. Useful for flagging
        models that exceed the frozen bounds (paper §3.3).
    epsilon : float
        The ε floor used; 0.0 by default (strict).
    """

    score: float
    individual: Dict[str, float]
    out_of_bounds: List[str]
    epsilon: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "jedi": self.score,
            "individual": dict(self.individual),
            "out_of_bounds": list(self.out_of_bounds),
            "epsilon": self.epsilon,
        }


def _desirability(raw: float, spec: MetricSpec, epsilon: float) -> tuple[float, bool]:
    """Compute one individual desirability d_k and the out-of-bounds flag.

    Returns
    -------
    d_k : float
        Clipped into ``[max(epsilon, 0), 1]``.
    out_of_bounds : bool
        True iff the raw value fell outside ``(worst, best)`` before
        clipping. Helps surface models that exceed the frozen window.
    """
    span = spec.best - spec.worst
    # span is positive for higher-is-better, negative for lower-is-better;
    # numerator (raw - worst) carries the same sign by construction, so the
    # ratio is in [0, 1] when raw ∈ [worst, best] regardless of direction.
    ratio = (raw - spec.worst) / span

    floor = max(epsilon, 0.0)
    clipped = float(np.clip(ratio, floor, 1.0))
    oob = bool(ratio < 0.0 or ratio > 1.0)
    return clipped, oob


def compute_jedi(
    raw_metrics: Mapping[str, float],
    *,
    epsilon: float = 0.0,
    specs: Optional[Mapping[str, MetricSpec]] = None,
) -> JEDIResult:
    """Compute JEDI for one (scene, phase) cell.

    Parameters
    ----------
    raw_metrics : Mapping[str, float]
        ``{metric_name: raw_score}`` for the K metrics of one task on
        one (scene, phase). Keys must have specs registered (either
        via the built-in registry in :mod:`rpx_benchmark.metrics.specs`
        or via explicit ``specs=`` override).
    epsilon : float
        Lower floor on individual desirabilities. Default 0.0 (strict
        Derringer–Suich: a single failure zeros J). Pass e.g. 0.01 for
        a softer penalty (paper §3.3, "Practitioners who prefer a softer
        penalty …").
    specs : Mapping[str, MetricSpec], optional
        Override the global spec registry. When None, each name is
        resolved via :func:`get_spec`.

    Returns
    -------
    JEDIResult
        The composite score, per-metric d_k breakdown, and any
        out-of-bounds flags.

    Raises
    ------
    MetricError
        If ``raw_metrics`` is empty or any metric has no registered spec.
    """
    if not raw_metrics:
        raise MetricError(
            "compute_jedi requires at least one metric",
            hint="Pass {metric_name: raw_score, ...} with K >= 1 entries.",
        )
    if epsilon < 0.0 or epsilon >= 1.0:
        raise MetricError(
            f"epsilon must be in [0, 1), got {epsilon}",
            hint="Use epsilon=0 for strict, or a small positive value (e.g. 0.01).",
        )

    individual: Dict[str, float] = {}
    oob: List[str] = []
    for name, raw in raw_metrics.items():
        if not np.isfinite(raw):
            raise MetricError(
                f"Metric {name!r} has non-finite raw value {raw!r}",
                hint="Filter NaN/Inf upstream or impute before computing JEDI.",
            )
        spec = specs[name] if specs is not None and name in specs else get_spec(name)
        d_k, was_oob = _desirability(float(raw), spec, epsilon)
        individual[name] = d_k
        if was_oob:
            oob.append(name)

    # Geometric mean. Numerically safer via exp(mean(log(·))) when no d_k=0.
    values = np.array(list(individual.values()), dtype=np.float64)
    if (values == 0.0).any():
        score = 0.0
    else:
        score = float(np.exp(np.log(values).mean()))

    return JEDIResult(
        score=score,
        individual=individual,
        out_of_bounds=oob,
        epsilon=epsilon,
    )


# --------------------------------------------------------------------------- #
# Per-model aggregation across (scene, phase)
# --------------------------------------------------------------------------- #


@dataclass
class JEDIModelResult:
    """Per-model JEDI aggregated over all (scene, phase) cells.

    Attributes
    ----------
    jedi : float
        ``mean_{s,p} J_{s,p}`` — paper Eq. 6.
    per_phase : Dict[str, float]
        Per-phase means: ``{"clu": J_clu, "int": J_int, "cln": J_cln}``.
        Phase names are passed through from input keys.
    n_cells : int
        Number of (scene, phase) cells contributing to the mean.
    n_out_of_bounds : int
        Total count of out-of-bounds metric flags across all cells.
    """

    jedi: float
    per_phase: Dict[str, float]
    n_cells: int
    n_out_of_bounds: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "jedi": self.jedi,
            "per_phase": dict(self.per_phase),
            "n_cells": self.n_cells,
            "n_out_of_bounds": self.n_out_of_bounds,
        }


def aggregate_jedi(
    per_cell: Sequence[tuple[str, JEDIResult]],
) -> JEDIModelResult:
    """Aggregate per-cell JEDI scores into a per-model summary.

    Parameters
    ----------
    per_cell : sequence of (phase_label, JEDIResult)
        One tuple per (scene, phase) cell. ``phase_label`` is the
        column key (e.g. ``"clu"`` / ``"int"`` / ``"cln"``); rows
        within the same phase are averaged together.

    Returns
    -------
    JEDIModelResult
    """
    if not per_cell:
        raise MetricError(
            "aggregate_jedi requires at least one per-cell result",
            hint="Pass [(phase_label, JEDIResult), ...] with len >= 1.",
        )

    by_phase: Dict[str, List[float]] = {}
    total_oob = 0
    for phase_label, r in per_cell:
        by_phase.setdefault(phase_label, []).append(r.score)
        total_oob += len(r.out_of_bounds)

    per_phase = {p: float(np.mean(vs)) for p, vs in by_phase.items()}
    overall = float(np.mean([r.score for _, r in per_cell]))

    return JEDIModelResult(
        jedi=overall,
        per_phase=per_phase,
        n_cells=len(per_cell),
        n_out_of_bounds=total_oob,
    )


__all__ = [
    "JEDIResult",
    "JEDIModelResult",
    "compute_jedi",
    "aggregate_jedi",
]

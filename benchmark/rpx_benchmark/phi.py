"""State-Transition Robustness Φ via repeated-measures MANOVA.

Implements paper §3.2 from raw H (between-phase) and E (within-phase) SSCP
matrices. We do not depend on ``statsmodels``; only ``numpy`` (core) and
``scipy.stats.f`` (in the ``[analysis]`` extras) are required.

What this module provides
-------------------------

- :func:`standardise` — z-score each metric column independently across all
  ``3N`` (scene, phase) cells (paper §3.2).
- :func:`compute_phi_oneway` — one-way repeated-measures MANOVA on the phase
  factor. Returns Wilks-based and Pillai-based Φ plus an F-approximation
  p-value.
- :func:`compute_phi_per_transition` — pairwise Hotelling T² on within-scene
  paired differences for each of (Clu→Int, Int→Cln, Clu→Cln). Returns
  ``η² = T² / (T² + N − 1)`` and ``Φ = 1 − η²`` per transition.
- :func:`phi_interpretation` — Cohen anchor lookup (negligible / small /
  medium / large) for a given Φ value.

Conventions
-----------

All array shapes use the convention ``(N_scenes, n_phases, K_metrics)``.
Lower-is-better metrics should be **negated before standardisation** by
the caller (matching paper §3.2 — sign convention for the MANOVA, separate
from the direction-aware bounds policy used by JEDI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .exceptions import MetricError
from .logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #


def standardise(x: np.ndarray) -> np.ndarray:
    """Z-score each metric column across all (scene, phase) cells.

    Parameters
    ----------
    x : ndarray, shape ``(N, P, K)``
        Raw metric values: ``N`` scenes, ``P`` phases, ``K`` metrics.

    Returns
    -------
    ndarray, shape ``(N, P, K)``
        Each metric column normalised to zero mean / unit variance across
        the ``N*P`` cells. Constant columns are mapped to zeros (so they
        contribute nothing to the MANOVA test).

    Notes
    -----
    Paper §3.2: standardisation is performed *across the 3N (scene, phase)
    pairs within a task*, so the K metrics enter the MANOVA on a common
    scale and the test is invariant to the units of the underlying raw
    scores.
    """
    if x.ndim != 3:
        raise MetricError(
            f"standardise expects shape (N, P, K), got {x.shape}",
        )
    N, P, K = x.shape
    flat = x.reshape(N * P, K).astype(np.float64)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0, ddof=0)
    # Constant columns: variance 0; zero out instead of div-by-zero.
    safe = np.where(sigma > 0, sigma, 1.0)
    z = (flat - mu) / safe
    z[:, sigma == 0] = 0.0
    return z.reshape(N, P, K)


# --------------------------------------------------------------------------- #
# Collinearity diagnostics
# --------------------------------------------------------------------------- #
#
# Λ = |E| / |H + E| is numerically unstable when E is ill-conditioned, and
# ill-conditioned E almost always means two K-vector metrics are near-linear
# combinations of each other (paper §3.2 warns against this but does not
# gate the computation). We surface a warning at moderate condition numbers
# and refuse the run at severe ones — with an actionable hint pointing at
# the worst-correlated metric pair if names are available.

#: Warn when cond(E) exceeds this — Λ may be unstable but the result is usable.
COLLINEARITY_WARN_COND: float = 30.0

#: Refuse when cond(E) exceeds this — Λ is numerically unreliable.
COLLINEARITY_FAIL_COND: float = 1000.0


def _worst_correlated_pair(
    z: np.ndarray,
    metric_names: Optional[Sequence[str]],
) -> str:
    """Return a short human-readable hint identifying the most collinear
    pair of metrics in the standardised K-vector.

    Falls back to an anonymous "column i vs column j" message when
    ``metric_names`` is None.
    """
    N, P, K = z.shape
    flat = z.reshape(N * P, K)
    if K < 2 or flat.shape[0] < 2:
        return ""
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(flat, rowvar=False)
    # np.corrcoef returns NaN when a column is constant; treat those as 0.
    C = np.where(np.isfinite(C), C, 0.0)
    np.fill_diagonal(C, 0.0)
    i, j = np.unravel_index(int(np.argmax(np.abs(C))), C.shape)
    r = float(C[i, j])
    if metric_names is not None and len(metric_names) == K:
        return f"worst pair: ({metric_names[i]!r}, {metric_names[j]!r}) with |r|={abs(r):.3f}"
    return f"worst pair: (column {i}, column {j}) with |r|={abs(r):.3f}"


# --------------------------------------------------------------------------- #
# One-way repeated-measures MANOVA → Φ
# --------------------------------------------------------------------------- #


@dataclass
class PhiOneway:
    """Result of the one-way phase MANOVA.

    Attributes
    ----------
    phi_wilks : float
        :math:`\\Phi = 1 - (1 - \\Lambda^{1/s})` from Wilks' Λ.
    phi_pillai : float
        :math:`\\Phi = 1 - V/s` from Pillai's trace V.
    p_value : float
        F-approximation p-value (Wilks → Rao's F). Treats per-test p as
        descriptive; multiple-comparison correction is the caller's
        responsibility (paper §3.2, Holm-Bonferroni within (model, task)).
    n_eff : int
        Number of scenes that contributed (complete-case; scenes with
        missing predictions in any phase are dropped upstream).
    K : int
        Number of metric dimensions.
    s : int
        Rank of the hypothesis matrix, ``min(K, df_effect)``.
    wilks_lambda : float
        The raw Λ statistic, in case callers want to recompute.
    pillai_V : float
        The raw Pillai trace, same reason.
    """

    phi_wilks: float
    phi_pillai: float
    p_value: float
    n_eff: int
    K: int
    s: int
    wilks_lambda: float
    pillai_V: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "phi_wilks": self.phi_wilks,
            "phi_pillai": self.phi_pillai,
            "phi_conservative": self.phi_conservative,
            "p_value": self.p_value,
            "n_eff": self.n_eff,
            "K": self.K,
            "s": self.s,
            "wilks_lambda": self.wilks_lambda,
            "pillai_V": self.pillai_V,
        }

    @property
    def phi_conservative(self) -> float:
        """The smaller (more conservative) of Wilks/Pillai Φ.

        Per paper §3.2: *"main results report the more conservative
        estimate, with the full comparison in Appendix C."*
        """
        return min(self.phi_wilks, self.phi_pillai)


def _h_e_matrices_oneway(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build H (between-phase) and E (within-phase) SSCP matrices.

    Repeated-measures structure: scenes are subjects; phase is the
    within-subjects factor with P levels.

    Parameters
    ----------
    z : ndarray, shape ``(N, P, K)``
        Standardised metric values per (scene, phase).

    Returns
    -------
    H : ndarray, shape ``(K, K)``
    E : ndarray, shape ``(K, K)``
    df_effect : int
        ``P - 1`` for the phase main effect.
    """
    N, P, K = z.shape

    # Per-phase mean across scenes (shape (P, K)) and overall mean (shape (K,)).
    phase_means = z.mean(axis=0)
    grand_mean = z.reshape(N * P, K).mean(axis=0)

    # H: between-phase SSCP, weighted by per-cell count N.
    diffs_h = phase_means - grand_mean  # (P, K)
    H = N * (diffs_h.T @ diffs_h)

    # E: within-phase SSCP — sum over scenes & phases of (cell - phase_mean) outer.
    centered = z - phase_means[np.newaxis, :, :]  # (N, P, K)
    flat = centered.reshape(N * P, K)
    E = flat.T @ flat

    return H, E, P - 1


def _rao_f_from_wilks(Lambda: float, K: int, q: int, N: int) -> Tuple[float, int, float]:
    """Rao's F-approximation for Wilks' Λ in a one-way design.

    Returns (F, df1, df2). With P=3 phases (q=2) and any K≥2 this gives an
    exact F distribution; for q=1 or K=1 it reduces to a Hotelling/F test
    exactly.
    """
    # p = K, q = df_effect, N_obs = N*P observations but for repeated-measures
    # the relevant sample is N subjects; we follow the standard transformation
    # used in textbooks (Tabachnick & Fidell ch. 7) with v = N - 1 - (K + q + 1)/2.
    p = K
    v = N - 1 - (p + q + 1) / 2.0

    pq2 = p * p + q * q
    if pq2 == 5:
        t = 1.0
    else:
        denom = pq2 - 5.0
        if denom <= 0:
            t = 1.0
        else:
            t = np.sqrt((p * p * q * q - 4.0) / denom)

    df1 = int(p * q)
    df2 = int(round(v * t - (p * q - 2) / 2.0))
    if df2 <= 0:
        df2 = max(df1, 1)

    if Lambda <= 0.0:
        F = float("inf")
    else:
        Lambda_t = Lambda ** (1.0 / t)
        F = ((1.0 - Lambda_t) / Lambda_t) * (df2 / df1)

    return F, df1, df2


def compute_phi_oneway(
    z: np.ndarray,
    *,
    metric_names: Optional[Sequence[str]] = None,
) -> PhiOneway:
    """One-way repeated-measures MANOVA on the phase factor.

    Parameters
    ----------
    z : ndarray, shape ``(N, P, K)``
        Standardised metric values. Use :func:`standardise` first; do
        not pass raw scores.
    metric_names : sequence of str, optional
        Names of the K metrics in column order. When provided, the
        collinearity guard's error messages / warnings identify the
        offending metric pair by name; otherwise it falls back to
        column indices.

    Returns
    -------
    PhiOneway
        Wilks-based Φ, Pillai-based Φ, F-approximation p-value, and the
        raw test statistics.

    Notes
    -----
    Paper §3.2 Eq. 2: :math:`\\Phi = 1 - \\eta^2_{phase}`, with
    :math:`\\eta^2_{phase} = 1 - \\Lambda^{1/s}` (Wilks) or
    :math:`\\eta^2_{phase} = V / s` (Pillai), where
    :math:`s = \\min(K, df_{eff})` and :math:`df_{eff} = P - 1`.

    The collinearity guard proactively rejects K-vectors whose within-
    phase SSCP is numerically ill-conditioned (``cond(E) > 1000``) — a
    signature of near-duplicate metrics that would otherwise silently
    destabilise Λ. A softer warning fires at ``cond(E) > 30``.
    """
    if z.ndim != 3:
        raise MetricError(
            f"compute_phi_oneway expects shape (N, P, K), got {z.shape}",
        )
    N, P, K = z.shape
    if N < 2:
        raise MetricError(
            f"compute_phi_oneway needs at least 2 scenes, got N={N}",
            hint="Drop scenes with missing predictions earlier (complete-case).",
        )
    if P < 2:
        raise MetricError(
            f"compute_phi_oneway needs at least 2 phases, got P={P}",
        )
    if metric_names is not None and len(metric_names) != K:
        raise MetricError(
            f"metric_names has {len(metric_names)} entries, expected K={K}",
        )

    H, E, df_effect = _h_e_matrices_oneway(z)
    s = min(K, df_effect)

    # Collinearity guard — cond(E) diagnoses near-duplicate K-vector metrics
    # before the log-determinant path silently produces a garbage Λ.
    if K >= 2:
        cond_E = float(np.linalg.cond(E))
        if not np.isfinite(cond_E) or cond_E > COLLINEARITY_FAIL_COND:
            hint = _worst_correlated_pair(z, metric_names)
            raise MetricError(
                f"Within-phase SSCP E is near-singular "
                f"(cond(E) = {cond_E:.2e} > {COLLINEARITY_FAIL_COND:.0e})",
                hint=(
                    "Two or more K-vector metrics are near-linear combinations "
                    "of each other; drop one before running MANOVA. "
                    + hint
                ).strip(),
            )
        if cond_E > COLLINEARITY_WARN_COND:
            pair = _worst_correlated_pair(z, metric_names)
            log.warning(
                "compute_phi_oneway: cond(E) = %.1f (threshold %.0f) — "
                "moderate collinearity may destabilise Λ. %s",
                cond_E, COLLINEARITY_WARN_COND, pair,
            )

    # Wilks' Λ = |E| / |H + E|.
    sign_e, logdet_e = np.linalg.slogdet(E)
    sign_he, logdet_he = np.linalg.slogdet(H + E)
    if sign_e <= 0 or sign_he <= 0:
        raise MetricError(
            "Wilks' Λ requires positive-definite E and (H+E); got singular matrices",
            hint=(
                "This usually means K > N (more metrics than scenes) or perfectly "
                "collinear metrics. Drop redundant metrics or increase scene count."
            ),
        )
    Lambda = float(np.exp(logdet_e - logdet_he))
    eta2_wilks = 1.0 - Lambda ** (1.0 / s)
    phi_wilks = 1.0 - eta2_wilks

    # Pillai's trace V = tr( H (H + E)^{-1} ).
    HE_inv = np.linalg.solve(H + E, H)
    V = float(np.trace(HE_inv))
    eta2_pillai = V / s
    phi_pillai = 1.0 - eta2_pillai

    # F-approximation p-value via Wilks → Rao's F.
    try:
        from scipy.stats import f as f_dist  # lazy: scipy is an [analysis] extra
    except ImportError as e:
        raise MetricError(
            "scipy is required for Φ p-values",
            hint="pip install 'rpx-benchmark[analysis]'",
        ) from e
    F, df1, df2 = _rao_f_from_wilks(Lambda, K, df_effect, N)
    p_value = float(1.0 - f_dist.cdf(F, df1, df2)) if np.isfinite(F) else 0.0

    return PhiOneway(
        phi_wilks=float(np.clip(phi_wilks, 0.0, 1.0)),
        phi_pillai=float(np.clip(phi_pillai, 0.0, 1.0)),
        p_value=p_value,
        n_eff=N,
        K=K,
        s=s,
        wilks_lambda=Lambda,
        pillai_V=V,
    )


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval for Φ
# --------------------------------------------------------------------------- #
#
# The one-shot Rao's-F p-value from ``compute_phi_oneway`` says whether the
# phase effect is significant; it does not say how tight the Φ estimate is.
# A scene-level nonparametric bootstrap answers that: resample scenes (rows
# of the cube) with replacement, recompute Φ per bootstrap, then report the
# percentile CI. This is the standard reviewer-defensible way to communicate
# uncertainty around a small-sample MANOVA effect size.


@dataclass
class PhiBootstrapCI:
    """Bootstrap distribution of Φ over resampled scene sets.

    Attributes
    ----------
    phi_wilks_mean, phi_pillai_mean : float
        Mean of the bootstrap Φ estimates (point estimate, robust to the
        original single-fit result).
    phi_wilks_ci : tuple[float, float]
    phi_pillai_ci : tuple[float, float]
        Percentile confidence intervals ``(low, high)`` at the requested
        confidence level.
    phi_wilks_se, phi_pillai_se : float
        Bootstrap standard error (standard deviation of the resampled Φ).
    n_boot : int
        Number of bootstrap resamples requested.
    n_success : int
        Number of resamples that yielded a valid Φ (non-singular E, no
        cond(E) failure). Resamples that fail are skipped and counted
        towards ``n_failed``.
    n_failed : int
    ci_level : float
        Confidence level used (e.g. 0.95).
    """

    phi_wilks_mean: float
    phi_pillai_mean: float
    phi_wilks_ci: Tuple[float, float]
    phi_pillai_ci: Tuple[float, float]
    phi_wilks_se: float
    phi_pillai_se: float
    n_boot: int
    n_success: int
    n_failed: int
    ci_level: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "phi_wilks_mean": self.phi_wilks_mean,
            "phi_pillai_mean": self.phi_pillai_mean,
            "phi_wilks_ci_low": self.phi_wilks_ci[0],
            "phi_wilks_ci_high": self.phi_wilks_ci[1],
            "phi_pillai_ci_low": self.phi_pillai_ci[0],
            "phi_pillai_ci_high": self.phi_pillai_ci[1],
            "phi_wilks_se": self.phi_wilks_se,
            "phi_pillai_se": self.phi_pillai_se,
            "n_boot": self.n_boot,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "ci_level": self.ci_level,
        }


def bootstrap_phi_oneway(
    z: np.ndarray,
    *,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    metric_names: Optional[Sequence[str]] = None,
    min_success_frac: float = 0.5,
) -> PhiBootstrapCI:
    """Scene-level nonparametric bootstrap of one-way Φ.

    Parameters
    ----------
    z : ndarray, shape ``(N, P, K)``
        Standardised metric values (use :func:`standardise` first).
        Bootstrap resamples along the scene axis with replacement; the
        (phase, metric) coupling within each scene is preserved.
    n_boot : int
        Number of bootstrap resamples. Defaults to 1000, which is
        adequate for percentile CIs at the 95% level.
    ci_level : float
        Confidence level in ``(0, 1)``. Defaults to 0.95.
    seed : int, optional
        Reproducible seed for the resampler. ``None`` uses fresh entropy.
    metric_names : sequence of str, optional
        Forwarded to :func:`compute_phi_oneway` on every resample; used
        only for collinearity-guard hints if a resample fails.
    min_success_frac : float
        Raise :class:`MetricError` if fewer than this fraction of
        resamples yield a valid Φ. Default 0.5 — protects against pipes
        where standardisation upstream produced degenerate columns.

    Returns
    -------
    PhiBootstrapCI
    """
    if z.ndim != 3:
        raise MetricError(f"bootstrap_phi_oneway expects (N, P, K), got {z.shape}")
    N, P, K = z.shape
    if N < 2:
        raise MetricError(
            f"bootstrap_phi_oneway needs at least 2 scenes, got N={N}",
        )
    if not (0.0 < ci_level < 1.0):
        raise MetricError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_boot < 50:
        raise MetricError(
            f"n_boot={n_boot} is too small for a stable percentile CI; use ≥ 50",
        )

    rng = np.random.default_rng(seed)
    wilks_samples: List[float] = []
    pillai_samples: List[float] = []
    n_failed = 0

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        z_boot = z[idx]
        try:
            r = compute_phi_oneway(z_boot, metric_names=metric_names)
        except MetricError:
            n_failed += 1
            continue
        wilks_samples.append(r.phi_wilks)
        pillai_samples.append(r.phi_pillai)

    n_success = len(wilks_samples)
    if n_success < int(min_success_frac * n_boot):
        raise MetricError(
            f"bootstrap failed on {n_failed}/{n_boot} resamples "
            f"(only {n_success} succeeded)",
            hint=(
                "The K-vector is likely marginally-conditioned — many "
                "scene resamples yield singular E. Reduce K or increase "
                "the scene pool before bootstrapping."
            ),
        )

    wilks_arr = np.array(wilks_samples, dtype=np.float64)
    pillai_arr = np.array(pillai_samples, dtype=np.float64)
    alpha = 1.0 - ci_level
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

    return PhiBootstrapCI(
        phi_wilks_mean=float(wilks_arr.mean()),
        phi_pillai_mean=float(pillai_arr.mean()),
        phi_wilks_ci=(
            float(np.percentile(wilks_arr, lo_q)),
            float(np.percentile(wilks_arr, hi_q)),
        ),
        phi_pillai_ci=(
            float(np.percentile(pillai_arr, lo_q)),
            float(np.percentile(pillai_arr, hi_q)),
        ),
        phi_wilks_se=float(wilks_arr.std(ddof=1)) if n_success > 1 else 0.0,
        phi_pillai_se=float(pillai_arr.std(ddof=1)) if n_success > 1 else 0.0,
        n_boot=n_boot,
        n_success=n_success,
        n_failed=n_failed,
        ci_level=ci_level,
    )


# --------------------------------------------------------------------------- #
# Pairwise Hotelling T² per transition
# --------------------------------------------------------------------------- #


@dataclass
class PhiTransition:
    """Result of one pairwise Hotelling T² test.

    Attributes
    ----------
    label : str
        Human-readable transition label (e.g. ``"0->1"``).
    phi : float
        :math:`\\Phi_{i \\to j} = 1 - T^2 / (T^2 + N - 1)`.
    eta2 : float
        :math:`\\eta^2 = T^2 / (T^2 + N - 1)`.
    T2 : float
        The Hotelling :math:`T^2` statistic on within-scene paired
        differences.
    p_value : float
        F-converted p-value.
    p_holm : Optional[float]
        Holm-Bonferroni-adjusted p (set by
        :func:`compute_phi_per_transition` when multiple transitions
        are tested jointly).
    n_eff : int
        Number of scenes contributing to the paired test.
    """

    label: str
    phi: float
    eta2: float
    T2: float
    p_value: float
    p_holm: Optional[float]
    n_eff: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "phi": self.phi,
            "eta2": self.eta2,
            "T2": self.T2,
            "p_value": self.p_value,
            "p_holm": self.p_holm,
            "n_eff": self.n_eff,
        }


def _hotelling_one_sample(d: np.ndarray) -> Tuple[float, float, float, int, int]:
    """One-sample Hotelling T² for paired differences.

    Parameters
    ----------
    d : ndarray, shape ``(N, K)``
        Paired-difference vectors (one row per scene).

    Returns
    -------
    T2 : float
        ``T^2 = N · d̄' S_d^{-1} d̄``.
    p_value : float
        F-converted, ``F = ((N-K) / ((N-1) K)) · T^2 ~ F(K, N-K)``.
    eta2 : float
        ``T^2 / (T^2 + N - 1)``.
    df1, df2 : int
    """
    if d.ndim != 2:
        raise MetricError(f"paired differences must be 2D, got {d.shape}")
    N, K = d.shape
    if N <= K:
        raise MetricError(
            f"Hotelling T² requires N > K, got N={N}, K={K}",
            hint="Drop metrics or increase scene count.",
        )
    d_bar = d.mean(axis=0)
    S_d = np.cov(d, rowvar=False, ddof=1)
    try:
        S_inv = np.linalg.inv(S_d)
    except np.linalg.LinAlgError as e:
        raise MetricError(
            "Within-difference covariance matrix is singular",
            hint="Drop collinear metrics before computing Φ_per_transition.",
        ) from e
    T2 = float(N * d_bar @ S_inv @ d_bar)

    # F-distribution conversion.
    df1 = K
    df2 = N - K
    F = ((N - K) / ((N - 1) * K)) * T2

    try:
        from scipy.stats import f as f_dist
    except ImportError as e:
        raise MetricError(
            "scipy is required for Hotelling T² p-values",
            hint="pip install 'rpx-benchmark[analysis]'",
        ) from e
    p_value = float(1.0 - f_dist.cdf(F, df1, df2))

    eta2 = T2 / (T2 + N - 1)
    return T2, p_value, eta2, df1, df2


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Parameters
    ----------
    p_values : list of float
        Raw p-values (any order).

    Returns
    -------
    list of float
        Adjusted p-values in the same order as input. Monotonically
        non-decreasing under the Holm rule and clipped to ``[0, 1]``.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank_i, idx in enumerate(order):
        adj = (m - rank_i) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def compute_phi_per_transition(
    z: np.ndarray,
    *,
    transitions: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2)),
    apply_holm: bool = True,
) -> Dict[str, PhiTransition]:
    """Pairwise Hotelling T² → Φ for each phase transition.

    Parameters
    ----------
    z : ndarray, shape ``(N, P, K)``
        Standardised metric values.
    transitions : tuple of (p1, p2)
        Phase-index pairs to test. Defaults to all three: Clu→Int,
        Int→Cln, Clu→Cln.
    apply_holm : bool
        Apply Holm-Bonferroni correction across the requested
        transitions (paper §3.2: family-wise control for the 3 pairwise
        transitions per (model, task)).

    Returns
    -------
    dict
        ``{"0->1": PhiTransition, "1->2": ..., "0->2": ...}``.
    """
    if z.ndim != 3:
        raise MetricError(f"expects (N, P, K), got {z.shape}")
    N, P, _ = z.shape

    raw_results: Dict[str, PhiTransition] = {}
    raw_ps: list[float] = []
    keys: list[str] = []
    for p1, p2 in transitions:
        if not (0 <= p1 < P) or not (0 <= p2 < P):
            raise MetricError(
                f"transition ({p1}, {p2}) out of range for P={P}",
            )
        if p1 == p2:
            raise MetricError(f"degenerate transition ({p1}, {p1})")
        d = z[:, p2, :] - z[:, p1, :]
        T2, p_raw, eta2, _, _ = _hotelling_one_sample(d)
        phi = 1.0 - eta2
        key = f"{p1}->{p2}"
        raw_results[key] = PhiTransition(
            label=key,
            phi=float(np.clip(phi, 0.0, 1.0)),
            eta2=float(np.clip(eta2, 0.0, 1.0)),
            T2=T2,
            p_value=p_raw,
            p_holm=None,
            n_eff=N,
        )
        raw_ps.append(p_raw)
        keys.append(key)

    if apply_holm and raw_ps:
        adj = _holm_bonferroni(raw_ps)
        for k, p_h in zip(keys, adj, strict=True):
            r = raw_results[k]
            raw_results[k] = PhiTransition(
                label=r.label,
                phi=r.phi,
                eta2=r.eta2,
                T2=r.T2,
                p_value=r.p_value,
                p_holm=p_h,
                n_eff=r.n_eff,
            )

    return raw_results


# --------------------------------------------------------------------------- #
# Interpretation rubric (Cohen anchors)
# --------------------------------------------------------------------------- #


def phi_interpretation(phi: float) -> str:
    """Map a Φ value to a Cohen partial-η² effect-size label.

    Thresholds (paper §3.2):

    - ``η² < 0.01``  →  Φ > 0.99   →  "negligible"
    - ``0.01 ≤ η² < 0.06``  →  0.94 < Φ ≤ 0.99  →  "small"
    - ``0.06 ≤ η² < 0.14``  →  0.86 < Φ ≤ 0.94  →  "medium"
    - ``η² ≥ 0.14``  →  Φ ≤ 0.86  →  "large"
    """
    if not (0.0 <= phi <= 1.0):
        raise MetricError(f"Φ must be in [0, 1], got {phi}")
    if phi > 0.99:
        return "negligible"
    if phi > 0.94:
        return "small"
    if phi > 0.86:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Two-way mixed-design MANOVA (paper §3.2)
# --------------------------------------------------------------------------- #
#
# Phase is within-subjects (each scene appears in all P phases), difficulty
# tier is between-subjects (each scene belongs to exactly one tier). The
# three effects we report:
#
#   - Phase main effect:           does the per-scene phase profile differ
#                                  from a flat one, pooled across tiers?
#   - Difficulty main effect:      do tier-averaged scene means differ?
#   - Phase × difficulty interaction: does the phase profile differ across
#                                  tiers? (The headline number paper §3.2
#                                  calls η²_{phase×diff}.)
#
# Standard recipe: build orthonormal phase contrasts C (P × (P−1)); for each
# scene, form the contrast vector c_s = C^T x_s ∈ R^{(P−1) × K}, flatten to
# R^{(P−1)K}. Then:
#
#   - Phase main effect  ≈ one-sample Hotelling on {c_s} vs. zero.
#   - Phase × diff       ≈ one-way between-groups MANOVA on {c_s} by tier.
#   - Diff main effect   ≈ one-way between-groups MANOVA on
#                          {m_s = mean_p x_{s,p}} by tier.
#
# This is the standard reduction used in repeated-measures multivariate
# analysis (Tabachnick & Fidell, ch. 8). It's not the only formulation
# possible but it's clean and the effect-size η² values match what the
# paper §3.2 asks for.


@dataclass
class PhiMixedDesign:
    """Result of the two-way mixed-design MANOVA.

    Attributes
    ----------
    phi_per_tier : Dict[str, PhiOneway]
        Per-tier one-way Φ (computed by restricting the cell tensor to
        each tier and running :func:`compute_phi_oneway`).
        Paper notation: $\\Phi_\\text{E}, \\Phi_\\text{M}, \\Phi_\\text{H}$.
    eta2_phase : float
        Main-effect partial $\\eta^2$ for the phase factor, pooled
        across tiers.
    eta2_diff : float
        Main-effect partial $\\eta^2$ for the difficulty factor.
    eta2_interaction : float
        Phase × difficulty interaction partial $\\eta^2$. The headline
        "do harder scenes amplify the phase effect?" number.
    phi_phase, phi_diff, phi_interaction : float
        Complements of the η² values, ∈ [0, 1].
    p_phase, p_diff, p_interaction : float
        F-approximation p-values.
    K : int
        Metric dimension carried into the contrast space.
    n_scenes : int
        Number of scenes that contributed.
    tier_counts : Dict[str, int]
        Per-tier scene counts.
    notes : List[str]
    """

    phi_per_tier: Dict[str, PhiOneway]
    eta2_phase: float
    eta2_diff: float
    eta2_interaction: float
    phi_phase: float
    phi_diff: float
    phi_interaction: float
    p_phase: float
    p_diff: float
    p_interaction: float
    K: int
    n_scenes: int
    tier_counts: Dict[str, int]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "phi_per_tier": {t: r.to_dict() for t, r in self.phi_per_tier.items()},
            "eta2_phase": self.eta2_phase,
            "eta2_diff": self.eta2_diff,
            "eta2_interaction": self.eta2_interaction,
            "phi_phase": self.phi_phase,
            "phi_diff": self.phi_diff,
            "phi_interaction": self.phi_interaction,
            "p_phase": self.p_phase,
            "p_diff": self.p_diff,
            "p_interaction": self.p_interaction,
            "K": self.K,
            "n_scenes": self.n_scenes,
            "tier_counts": dict(self.tier_counts),
            "notes": list(self.notes),
        }


def _orthonormal_phase_contrasts(P: int) -> np.ndarray:
    """Return a $P \\times (P-1)$ matrix whose columns are an
    orthonormal basis for the subspace orthogonal to $(1,\\dots,1)$.

    Built via QR of the centering matrix; numerically stable for any
    reasonable P. Caller does not need to know the specific basis ---
    Λ, V, and η² are invariant under orthonormal change of basis on
    the contrast subspace.
    """
    if P < 2:
        raise MetricError(f"need P>=2 phases for contrasts, got {P}")
    centering = np.eye(P) - np.ones((P, P)) / P
    Q, _ = np.linalg.qr(centering[:, : P - 1])
    return Q  # shape (P, P-1)


def _one_way_between_groups(
    X: np.ndarray,
    group_labels: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build H, E matrices for a one-way between-groups MANOVA.

    Parameters
    ----------
    X : (N, D) ndarray
        Per-subject observation vectors.
    group_labels : length-N sequence
        Group assignment per row.

    Returns
    -------
    H : (D, D) between-groups SSCP.
    E : (D, D) within-groups (pooled) SSCP.
    df_h : int
        Hypothesis degrees of freedom: G - 1.
    G : int
        Number of distinct groups.
    """
    N, D = X.shape
    if N != len(group_labels):
        raise MetricError(
            f"length mismatch: X has {N} rows, group_labels has {len(group_labels)}"
        )
    unique = sorted(set(group_labels), key=str)
    G = len(unique)
    if G < 2:
        raise MetricError(
            f"between-groups MANOVA needs >=2 groups, got {G}",
            hint="The interaction term is undefined with a single tier.",
        )

    grand = X.mean(axis=0)
    H = np.zeros((D, D), dtype=np.float64)
    E = np.zeros((D, D), dtype=np.float64)
    for g in unique:
        mask = np.array([gl == g for gl in group_labels], dtype=bool)
        n_g = int(mask.sum())
        if n_g < 1:
            continue
        X_g = X[mask]
        mean_g = X_g.mean(axis=0)
        diff_g = (mean_g - grand)[:, None]
        H += n_g * (diff_g @ diff_g.T)
        centered = X_g - mean_g
        E += centered.T @ centered

    return H, E, G - 1, G


def _wilks_eta_squared_from_he(
    H: np.ndarray, E: np.ndarray, K: int, df_h: int, n_obs: int
) -> Tuple[float, float, float]:
    """Convert (H, E, df_h, n_obs) to (eta²_wilks, F, p_value)."""
    sign_e, logdet_e = np.linalg.slogdet(E)
    sign_he, logdet_he = np.linalg.slogdet(H + E)
    if sign_e <= 0 or sign_he <= 0:
        raise MetricError(
            "Wilks Λ requires positive-definite E and (H+E); got singular matrices",
            hint=(
                "Drop collinear metrics, increase scene count, or merge "
                "tiers — the input is too rank-deficient for the test."
            ),
        )
    Lambda = float(np.exp(logdet_e - logdet_he))
    s = max(1, min(K, df_h))
    eta2 = 1.0 - Lambda ** (1.0 / s)
    F, df1, df2 = _rao_f_from_wilks(Lambda, K, df_h, n_obs)

    try:
        from scipy.stats import f as f_dist
    except ImportError as e:
        raise MetricError(
            "scipy required for F-approximation p-values",
            hint="pip install 'rpx-benchmark[analysis]'",
        ) from e

    p_value = float(1.0 - f_dist.cdf(F, df1, df2)) if np.isfinite(F) else 0.0
    return float(np.clip(eta2, 0.0, 1.0)), F, p_value


def compute_phi_mixed_design(
    z: np.ndarray,
    tier_by_scene: Sequence[str],
) -> PhiMixedDesign:
    """Two-way mixed-design MANOVA: phase within, tier between.

    Parameters
    ----------
    z : ndarray, shape ``(N, P, K)``
        Standardised metric values (use :func:`standardise` first).
    tier_by_scene : length-N sequence
        Tier label per scene (e.g. ``"easy"``, ``"medium"``, ``"hard"``).
        Must align with the scene axis of ``z``.

    Returns
    -------
    PhiMixedDesign
    """
    if z.ndim != 3:
        raise MetricError(f"compute_phi_mixed_design expects (N, P, K), got {z.shape}")
    N, P, K = z.shape
    if N != len(tier_by_scene):
        raise MetricError(
            f"tier_by_scene length {len(tier_by_scene)} != N={N}",
        )
    if P < 2:
        raise MetricError(f"need P>=2 phases, got {P}")
    if N < 4:
        raise MetricError(
            f"need at least 4 scenes for a meaningful mixed design, got {N}",
        )

    notes: List[str] = []

    # ---- Per-tier Φ (just restrict compute_phi_oneway to each subset) ----
    tier_counts: Dict[str, int] = {}
    phi_per_tier: Dict[str, PhiOneway] = {}
    tier_indices: Dict[str, List[int]] = {}
    for i, t in enumerate(tier_by_scene):
        tier_indices.setdefault(str(t), []).append(i)
    for tier, indices in tier_indices.items():
        tier_counts[tier] = len(indices)
        if len(indices) < 2:
            notes.append(f"tier {tier!r}: only {len(indices)} scene(s); skipping per-tier Φ")
            continue
        try:
            phi_per_tier[tier] = compute_phi_oneway(z[indices])
        except MetricError as e:
            notes.append(f"tier {tier!r}: compute_phi_oneway failed ({e})")

    # ---- Build contrast vectors c_s ∈ R^{(P-1)*K} per scene ----
    C = _orthonormal_phase_contrasts(P)  # (P, P-1)
    # c_s = C^T x_s gives (P-1, K); flatten to ((P-1)*K,)
    c = np.einsum("ip,npk->nik", C.T, z)  # (N, P-1, K)
    D_c = (P - 1) * K
    c_flat = c.reshape(N, D_c)

    # ---- Phase main effect: one-sample Hotelling on c_s vs zero, pooled
    # over all scenes. The pool may include tier-mean shifts; we therefore
    # center within-tier before testing the grand mean, which is the
    # standard within-subjects-only main-effect formulation.
    # H_phase = N * c̄_centered c̄_centered^T,  E_phase = within-tier SSCP.
    eta2_phase = phi_phase = p_phase = 0.0
    try:
        unique_tiers = sorted(set(str(t) for t in tier_by_scene), key=str)
        H_phase = np.zeros((D_c, D_c))
        E_phase = np.zeros((D_c, D_c))
        tier_centered_means_weighted_sum = np.zeros(D_c)
        n_total = 0
        for tier in unique_tiers:
            mask = np.array([str(t) == tier for t in tier_by_scene], dtype=bool)
            c_g = c_flat[mask]
            if c_g.shape[0] == 0:
                continue
            mean_g = c_g.mean(axis=0)
            tier_centered_means_weighted_sum += c_g.shape[0] * mean_g
            n_total += c_g.shape[0]
            centered_g = c_g - mean_g
            E_phase += centered_g.T @ centered_g
        grand_c = tier_centered_means_weighted_sum / max(1, n_total)
        # H_phase tests grand_c == 0 (subject-pooled phase profile is flat).
        # With df_h = 1, we get a within-subjects T² style test.
        H_phase = n_total * np.outer(grand_c, grand_c)
        eta2_phase, _F_phase, p_phase = _wilks_eta_squared_from_he(
            H_phase, E_phase, K=D_c, df_h=1, n_obs=n_total
        )
        phi_phase = float(np.clip(1.0 - eta2_phase, 0.0, 1.0))
    except MetricError as e:
        notes.append(f"phase main effect: {e}")

    # ---- Phase × difficulty interaction: one-way between-groups MANOVA
    # on the contrast vectors {c_s} with tier as the grouping factor.
    eta2_int = phi_int = p_int = 0.0
    try:
        H_int, E_int, df_h_int, _G = _one_way_between_groups(c_flat, list(map(str, tier_by_scene)))
        eta2_int, _F_int, p_int = _wilks_eta_squared_from_he(
            H_int, E_int, K=D_c, df_h=df_h_int, n_obs=N
        )
        phi_int = float(np.clip(1.0 - eta2_int, 0.0, 1.0))
    except MetricError as e:
        notes.append(f"phase × difficulty interaction: {e}")

    # ---- Difficulty main effect: one-way between-groups MANOVA on
    # phase-averaged scene means.
    eta2_diff = phi_diff = p_diff = 0.0
    try:
        m_s = z.mean(axis=1)  # (N, K)
        H_d, E_d, df_h_d, _G2 = _one_way_between_groups(m_s, list(map(str, tier_by_scene)))
        eta2_diff, _F_d, p_diff = _wilks_eta_squared_from_he(
            H_d, E_d, K=K, df_h=df_h_d, n_obs=N
        )
        phi_diff = float(np.clip(1.0 - eta2_diff, 0.0, 1.0))
    except MetricError as e:
        notes.append(f"difficulty main effect: {e}")

    return PhiMixedDesign(
        phi_per_tier=phi_per_tier,
        eta2_phase=eta2_phase,
        eta2_diff=eta2_diff,
        eta2_interaction=eta2_int,
        phi_phase=phi_phase,
        phi_diff=phi_diff,
        phi_interaction=phi_int,
        p_phase=p_phase,
        p_diff=p_diff,
        p_interaction=p_int,
        K=K,
        n_scenes=N,
        tier_counts=tier_counts,
        notes=notes,
    )


__all__ = [
    "COLLINEARITY_FAIL_COND",
    "COLLINEARITY_WARN_COND",
    "PhiBootstrapCI",
    "PhiMixedDesign",
    "PhiOneway",
    "PhiTransition",
    "bootstrap_phi_oneway",
    "compute_phi_mixed_design",
    "compute_phi_oneway",
    "compute_phi_per_transition",
    "phi_interpretation",
    "standardise",
]

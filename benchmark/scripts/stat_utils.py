"""Shared statistical aggregation helpers for the benchmark pipelines.

Exists so per-stage timing (`run_depth.py`) and per-metric aggregation
(`comprehensive_depth_metrics.py`) report uncertainty in the same shape:
mean, median, std, p5/p95, plus 95% confidence intervals on the mean
computed two ways (t-distribution asymptotic + percentile bootstrap).

Latency and depth-error distributions are typically right-skewed, so the
bootstrap CI is the trustworthy one for small n; the t-CI matches
asymptotically for n > 30.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


#: Asymptotic-normal critical value for a 95% two-sided CI. For n>30
#: this matches the t-distribution to <1%; for small n the bootstrap
#: CI is the trustworthy one and we report both.
Z_CRITICAL_95 = 1.96

#: Number of bootstrap resamples for the percentile CI on the mean.
#: 2000 keeps Monte Carlo error <1% of the CI half-width and runs in
#: <100 ms even for n in the low thousands.
_BOOTSTRAP_REPLICATES = 2000

#: Bootstrap RNG seed. Sourced from the project-wide canonical seed
#: (``rpx_benchmark.determinism.RPX_SEED`` = MMDDYYYY 05/06/2026) so the
#: CI produced here, the test data in ``tests/test_stat_utils.py``, and
#: the ORD pair sampler in ``comprehensive_depth_metrics.py`` all share
#: one seed --- meaning result.json diffs are bit-deterministic across
#: reruns and across modules.
from rpx_benchmark.determinism import RPX_SEED as _BOOTSTRAP_SEED  # noqa: E402


def summarize_with_ci(values: Iterable[float]) -> dict:
    """Aggregate a sample with mean ± 95% CI.

    Returns a dict with keys: ``mean median std p5 p95 ci95_low_t
    ci95_high_t ci95_low_boot ci95_high_boot n``. Empty inputs yield a
    zero-filled dict with ``n=0`` so downstream serialization stays
    structurally homogeneous.
    """
    a = np.asarray(list(values), dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "p5": 0.0, "p95": 0.0,
            "ci95_low_t": 0.0, "ci95_high_t": 0.0,
            "ci95_low_boot": 0.0, "ci95_high_boot": 0.0, "n": 0,
        }

    mean = float(a.mean())
    std  = float(a.std(ddof=1)) if n > 1 else 0.0
    se   = (std / (n ** 0.5)) if n > 0 else 0.0
    ci_low_t  = mean - Z_CRITICAL_95 * se
    ci_high_t = mean + Z_CRITICAL_95 * se

    if n > 1:
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        idx = rng.integers(0, n, size=(_BOOTSTRAP_REPLICATES, n))
        boot_means = a[idx].mean(axis=1)
        ci_low_boot  = float(np.percentile(boot_means,  2.5))
        ci_high_boot = float(np.percentile(boot_means, 97.5))
    else:
        ci_low_boot, ci_high_boot = mean, mean

    return {
        "mean":           mean,
        "median":         float(np.median(a)),
        "std":            std,
        "p5":             float(np.percentile(a,  5)),
        "p95":            float(np.percentile(a, 95)),
        "ci95_low_t":     ci_low_t,
        "ci95_high_t":    ci_high_t,
        "ci95_low_boot":  ci_low_boot,
        "ci95_high_boot": ci_high_boot,
        "n":              n,
    }


def aggregate_per_sample_with_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    drop_keys: Sequence[str] = ("id", "scene_id", "phase", "alignment"),
) -> dict:
    """Take a list of per-sample dicts and aggregate every numeric key
    with mean + 95% CI. Useful for the comprehensive metric basket where
    each row carries 20+ keys (AbsRel, RMSE, δ1/2/3, depth-band, mask, ...).
    """
    keys = sorted({k for r in rows for k in r if k not in set(drop_keys)})
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in rows
                if k in r and isinstance(r[k], (int, float))
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            out[k] = summarize_with_ci(vals)
    return out

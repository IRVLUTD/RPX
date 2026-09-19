"""Frame-budget sweep runner and degradation analysis for video tasks.

Runs a video-task pipeline at multiple frame budgets (e.g. {50, 100, 150,
250}) on the same clips and produces a degradation analysis that answers:

* **Temporal Cue Value (TCV):** per model × metric, how much does the full
  clip improve over the smallest budget?
* **Area Under Degradation Curve (AUDC):** integral of metric vs. budget,
  normalized — models with flat curves are budget-robust.
* **Critical Budget:** smallest budget where metric is within ``tol`` of
  full-budget performance.
* **Paired test:** Wilcoxon signed-rank across scenes at min vs. max budget.

Design note
-----------
The frame-budget ablation is a **first-class diagnostic** unique to RPX
(no other benchmark offers it). It tests whether temporal cues actually
help each model and at what granularity. The sweep is task-agnostic:
any task that uses :class:`~rpx_benchmark.video_loader.VideoDepthDataset`
with ``frame_budget`` + ``sampling`` benefits.  Object tracking will
plug in through the same harness.

The Φ MANOVA is computed **only** on the full-budget (canonical) run.
Budget ablations are reported as a separate degradation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "BudgetSweepConfig",
    "DegradationResult",
    "run_budget_sweep",
    "degradation_analysis",
]

#: Default frame budgets for the temporal resolution ablation.
DEFAULT_BUDGETS: Tuple[int, ...] = (50, 100, 150, 250)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class BudgetSweepConfig:
    """Configuration for a frame-budget sweep.

    Parameters
    ----------
    budgets : sequence of int
        Frame budgets to evaluate. The largest budget is treated as
        the canonical "full" run for degradation analysis. Example:
        ``(50, 100, 150, 250)``.
    sampling : str
        Sampling mode for frame selection. ``"stride"`` (default) gives
        uniform-with-endpoint-anchored selection — equivalent to
        "what if your camera ran at a lower FPS". ``"fps_se3"``
        selects frames that maximise spatial coverage of the camera path.
    metric_keys : list of str or None
        If provided, restrict the degradation analysis to these metric
        keys. If ``None``, analyze all numeric columns in the cell logs.
    tol : float
        Tolerance for "critical budget" — the smallest budget where
        metric is within ``tol`` (relative) of full-budget performance.
        Default: 0.05 (5%).
    """

    budgets: Sequence[int] = DEFAULT_BUDGETS
    sampling: str = "stride"
    metric_keys: Optional[List[str]] = None
    tol: float = 0.05


# --------------------------------------------------------------------------- #
# Degradation result
# --------------------------------------------------------------------------- #


@dataclass
class DegradationResult:
    """Per-model × per-metric degradation analysis across frame budgets.

    Attributes
    ----------
    model : str
    metric : str
    direction : str
        ``"lower"`` or ``"higher"`` — whether lower/higher is better.
    budgets : list[int]
        Frame budgets (ascending).
    values : list[float]
        Mean metric value at each budget (over scenes/phases).
    per_scene_values : dict[int, list[float]]
        Per-scene metric values at each budget (for paired tests).
    tcv : float
        Temporal Cue Value = metric@max_budget - metric@min_budget
        (sign-adjusted so positive = temporal cues help).
    audc : float
        Area Under Degradation Curve, normalized by metric@max_budget.
    critical_budget : int or None
        Smallest budget where metric is within ``tol`` of full-budget.
        ``None`` if no budget meets the tolerance.
    wilcoxon_p : float or None
        p-value of Wilcoxon signed-rank test (min vs max budget).
        ``None`` if insufficient data.
    """

    model: str = ""
    metric: str = ""
    direction: str = "lower"
    budgets: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    per_scene_values: Dict[int, List[float]] = field(default_factory=dict)
    tcv: float = float("nan")
    audc: float = float("nan")
    critical_budget: Optional[int] = None
    wilcoxon_p: Optional[float] = None


# --------------------------------------------------------------------------- #
# Degradation analysis
# --------------------------------------------------------------------------- #


def degradation_analysis(
    cells_by_budget: Dict[int, List[dict]],
    metric_keys: Optional[List[str]] = None,
    tol: float = 0.05,
    model_name: str = "model",
) -> List[DegradationResult]:
    """Compute degradation statistics from cell-log dicts at each budget.

    Parameters
    ----------
    cells_by_budget : dict[int, list[dict]]
        Mapping from frame_budget → list of cell-log row dicts. Each
        dict must contain the metric columns + ``scene`` (or
        ``scene_id``) + ``phase`` for scene-level pairing.
    metric_keys : list[str] or None
        Metrics to analyze. ``None`` means auto-detect all float columns.
    tol : float
        Critical-budget tolerance (default 5%).
    model_name : str
        Model name for the result objects.

    Returns
    -------
    list[DegradationResult]
        One result per metric, sorted by metric name.
    """
    if not cells_by_budget:
        return []

    budgets_sorted = sorted(cells_by_budget.keys())
    max_budget = budgets_sorted[-1]
    min_budget = budgets_sorted[0]

    # Auto-detect metric keys from the max-budget cells.
    sample_row = cells_by_budget[max_budget][0] if cells_by_budget[max_budget] else {}
    if metric_keys is None:
        skip = {"frame_budget", "latency_ms", "scene", "scene_id", "phase",
                "difficulty", "id", "model", "task"}
        metric_keys = sorted(
            k for k, v in sample_row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in skip
        )

    # Determine metric direction from specs registry (best-effort).
    try:
        from .metrics.specs import get_spec, has_spec
    except ImportError:
        get_spec = has_spec = None  # type: ignore[assignment]

    results = []
    for mkey in metric_keys:
        direction = "lower"
        if has_spec and has_spec(mkey):
            direction = get_spec(mkey).direction

        # Collect per-budget mean and per-scene values.
        means = []
        per_scene: Dict[int, List[float]] = {}
        for b in budgets_sorted:
            vals = [
                row[mkey] for row in cells_by_budget[b]
                if mkey in row and isinstance(row[mkey], (int, float))
                and np.isfinite(row[mkey])
            ]
            means.append(float(np.mean(vals)) if vals else float("nan"))
            per_scene[b] = vals

        # TCV: sign-adjusted difference (positive = temporal cues help).
        val_max = means[-1]  # metric @ max_budget
        val_min = means[0]   # metric @ min_budget
        if direction == "lower":
            # Lower is better → if max_budget is lower, temporal cues help.
            tcv = val_min - val_max  # positive when full clip is better
        else:
            # Higher is better → if max_budget is higher, temporal cues help.
            tcv = val_max - val_min

        # AUDC: trapezoidal integral, normalized by full-budget metric.
        areas = []
        for i in range(len(budgets_sorted) - 1):
            db = budgets_sorted[i + 1] - budgets_sorted[i]
            avg_val = (means[i] + means[i + 1]) / 2.0
            areas.append(db * avg_val)
        total_area = sum(areas)
        budget_range = budgets_sorted[-1] - budgets_sorted[0]
        if abs(val_max) > 1e-12 and budget_range > 0:
            audc = total_area / (budget_range * abs(val_max))
        else:
            audc = float("nan")

        # Critical budget: smallest where metric is within tol of full.
        critical = None
        for i, b in enumerate(budgets_sorted):
            if abs(val_max) < 1e-12:
                break
            rel_diff = abs(means[i] - val_max) / abs(val_max)
            if rel_diff <= tol:
                critical = b
                break

        # Wilcoxon signed-rank test (min vs max budget, scene-paired).
        wilcoxon_p = None
        if len(per_scene[min_budget]) >= 5 and len(per_scene[max_budget]) >= 5:
            # Pair by position (assumes same scene ordering).
            n = min(len(per_scene[min_budget]), len(per_scene[max_budget]))
            diffs = np.array(per_scene[max_budget][:n]) - np.array(per_scene[min_budget][:n])
            diffs = diffs[np.isfinite(diffs)]
            if len(diffs) >= 5 and np.any(diffs != 0):
                try:
                    from scipy.stats import wilcoxon  # type: ignore[import-untyped]

                    _, wilcoxon_p = wilcoxon(diffs, alternative="two-sided")
                    wilcoxon_p = float(wilcoxon_p)
                except ImportError:
                    log.debug("scipy not available — Wilcoxon test skipped")

        results.append(DegradationResult(
            model=model_name,
            metric=mkey,
            direction=direction,
            budgets=budgets_sorted,
            values=means,
            per_scene_values=per_scene,
            tcv=tcv,
            audc=audc,
            critical_budget=critical,
            wilcoxon_p=wilcoxon_p,
        ))

    return sorted(results, key=lambda r: r.metric)


# --------------------------------------------------------------------------- #
# Sweep runner (orchestrates multiple pipeline calls)
# --------------------------------------------------------------------------- #


def run_budget_sweep(
    *,
    run_fn,
    make_config_fn,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    sampling: str = "stride",
    model_name: str = "model",
    metric_keys: Optional[List[str]] = None,
    tol: float = 0.05,
) -> Tuple[Dict[int, List[dict]], List[DegradationResult]]:
    """Run a video task at multiple frame budgets and analyze degradation.

    Parameters
    ----------
    run_fn : callable
        ``run_fn(config) -> (result, dr_report, paths)`` — the task
        pipeline function (e.g. ``run_video_depth``).
    make_config_fn : callable
        ``make_config_fn(budget, sampling) -> config`` — builds a
        run config with the given ``frame_budget`` and ``sampling``.
    budgets, sampling, model_name, metric_keys, tol :
        Forwarded to :func:`degradation_analysis`.

    Returns
    -------
    cells_by_budget : dict[int, list[dict]]
        Raw cell-log dicts per budget.
    degradation : list[DegradationResult]
        One per metric.
    """
    cells_by_budget: Dict[int, List[dict]] = {}

    for budget in sorted(budgets):
        log.info("budget sweep: running at frame_budget=%d, sampling=%s", budget, sampling)
        cfg = make_config_fn(budget, sampling)
        result, _, paths = run_fn(cfg)

        # Read back the cell-log from disk.
        cells_path = paths.get("cells")
        if cells_path and Path(cells_path).exists():
            try:
                import pyarrow.parquet as pq  # type: ignore[import-untyped]

                table = pq.read_table(str(cells_path))
                cells = table.to_pylist()
            except ImportError:
                log.warning("pyarrow not available — reading cells from result.per_sample")
                cells = list(getattr(result, "per_sample", []))
        else:
            cells = list(getattr(result, "per_sample", []))

        cells_by_budget[budget] = cells
        log.info("budget sweep: budget=%d → %d cells collected", budget, len(cells))

    analysis = degradation_analysis(
        cells_by_budget,
        metric_keys=metric_keys,
        tol=tol,
        model_name=model_name,
    )

    return cells_by_budget, analysis


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def format_degradation_markdown(results: List[DegradationResult]) -> str:
    """Format degradation analysis as a Markdown summary table."""
    if not results:
        return "_No degradation results._\n"

    lines = [
        "## Frame-Budget Degradation Analysis\n",
        "| Metric | Direction | Budgets | Values | TCV | AUDC | Critical Budget | Wilcoxon p |",
        "|--------|-----------|---------|--------|-----|------|-----------------|------------|",
    ]
    for r in results:
        vals_str = " → ".join(f"{v:.4f}" for v in r.values)
        budgets_str = ", ".join(str(b) for b in r.budgets)
        tcv_str = f"{r.tcv:+.4f}" if np.isfinite(r.tcv) else "—"
        audc_str = f"{r.audc:.3f}" if np.isfinite(r.audc) else "—"
        crit_str = str(r.critical_budget) if r.critical_budget is not None else "—"
        p_str = f"{r.wilcoxon_p:.4f}" if r.wilcoxon_p is not None else "—"
        lines.append(
            f"| `{r.metric}` | {r.direction} | {budgets_str} "
            f"| {vals_str} | {tcv_str} | {audc_str} | {crit_str} | {p_str} |"
        )

    lines.append("")
    lines.append("**TCV** = Temporal Cue Value (positive = temporal cues help). "
                 "**AUDC** = Area Under Degradation Curve (normalized). "
                 "**Critical Budget** = smallest budget within 5% of full-budget.")
    lines.append("")
    return "\n".join(lines)

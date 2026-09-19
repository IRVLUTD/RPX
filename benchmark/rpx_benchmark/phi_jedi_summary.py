"""Glue between the runner's per-sample metrics and Φ / JEDI.

The runner produces a flat list of per-sample metric dicts annotated with
``scene``, ``phase``, and ``difficulty``. This module groups those rows into
a ``(N, P, K)`` tensor and calls :mod:`rpx_benchmark.phi` and
:mod:`rpx_benchmark.jedi` to produce a single :class:`PhiJediSummary`
payload that the runner attaches to the deployment-readiness report.

Design notes
------------

- **Complete-case analysis** (paper §3.2). Scenes that lack predictions for
  any phase are dropped from the MANOVA; ``n_eff`` records how many
  survived.
- **JEDI uses the same registered ``MetricSpec`` set as Φ**, but its
  direction/bounds handling is independent of the MANOVA sign-flipping.
  The convention paper §3.2 specifies is: lower-is-better metrics are
  negated *before standardisation* for Φ, separately from the
  direction-aware bounds used by JEDI.
- The runner already aggregates per-sample metrics into a ``BenchmarkResult``;
  this helper consumes the same list before aggregation so we can group
  cells without losing the (scene, phase) keying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .exceptions import MetricError
from .jedi import JEDIModelResult, aggregate_jedi, compute_jedi
from .logging_utils import get_logger
from .metrics.specs import MetricSpec, get_spec, has_spec
from .phi import (
    PhiMixedDesign,
    PhiOneway,
    PhiTransition,
    compute_phi_mixed_design,
    compute_phi_oneway,
    compute_phi_per_transition,
    phi_interpretation,
    standardise,
)

log = get_logger(__name__)


# Canonical phase order matches paper §3.1: Phase 0 = Clutter,
# Phase 1 = Interaction, Phase 2 = Clean. Alphabetical sort of the
# string labels would put Clean first, which silently mislabels the
# per-transition tests. We keep an explicit map so the runner's phase
# strings (rpx_benchmark.api.Phase values) always land on the right
# tensor axis.
CANONICAL_PHASE_ORDER: tuple[str, ...] = ("clutter", "interaction", "clean")
CANONICAL_PHASE_SHORT: dict[str, str] = {
    "clutter": "clu",
    "interaction": "int",
    "clean": "cln",
}


def _phase_sort_key(phase: str) -> tuple[int, str]:
    """Return a sort key that places canonical phases in paper order
    (Clutter → Interaction → Clean), with any unknown phase strings
    appended afterwards in alphabetical order (and clearly flagged in
    the summary's `notes`)."""
    try:
        return (CANONICAL_PHASE_ORDER.index(phase), phase)
    except ValueError:
        return (len(CANONICAL_PHASE_ORDER), phase)


@dataclass
class PhiJediSummary:
    """Bundled Φ + JEDI output for one (model, task) pair.

    Attributes
    ----------
    phi : PhiOneway | None
        One-way Φ from repeated-measures MANOVA. ``None`` when the
        data shape rules out the test (e.g. K > N).
    phi_per_transition : Dict[str, PhiTransition]
        Pairwise Hotelling-T² Φ per transition. Empty if Φ could not
        be computed.
    phi_interpretation : str | None
        Cohen anchor label for the conservative Φ.
    jedi : JEDIModelResult | None
        Aggregated JEDI score across all (scene, phase) cells.
    metrics_used : List[str]
        The metric keys that fed the analysis (in the order they appear
        on each row of the ``(N, P, K)`` tensor).
    n_eff : int
        Number of scenes that survived complete-case filtering.
    n_dropped : int
        Number of scenes dropped because they lacked a complete
        ``(P, K)`` block.
    notes : List[str]
        Free-text caveats surfaced during computation (e.g. "Pillai used
        because Wilks Λ was non-finite").
    """

    phi: Optional[PhiOneway] = None
    phi_per_transition: Dict[str, PhiTransition] = field(default_factory=dict)
    phi_interpretation: Optional[str] = None
    phi_mixed: Optional[PhiMixedDesign] = None
    """Two-way mixed-design MANOVA: per-tier Φ + interaction effect.
    ``None`` when scenes don't carry a ``difficulty`` tag."""
    jedi: Optional[JEDIModelResult] = None
    metrics_used: List[str] = field(default_factory=list)
    n_eff: int = 0
    n_dropped: int = 0
    notes: List[str] = field(default_factory=list)
    # Per-metric per-phase raw means, ``{metric: {phase_label: float}}``.
    # Populated alongside the MANOVA tensor so the paper-table filler
    # (``scripts/fill_paper_table.py``) can populate per-phase columns.
    per_phase_metric_means: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phi": self.phi.to_dict() if self.phi else None,
            "phi_per_transition": {
                k: v.to_dict() for k, v in self.phi_per_transition.items()
            },
            "phi_interpretation": self.phi_interpretation,
            "phi_mixed": self.phi_mixed.to_dict() if self.phi_mixed else None,
            "jedi": self.jedi.to_dict() if self.jedi else None,
            "metrics_used": list(self.metrics_used),
            "n_eff": self.n_eff,
            "n_dropped": self.n_dropped,
            "notes": list(self.notes),
            "per_phase_metric_means": {
                m: dict(by_phase) for m, by_phase in self.per_phase_metric_means.items()
            },
        }


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _select_metric_keys(
    per_sample_metrics: Sequence[Mapping[str, Any]],
    explicit_keys: Optional[Sequence[str]],
) -> List[str]:
    """Pick which metric keys to feed into Φ / JEDI.

    If ``explicit_keys`` is provided, validate that each has a registered
    spec and that at least one row has a numeric value for it. Otherwise,
    auto-detect: take the intersection of "numeric keys in row 0" and
    "keys with registered specs", excluding metadata (phase, scene, etc.).
    """
    metadata = {"id", "phase", "difficulty", "scene", "latency_ms"}

    if explicit_keys:
        keys = list(explicit_keys)
        for k in keys:
            if not has_spec(k):
                raise MetricError(
                    f"Metric {k!r} has no registered MetricSpec",
                    hint="Register a spec via rpx_benchmark.metrics.register_spec.",
                )
        return keys

    if not per_sample_metrics:
        return []
    row0 = per_sample_metrics[0]
    keys = []
    for k, v in row0.items():
        if k in metadata:
            continue
        if isinstance(v, bool):
            continue
        if not isinstance(v, (int, float)):
            continue
        if has_spec(k):
            keys.append(k)
    return keys


def _group_by_scene_phase(
    per_sample_metrics: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str],
) -> tuple[np.ndarray, List[str], List[Any], int, int, Dict[Any, str]]:
    """Group per-sample rows into a (N, P, K) tensor.

    Returns
    -------
    tensor : ndarray, shape (N_eff, P, K)
        Per (scene, phase) means of each metric.
    phase_order : List[str]
        The phase labels in tensor-axis order. Sorted alphabetically by
        the string value to keep results stable across runs.
    scene_ids : List
        The N_eff scene identifiers retained (complete-case).
    n_dropped : int
        Scenes dropped because they lacked at least one (scene, phase)
        observation in every phase.
    K : int
        Number of metric columns kept.
    """
    K = len(metric_keys)
    # Bucket rows by (scene, phase), keeping a list of per-row vectors so
    # we can average across multiple samples of the same (scene, phase).
    buckets: Dict[tuple, Dict[Any, List[List[float]]]] = {}
    phases_seen: set = set()
    scenes_seen: List[Any] = []
    scene_difficulty: Dict[Any, str] = {}

    for row in per_sample_metrics:
        scene = row.get("scene")
        phase = row.get("phase")
        if scene is None or phase is None:
            continue
        vec: List[float] = []
        complete = True
        for k in metric_keys:
            v = row.get(k)
            if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
                complete = False
                break
            vec.append(float(v))
        if not complete:
            continue
        phases_seen.add(phase)
        if scene not in buckets:
            buckets[scene] = {}
            scenes_seen.append(scene)
        buckets[scene].setdefault(phase, []).append(vec)
        # Difficulty is assumed constant across phases of a scene (ESD
        # design). First non-null wins.
        if scene not in scene_difficulty:
            diff = row.get("difficulty")
            if diff is not None:
                scene_difficulty[scene] = str(
                    diff.value if hasattr(diff, "value") else diff
                )

    if not buckets or not phases_seen:
        return np.zeros((0, 0, K)), [], [], 0, K, {}

    # Canonical paper order: Clutter → Interaction → Clean.
    # Unknown phase strings (if any) trail afterwards alphabetically.
    phase_order = sorted((str(p) for p in phases_seen), key=_phase_sort_key)
    P = len(phase_order)
    if P < 2:
        return np.zeros((0, P, K)), phase_order, [], 0, K, scene_difficulty

    # Drop scenes that don't have at least one observation for every phase.
    complete_scenes: List[Any] = []
    cells: List[np.ndarray] = []
    for scene in scenes_seen:
        cell = np.empty((P, K), dtype=np.float64)
        ok = True
        for i, p in enumerate(phase_order):
            rows = buckets[scene].get(p)
            if not rows:
                ok = False
                break
            cell[i] = np.mean(rows, axis=0)
        if ok:
            cells.append(cell)
            complete_scenes.append(scene)

    n_eff = len(complete_scenes)
    n_dropped = len(scenes_seen) - n_eff
    if n_eff == 0:
        return np.zeros((0, P, K)), phase_order, [], n_dropped, K, scene_difficulty

    tensor = np.stack(cells, axis=0)  # (N_eff, P, K)
    return tensor, phase_order, complete_scenes, n_dropped, K, scene_difficulty


def _orient_for_phi(
    tensor: np.ndarray,
    metric_keys: Sequence[str],
    specs: Mapping[str, MetricSpec],
) -> np.ndarray:
    """Flip lower-is-better columns so higher consistently means better.

    Paper §3.2: *"Lower-is-better metrics are negated before
    standardisation so that higher values consistently indicate better
    performance."* The negation is for Φ only; JEDI handles direction
    via its own bounds policy.
    """
    out = tensor.copy()
    for j, key in enumerate(metric_keys):
        spec = specs[key]
        if spec.direction == "lower":
            out[:, :, j] = -out[:, :, j]
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def summarize_phi_jedi(
    per_sample_metrics: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Optional[Sequence[str]] = None,
    jedi_epsilon: float = 0.0,
    apply_holm: bool = True,
) -> PhiJediSummary:
    """Compute Φ + JEDI for one (model, task) run.

    Parameters
    ----------
    per_sample_metrics : sequence of dict
        The runner's per-sample metric rows. Each row must carry
        numeric metric keys plus ``scene`` and ``phase`` metadata.
    metric_keys : sequence of str, optional
        Explicit metric keys to feed into the analysis (in canonical
        order). When ``None``, auto-detected as the numeric keys in
        row 0 that have registered :class:`MetricSpec`s.
    jedi_epsilon : float
        Lower floor on individual desirabilities (paper §3.3). Default
        ``0.0`` (strict).
    apply_holm : bool
        Pass through to :func:`compute_phi_per_transition` for the
        Holm-Bonferroni adjustment of the three pairwise tests.

    Returns
    -------
    PhiJediSummary
        Populated as fully as the data allows; ``phi`` may be ``None``
        when N_eff is too small or K is too large.
    """
    metric_keys = _select_metric_keys(per_sample_metrics, metric_keys)
    if not metric_keys:
        return PhiJediSummary(notes=["No registered metric keys found in inputs."])

    specs = {k: get_spec(k) for k in metric_keys}

    tensor, phase_order, scene_ids, n_dropped, K, scene_difficulty = _group_by_scene_phase(
        per_sample_metrics, metric_keys
    )
    n_eff = tensor.shape[0]

    summary = PhiJediSummary(
        metrics_used=list(metric_keys),
        n_eff=n_eff,
        n_dropped=n_dropped,
    )

    # Per-phase per-metric raw means: average each metric column across the
    # N_eff scenes within a phase. Drives the per-phase columns of the
    # paper's per-task tables.
    if n_eff > 0 and tensor.shape[1] >= 1:
        for j, key in enumerate(metric_keys):
            by_phase: Dict[str, float] = {}
            for i, phase_label in enumerate(phase_order):
                by_phase[str(phase_label)] = float(tensor[:, i, j].mean())
            summary.per_phase_metric_means[key] = by_phase

    if n_eff < 2 or tensor.shape[1] < 2:
        summary.notes.append(
            f"Insufficient data for Φ: N_eff={n_eff}, P={tensor.shape[1]} (need ≥2 of each)."
        )
        # JEDI can still run per-cell even if Φ can't.
    elif n_eff <= K:
        summary.notes.append(
            f"K={K} ≥ N_eff={n_eff}; MANOVA singular. Φ skipped."
        )
    else:
        z = standardise(_orient_for_phi(tensor, metric_keys, specs))
        try:
            # Passing metric_keys lets phi.py's collinearity guard name the
            # offending pair (e.g. "absrel vs rmse |r|=0.98") in warnings
            # and error hints, instead of falling back to column indices.
            phi = compute_phi_oneway(z, metric_names=list(metric_keys))
            summary.phi = phi
            summary.phi_interpretation = phi_interpretation(phi.phi_conservative)
        except MetricError as e:
            summary.notes.append(f"compute_phi_oneway failed: {e}")
        try:
            raw_transitions = compute_phi_per_transition(z, apply_holm=apply_holm)
            # Translate "0->1" / "1->2" / "0->2" into phase-name labels
            # ("clu->int" etc.) using the canonical phase_order so the
            # JSON output is unambiguous and matches paper §3.2.
            relabelled: Dict[str, PhiTransition] = {}
            for key, t in raw_transitions.items():
                try:
                    p1_idx, p2_idx = (int(x) for x in key.split("->"))
                    p1_name = phase_order[p1_idx]
                    p2_name = phase_order[p2_idx]
                    short_p1 = CANONICAL_PHASE_SHORT.get(p1_name, p1_name)
                    short_p2 = CANONICAL_PHASE_SHORT.get(p2_name, p2_name)
                    new_key = f"{short_p1}->{short_p2}"
                except (ValueError, IndexError):
                    new_key = key
                relabelled[new_key] = PhiTransition(
                    label=new_key,
                    phi=t.phi,
                    eta2=t.eta2,
                    T2=t.T2,
                    p_value=t.p_value,
                    p_holm=t.p_holm,
                    n_eff=t.n_eff,
                )
            summary.phi_per_transition = relabelled
        except MetricError as e:
            summary.notes.append(f"compute_phi_per_transition failed: {e}")
        # Flag unknown phase strings if any leaked through.
        unknown = [p for p in phase_order if p not in CANONICAL_PHASE_ORDER]
        if unknown:
            summary.notes.append(
                f"non-canonical phase labels present: {unknown}; "
                "transition labels include them with their raw names."
            )

        # Two-way mixed-design MANOVA: per-tier Φ + interaction effect.
        # Only runs when every retained scene has a `difficulty` label and
        # at least two tiers are present.
        tier_by_scene = [scene_difficulty.get(s) for s in scene_ids]
        if all(t is not None for t in tier_by_scene) and len(set(tier_by_scene)) >= 2:
            try:
                summary.phi_mixed = compute_phi_mixed_design(z, [t for t in tier_by_scene if t is not None])
            except MetricError as e:
                summary.notes.append(f"compute_phi_mixed_design failed: {e}")
        else:
            missing = sum(1 for t in tier_by_scene if t is None)
            n_tiers = len(set(t for t in tier_by_scene if t is not None))
            if missing > 0:
                summary.notes.append(
                    f"mixed-design MANOVA skipped: {missing}/{n_eff} scenes "
                    "lack a `difficulty` tier; ESD tier file probably absent."
                )
            elif n_tiers < 2:
                summary.notes.append(
                    f"mixed-design MANOVA skipped: only {n_tiers} tier(s) present."
                )

    # JEDI per cell (doesn't need standardisation; uses raw values via specs).
    if n_eff > 0 and tensor.shape[1] >= 1:
        per_cell: List[tuple[str, Any]] = []
        for i in range(n_eff):
            for j, phase_label in enumerate(phase_order):
                raw = {k: float(tensor[i, j, idx]) for idx, k in enumerate(metric_keys)}
                try:
                    r = compute_jedi(raw, epsilon=jedi_epsilon, specs=specs)
                except MetricError as e:
                    summary.notes.append(
                        f"compute_jedi failed on scene={scene_ids[i]} phase={phase_label}: {e}"
                    )
                    continue
                per_cell.append((str(phase_label), r))
        if per_cell:
            summary.jedi = aggregate_jedi(per_cell)

    return summary


__all__ = [
    "PhiJediSummary",
    "summarize_phi_jedi",
]

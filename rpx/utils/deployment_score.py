"""
rpx.utils.deployment_score
---------------------------
Computes a composite **Deployment Readiness Score (DRS)** for a model
evaluated on the RPX benchmark.

The DRS answers: *How ready is this model for real-world robotics deployment?*

It combines:
  - Task accuracy  (higher is better)
  - Computational efficiency (lower FLOPs / latency is better)
  - Robustness across deployment phases (clean → human → clutter degradation)

The score is in [0, 100]. Higher = more deployment-ready.

Design philosophy
-----------------
All sub-scores are normalised to [0, 1] using configurable reference values
(representing a "good enough for deployment" target). Researchers can override
these targets for their specific robotics platform.

During paper writing, individual sub-scores are also available for ablation.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional
import math


# ── Reference targets ──────────────────────────────────────────────────────────
# These represent "deployment-ready" thresholds for a general-purpose robot.
# Override them for your specific hardware/task requirements.

DEFAULT_TARGETS = {
    # Accuracy (task-generic, scale 0-1 or error-based)
    'accuracy': 0.75,           # target task score (e.g. mIoU, delta1)

    # Efficiency
    'flops_gflops': 50.0,       # max acceptable GFLOPs per sample
    'params_m': 100.0,          # max acceptable parameters (Millions)
    'latency_ms': 100.0,        # max acceptable latency per sample (ms)
    'model_size_mb': 500.0,     # max acceptable model size on disk (MB)

    # Robustness: max acceptable performance drop from clean → hardest phase
    'phase_drop': 0.15,         # 15% drop is the max acceptable degradation
}

# Sub-score weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    'accuracy':   0.40,  # task accuracy is most important
    'efficiency': 0.30,  # compute / memory / latency
    'robustness': 0.30,  # generalisation across deployment phases
}


@dataclass
class DeploymentScore:
    """
    Full breakdown of the Deployment Readiness Score.
    """
    # ── Sub-scores (all in [0, 1]) ────────────────────────────────────────────
    accuracy_score:   Optional[float] = None   # how good the model is at the task
    efficiency_score: Optional[float] = None   # how lightweight / fast the model is
    robustness_score: Optional[float] = None   # how stable across deployment phases

    # ── Composite ─────────────────────────────────────────────────────────────
    deployment_readiness_score: Optional[float] = None   # weighted sum, [0, 100]

    # ── Breakdown of efficiency sub-components ────────────────────────────────
    flops_score:    Optional[float] = None
    params_score:   Optional[float] = None
    latency_score:  Optional[float] = None
    size_score:     Optional[float] = None

    # ── Phase-specific accuracy breakdown ─────────────────────────────────────
    phase_scores: Dict[str, float] = field(default_factory=dict)

    # ── Metadata ──────────────────────────────────────────────────────────────
    weights: Dict[str, float] = field(default_factory=dict)
    targets: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def summary(self) -> str:
        lines = ["── Deployment Readiness Score ────────────────────"]
        if self.accuracy_score is not None:
            lines.append(f"  Accuracy Score          {self.accuracy_score*100:.1f} / 100")
        if self.efficiency_score is not None:
            lines.append(f"  Efficiency Score        {self.efficiency_score*100:.1f} / 100")
            if self.flops_score is not None:
                lines.append(f"    ↳ FLOPs               {self.flops_score*100:.1f}")
            if self.params_score is not None:
                lines.append(f"    ↳ Params              {self.params_score*100:.1f}")
            if self.latency_score is not None:
                lines.append(f"    ↳ Latency             {self.latency_score*100:.1f}")
            if self.size_score is not None:
                lines.append(f"    ↳ Model Size          {self.size_score*100:.1f}")
        if self.robustness_score is not None:
            lines.append(f"  Robustness Score        {self.robustness_score*100:.1f} / 100")
        if self.phase_scores:
            for phase, score in self.phase_scores.items():
                lines.append(f"    ↳ {phase:<20} {score*100:.1f}")
        lines.append("─" * 50)
        if self.deployment_readiness_score is not None:
            lines.append(f"  ★ DEPLOYMENT READINESS  {self.deployment_readiness_score:.1f} / 100")
        lines.append("─" * 50)
        return "\n".join(lines)


def _bounded_score(value, target, lower_is_better=True) -> float:
    """
    Converts a raw metric value to a [0, 1] score.
    For higher-is-better: score = min(value / target, 1.0)
    For lower-is-better:  score = max(1 - value / target, 0.0)
    """
    if value is None:
        return 1.0  # assume fine if unknown
    ratio = value / target
    if lower_is_better:
        return max(0.0, 1.0 - ratio)
    else:
        return min(1.0, ratio)


def compute_deployment_score(
    task_results: dict,
    complexity_profile=None,
    targets: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> DeploymentScore:
    """
    Compute the Deployment Readiness Score.

    Args:
        task_results:
            Output of `benchmark.evaluate(model)`. Expected to have:
              - 'overall': dict with a primary metric key (e.g. 'mean_iou', 'mean_trans_error')
              - 'clean', 'human', 'clutter': per-phase dicts with the same key
        complexity_profile:
            A `ModelComplexityProfile` from `rpx.utils.model_profiler.profile_model()`.
        targets:
            Override DEFAULT_TARGETS for specific hardware/task constraints.
        weights:
            Override DEFAULT_WEIGHTS for different scoring priorities.

    Returns:
        DeploymentScore
    """
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    ds = DeploymentScore(weights=weights, targets=targets)

    # ── 1. Accuracy sub-score ────────────────────────────────────────────────
    # Extract the primary metric from overall results.
    # We try common keys in priority order.
    _ACCURACY_KEYS_HIGH = ['mean_iou', 'delta1', 'delta2', 'delta3',
                           'mean_psnr']          # higher is better
    _ACCURACY_KEYS_LOW  = ['mean_trans_error', 'mean_rot_error',
                           'abs_rel', 'rmse',
                           'mean_count_error']    # lower is better

    overall = task_results.get('overall', task_results)
    accuracy_val = None
    accuracy_higher_better = True

    for k in _ACCURACY_KEYS_HIGH:
        if k in overall:
            accuracy_val = overall[k]
            accuracy_higher_better = True
            break
    if accuracy_val is None:
        for k in _ACCURACY_KEYS_LOW:
            if k in overall:
                accuracy_val = overall[k]
                accuracy_higher_better = False
                break

    if accuracy_val is not None:
        ds.accuracy_score = _bounded_score(
            accuracy_val, targets['accuracy'],
            lower_is_better=not accuracy_higher_better
        )

    # ── 2. Phase-specific scores (robustness) ────────────────────────────────
    phase_scores = {}
    for phase in ('clean', 'human', 'clutter'):
        phase_res = task_results.get(phase)
        if phase_res is None:
            continue
        for k in _ACCURACY_KEYS_HIGH + _ACCURACY_KEYS_LOW:
            if k in phase_res:
                high = k in _ACCURACY_KEYS_HIGH
                phase_scores[phase] = _bounded_score(
                    phase_res[k], targets['accuracy'],
                    lower_is_better=not high
                )
                break

    ds.phase_scores = phase_scores

    # Robustness score: penalise large drop from clean → worst phase
    if 'clean' in phase_scores and phase_scores:
        worst = min(phase_scores.values())
        best = phase_scores.get('clean', max(phase_scores.values()))
        drop = max(0.0, best - worst)
        ds.robustness_score = _bounded_score(
            drop, targets['phase_drop'], lower_is_better=True
        )
        # If no clean phase, use mean across phases
        if 'clean' not in phase_scores:
            ds.robustness_score = float(sum(phase_scores.values()) / len(phase_scores))
    elif phase_scores:
        ds.robustness_score = float(sum(phase_scores.values()) / len(phase_scores))

    # ── 3. Efficiency sub-score ──────────────────────────────────────────────
    eff_scores = []
    if complexity_profile is not None:
        if complexity_profile.flops_fvcore is not None:
            ds.flops_score = _bounded_score(
                complexity_profile.flops_fvcore, targets['flops_gflops'])
            eff_scores.append(ds.flops_score)

        if complexity_profile.total_params is not None:
            ds.params_score = _bounded_score(
                complexity_profile.total_params / 1e6, targets['params_m'])
            eff_scores.append(ds.params_score)

        if complexity_profile.latency_ms_mean is not None:
            ds.latency_score = _bounded_score(
                complexity_profile.latency_ms_mean, targets['latency_ms'])
            eff_scores.append(ds.latency_score)

        if complexity_profile.model_size_mb is not None:
            ds.size_score = _bounded_score(
                complexity_profile.model_size_mb, targets['model_size_mb'])
            eff_scores.append(ds.size_score)

    ds.efficiency_score = float(sum(eff_scores) / len(eff_scores)) if eff_scores else None

    # ── 4. Weighted composite ─────────────────────────────────────────────────
    sub_scores = {
        'accuracy':   ds.accuracy_score,
        'efficiency': ds.efficiency_score,
        'robustness': ds.robustness_score,
    }
    available = {k: v for k, v in sub_scores.items() if v is not None}
    if available:
        total_weight = sum(weights[k] for k in available)
        drs = sum(v * weights[k] / total_weight for k, v in available.items())
        ds.deployment_readiness_score = round(drs * 100, 2)

    return ds

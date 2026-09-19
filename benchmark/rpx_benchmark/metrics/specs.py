"""Per-metric specifications used by JEDI and Φ.

A :class:`MetricSpec` carries the *direction* (higher-is-better vs lower-is-
better), the per-metric *bounds* ``(best, worst)``, and whether those bounds
are theoretical (e.g., AP ∈ [0,1]) or empirically frozen across a fixed model
zoo. Specs are registered alongside the metric calculators that produce the
numeric values; downstream code (JEDI in :mod:`rpx_benchmark.jedi`, Φ in
:mod:`rpx_benchmark.phi`) consults the spec registry rather than hard-coding
direction or bounds.

The design follows paper §3.3: direction-aware bounds avoid the "negate
lower-is-better metrics" foot-gun, and the frozen-observed bounds policy
makes JEDI reproducible across re-evaluations.

Usage
-----

Calculators expose their specs via a ``specs()`` classmethod; the metric
runtime imports the module, the ``@register_metric`` decorator runs, and
specs are auto-registered. Direct lookup::

    from rpx_benchmark.metrics import get_spec
    spec = get_spec("absrel")
    assert spec.direction == "lower"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

from ..exceptions import MetricError
from ..logging_utils import get_logger

log = get_logger(__name__)


Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricSpec:
    """Static metadata about a single scalar metric.

    Attributes
    ----------
    name : str
        Key under which the metric is emitted by the calculator
        (e.g. ``"absrel"``, ``"ap50"``). Must match the dict key that
        :meth:`MetricCalculator.compute` returns.
    direction : {"higher", "lower"}
        Whether higher raw values indicate better performance.
    best : float
        The optimal end of the metric scale. For higher-is-better
        metrics this is the maximum; for lower-is-better, the minimum.
    worst : float
        The non-optimal end. For higher-is-better, the minimum; for
        lower-is-better, the maximum.
    theoretical : bool
        True iff (best, worst) are derived from the metric's
        mathematical definition (e.g. AP ∈ [0, 1]). False if they are
        empirically frozen over a model zoo. The distinction is
        documented per paper §3.3 — theoretical bounds are stable
        forever; empirical bounds are part of a versioned release.
    description : str
        One-line human-readable summary; printed in the JEDI/Φ output
        cards for debugging.
    """

    name: str
    direction: Direction
    best: float
    worst: float
    theoretical: bool
    description: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("higher", "lower"):
            raise MetricError(
                f"MetricSpec direction must be 'higher' or 'lower', got {self.direction!r}",
                hint="See rpx_benchmark.metrics.specs.Direction.",
            )
        if self.direction == "higher" and self.best <= self.worst:
            raise MetricError(
                f"MetricSpec({self.name!r}): direction=higher requires best > worst, "
                f"got best={self.best}, worst={self.worst}",
            )
        if self.direction == "lower" and self.best >= self.worst:
            raise MetricError(
                f"MetricSpec({self.name!r}): direction=lower requires best < worst, "
                f"got best={self.best}, worst={self.worst}",
            )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_SPECS: Dict[str, MetricSpec] = {}


def register_spec(spec: MetricSpec) -> MetricSpec:
    """Register a :class:`MetricSpec` under its ``name``.

    Idempotent: re-registering the same name with an *identical* spec is a
    silent no-op; re-registering with a *different* spec raises
    :class:`MetricError` (to catch accidental overwrites during module
    reloads).
    """
    existing = _SPECS.get(spec.name)
    if existing is not None and existing != spec:
        raise MetricError(
            f"MetricSpec for {spec.name!r} already registered with different values; "
            f"existing={existing}, new={spec}",
            hint="Either match the existing spec or unregister it first.",
        )
    _SPECS[spec.name] = spec
    return spec


def register_specs(specs: Iterable[MetricSpec]) -> None:
    """Convenience wrapper to register a batch."""
    for s in specs:
        register_spec(s)


def get_spec(name: str) -> MetricSpec:
    """Look up the spec for a metric ``name``; raise if unknown."""
    try:
        return _SPECS[name]
    except KeyError as e:
        raise MetricError(
            f"No MetricSpec registered for {name!r}",
            hint=(
                "Import the metric module that owns this name "
                "(e.g. `import rpx_benchmark.metrics.depth`) so its specs "
                "auto-register, or call register_spec(...) explicitly."
            ),
        ) from e


def has_spec(name: str) -> bool:
    """True iff a spec is registered under ``name``."""
    return name in _SPECS


def registered_specs() -> Dict[str, MetricSpec]:
    """Snapshot of the current spec registry (defensive copy)."""
    return dict(_SPECS)


def clear_registry() -> None:
    """Remove every registered spec. Primarily for tests."""
    _SPECS.clear()


# --------------------------------------------------------------------------- #
# Built-in specs for every shipped task
# --------------------------------------------------------------------------- #
#
# Bounds policy (paper §3.3):
#   - theoretical=True  → derived from the metric's mathematical definition
#                          (e.g. AP_50 ∈ [0, 1]).
#   - theoretical=False → frozen observed bounds across the shipped model
#                          zoo. The numbers below are the v1 freeze; rerunning
#                          on a new model set requires an explicit re-release
#                          (see §3.3 paragraph on out-of-bounds clipping).
#
# When a new model produces a score outside (worst, best), the individual
# desirability d_k clips to [0, 1] and the toolkit flags the cell
# `out-of-bounds`. Bounds are never silently re-fit.

# Depth (Image Depth frame-level & Video Depth video) — robotics-first set.
#
# Organised into three blocks (see experiments/depth_benchmark/METRICS_FINAL.md):
#   A. Standard accuracy   — citable, general-perception comparability.
#   B. Robotics deployment — near-range, grasp-tolerance 3D, surface geometry.
#   C. Temporal (video)    — per-clip stability; lower-is-better except TCC.
#
# Range-stratified variants (e.g. "absrel_near"/"_mid"/"_far") are NOT
# registered individually: the summary layer resolves a suffixed name to its
# base spec by stripping the trailing _near/_mid/_far token.
#
# Intentionally EXCLUDED from the final set (feasibility, see METRICS_FINAL.md):
#   - relnormal : 1M-Sobol curvature consistency — highest cost-to-signal.
#   - irmse     : inverse-depth RMSE — low added info over AbsRel/SILog on D435.
# delta2/delta3 stay registered (the calculator still emits them) but are
# reported as secondary; delta1 is the primary threshold metric.
#
# Bounds tagged "provisional v0" are placeholders to be FROZEN after the v1
# model-zoo run (cell_log → frozen empirical bounds). Theoretical bounds
# (fractions ∈ [0,1]) are stable forever.
DEPTH_SPECS = (
    # A. Standard accuracy — metric mode; relative models evaluated post-alignment
    MetricSpec("absrel",    "lower",  best=0.0,  worst=0.5,  theoretical=False,
               description="Absolute relative depth error |d̂−d|/d"),
    MetricSpec("rmse",      "lower",  best=0.0,  worst=2.0,  theoretical=False,
               description="Root-mean-square depth error (metres)"),
    MetricSpec("rmselog",   "lower",  best=0.0,  worst=0.5,  theoretical=False,
               description="RMSE in log-depth space (provisional v0)"),
    MetricSpec("silog",     "lower",  best=0.0,  worst=25.0, theoretical=False,
               description="Scale-invariant log error Var(log d̂−log d) (provisional v0)"),
    MetricSpec("delta1",    "higher", best=1.0,  worst=0.0,  theoretical=True,
               description="Fraction of pixels with max(d̂/d, d/d̂) < 1.25 (primary)"),
    MetricSpec("delta2",    "higher", best=1.0,  worst=0.0,  theoretical=True,
               description="Threshold accuracy < 1.25² (secondary)"),
    MetricSpec("delta3",    "higher", best=1.0,  worst=0.0,  theoretical=True,
               description="Threshold accuracy < 1.25³ (secondary)"),
    # B. Robotics deployment — near-range, grasp-tolerance 3D, surface geometry
    MetricSpec("chamfer_l1",    "lower",  best=0.0,  worst=0.5,  theoretical=False,
               description="Camera-frame Chamfer-L1 of back-projected cloud, metres — pose-free (provisional v0)"),
    MetricSpec("fscore_5cm",    "higher", best=1.0,  worst=0.0,  theoretical=True,
               description="Grasp-reachability F-score @ 5cm, camera-frame. Also report @1cm/@2cm."),
    MetricSpec("normal_mae",    "lower",  best=0.0,  worst=45.0, theoretical=False,
               description="Mean angular error of depth-derived normals, degrees — grasp orientation (provisional v0)"),
    MetricSpec("normal_acc1125","higher", best=1.0,  worst=0.0,  theoretical=True,
               description="Fraction of surface normals within 11.25° of GT"),
    MetricSpec("boundary_f1",   "higher", best=1.0,  worst=0.0,  theoretical=True,
               description="Scale-invariant occluding-contour F1 from depth ratios — relative-only on holey GT"),
    # C. Temporal (Video Depth only; per phase clip)
    MetricSpec("opw",  "lower",  best=0.0, worst=0.1, theoretical=False,
               description="Optical-flow warping consistency, RAFT flow. Not in RPX K-vector (external flow model dependency; RGB-D-only policy). Diagnostic only (provisional v0)"),
    MetricSpec("tae",  "lower",  best=0.0, worst=0.1, theoretical=False,
               description="Temporal alignment error via SE(3) reprojection. Not in RPX K-vector (needs T265 poses; RGB-D-only policy). Diagnostic only (provisional v0)"),
    MetricSpec("tgm",  "lower",  best=0.0, worst=0.1, theoretical=False,
               description="Temporal gradient matching vs GT depth (provisional v0)"),
    MetricSpec("tcc",  "higher", best=1.0, worst=0.0, theoretical=True,
               description="Temporal consistency (SSIM on depth-change maps)"),
    MetricSpec("tgse", "lower",  best=0.0, worst=0.1, theoretical=False,
               description="Temporal gradient squared error (L2, sign-preserving; VDPP, arXiv:2604.06665)"),
    MetricSpec("tmc",  "higher", best=1.0, worst=0.0, theoretical=True,
               description="Temporal motion consistency (SSIM on depth-flow maps; Zhang ICCV 2019)"),
    # C.1 Range-stratified temporal (per depth bin)
    MetricSpec("tae_near", "lower", best=0.0, worst=0.1, theoretical=False,
               description="TAE at near range (0.3-1.0m, tabletop manipulation)"),
    MetricSpec("tae_mid",  "lower", best=0.0, worst=0.1, theoretical=False,
               description="TAE at mid range (1.0-2.5m, arm's reach)"),
    MetricSpec("tae_far",  "lower", best=0.0, worst=0.1, theoretical=False,
               description="TAE at far range (2.5-5.0m, room-scale)"),
    # C.2 Range-stratified spatial accuracy (per depth bin, clip-averaged)
    MetricSpec("absrel_near", "lower",  best=0.0, worst=1.0, theoretical=False,
               description="AbsRel at near range (0.3-1.0m)"),
    MetricSpec("absrel_mid",  "lower",  best=0.0, worst=1.0, theoretical=False,
               description="AbsRel at mid range (1.0-2.5m)"),
    MetricSpec("absrel_far",  "lower",  best=0.0, worst=1.0, theoretical=False,
               description="AbsRel at far range (2.5-5.0m)"),
    MetricSpec("rmse_near", "lower",  best=0.0, worst=1.0, theoretical=False,
               description="RMSE at near range (0.3-1.0m)"),
    MetricSpec("rmse_mid",  "lower",  best=0.0, worst=1.0, theoretical=False,
               description="RMSE at mid range (1.0-2.5m)"),
    MetricSpec("rmse_far",  "lower",  best=0.0, worst=1.0, theoretical=False,
               description="RMSE at far range (2.5-5.0m)"),
    MetricSpec("delta1_near", "higher", best=1.0, worst=0.0, theoretical=True,
               description="delta1 at near range (0.3-1.0m)"),
    MetricSpec("delta1_mid",  "higher", best=1.0, worst=0.0, theoretical=True,
               description="delta1 at mid range (1.0-2.5m)"),
    MetricSpec("delta1_far",  "higher", best=1.0, worst=0.0, theoretical=True,
               description="delta1 at far range (2.5-5.0m)"),
    # C.3 Range-stratified temporal (per depth bin, pose-free / flow-free)
    MetricSpec("tgm_near", "lower", best=0.0, worst=0.1, theoretical=False,
               description="TGM at near range (0.3-1.0m, tabletop manipulation)"),
    MetricSpec("tgm_mid",  "lower", best=0.0, worst=0.1, theoretical=False,
               description="TGM at mid range (1.0-2.5m, arm's reach)"),
    MetricSpec("tgm_far",  "lower", best=0.0, worst=0.1, theoretical=False,
               description="TGM at far range (2.5-5.0m, room-scale)"),
    MetricSpec("tgse_near", "lower", best=0.0, worst=0.1, theoretical=False,
               description="TGSE at near range (0.3-1.0m, tabletop manipulation)"),
    MetricSpec("tgse_mid",  "lower", best=0.0, worst=0.1, theoretical=False,
               description="TGSE at mid range (1.0-2.5m, arm's reach)"),
    MetricSpec("tgse_far",  "lower", best=0.0, worst=0.1, theoretical=False,
               description="TGSE at far range (2.5-5.0m, room-scale)"),
)

# Detection / Grounding (D2) — K=4
DETECTION_SPECS = (
    MetricSpec("ap50",    "higher", best=1.0, worst=0.0, theoretical=True,
               description="AP at IoU≥0.50"),
    MetricSpec("ap75",    "higher", best=1.0, worst=0.0, theoretical=True,
               description="AP at IoU≥0.75"),
    MetricSpec("map",     "higher", best=1.0, worst=0.0, theoretical=True,
               description="mean AP averaged over IoU thresholds"),
    MetricSpec("ar",      "higher", best=1.0, worst=0.0, theoretical=True,
               description="Average recall"),
)

# Tracking (D3) — K=4
TRACKING_SPECS = (
    MetricSpec("mota",    "higher", best=1.0, worst=0.0, theoretical=True,
               description="Multi-Object Tracking Accuracy"),
    MetricSpec("idf1",    "higher", best=1.0, worst=0.0, theoretical=True,
               description="ID F1 (identity matching F-score)"),
    MetricSpec("hota",    "higher", best=1.0, worst=0.0, theoretical=True,
               description="Higher-Order Tracking Accuracy"),
    MetricSpec("id_sw",   "lower",  best=0.0, worst=100.0, theoretical=False,
               description="Identity switches per scene (count)"),
)

# Scene QA (D4) and Spatial QA (D5) — K=2 each
QA_SPECS = (
    MetricSpec("fuzzy_acc",   "higher", best=1.0, worst=0.0, theoretical=True,
               description="Normalised-similarity fuzzy match accuracy"),
    MetricSpec("exact_match", "higher", best=1.0, worst=0.0, theoretical=True,
               description="Strict string-equality accuracy"),
)

# Relative Camera Pose (D6) — K=4
POSE_SPECS = (
    MetricSpec("auc_5deg",   "higher", best=1.0, worst=0.0, theoretical=True,
               description="AUC of rotation error CDF up to 5°"),
    MetricSpec("auc_10deg",  "higher", best=1.0, worst=0.0, theoretical=True,
               description="AUC of rotation error CDF up to 10°"),
    MetricSpec("auc_20deg",  "higher", best=1.0, worst=0.0, theoretical=True,
               description="AUC of rotation error CDF up to 20°"),
    MetricSpec("metric_auc", "higher", best=1.0, worst=0.0, theoretical=True,
               description="Joint rotation × metric-translation AUC"),
)



def _register_builtins() -> None:
    """Populate the registry with every shipped task's specs at import time."""
    for batch in (
        DEPTH_SPECS,
        DETECTION_SPECS,
        TRACKING_SPECS,
        QA_SPECS,
        POSE_SPECS,
    ):
        register_specs(batch)


_register_builtins()


__all__ = [
    "Direction",
    "MetricSpec",
    "register_spec",
    "register_specs",
    "get_spec",
    "has_spec",
    "registered_specs",
    "clear_registry",
    "DEPTH_SPECS",
    "DETECTION_SPECS",
    "TRACKING_SPECS",
    "QA_SPECS",
    "POSE_SPECS",
]

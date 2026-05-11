"""Deployment-readiness metrics for RPX: TS, STR, SGC, ESD, weighted scoring.

These are the core novel contributions of the RPX benchmark paper.

Four metrics characterise how well a model generalises under deployment conditions:

TS   — Temporal Stability:         pose-compensated frame-to-frame consistency
STR  — State-Transition Robustness: performance drop / recovery across scene phases
SGC  — Stack-Level Geometric Coherence: mask–depth boundary alignment
ESD  — Effort-Stratified Difficulty: per-difficulty breakdown (Easy/Medium/Hard)

Plus the weighted phase scoring scheme:
  S_p       = 0.25·M(p,Easy) + 0.35·M(p,Med) + 0.40·M(p,Hard)
  S_overall = (S_clutter + S_interaction + S_clean) / 3
  Δ_int     = S_interaction − S_clutter    (interaction drop)
  Δ_rec     = S_clean       − S_interaction (recovery)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .api import ESD_WEIGHTS, Difficulty, Phase

# ------------------------------------------------------------------ #
# Result dataclasses
# ------------------------------------------------------------------ #


@dataclass
class TemporalStabilityResult:
    """TS score per task type.

    TS_seg   = E[IoU(P_t, warp(P_{t+1}, ΔT))]  (segmentation)
    TS_depth = E[||D_t − warp(D_{t+1}, ΔT)||_1 / N_valid]  (depth)

    Warping uses the relative SE(3) pose from the T265 ground-truth track.
    When exact warping is not feasible, a consistency proxy (unchanged-pixel
    fraction) is used as a lower bound.
    """

    ts_score: float  # primary TS value (higher = more stable)
    num_pairs: int  # number of consecutive frame pairs evaluated
    per_pair: List[float] = field(default_factory=list)


@dataclass
class StateTransitionRobustnessResult:
    """STR captures performance change across phase boundaries.

    STR_{C→I} = M(interaction) − M(clutter)    ← interaction drop (negative = worse)
    STR_{I→L} = M(clean)       − M(interaction) ← recovery        (positive = better)
    """

    str_c_to_i: float  # interaction drop Δ = M_I − M_C
    str_i_to_l: float  # recovery          Δ = M_L − M_I
    metric_clutter: float
    metric_interaction: float
    metric_clean: float


@dataclass
class StackGeometricCoherenceResult:
    """SGC measures mask–depth boundary alignment.

    SGC = F-score(boundary(mask), boundary(depth_gradient > τ))
    Boundary pixels are extracted via Sobel gradient magnitude thresholding.
    """

    sgc_score: float  # F-score of mask/depth boundary overlap
    precision: float
    recall: float
    num_samples: int


@dataclass
class ESDResult:
    """Per-difficulty metric breakdown (Effort-Stratified Difficulty)."""

    easy: float | None
    medium: float | None
    hard: float | None
    metric_key: str  # which metric was stratified (e.g. "absrel", "miou")

    def weighted_score(self) -> float:
        """S_p = 0.25·Easy + 0.35·Medium + 0.40·Hard."""
        total, weight = 0.0, 0.0
        for diff, w in ESD_WEIGHTS.items():
            val = getattr(self, diff.value)
            if val is not None:
                total += w * val
                weight += w
        return total / weight if weight > 0 else 0.0


@dataclass
class WeightedPhaseScore:
    """Full deployment-readiness scoring table.

    Per phase:  S_p = 0.25·M(p,Easy) + 0.35·M(p,Medium) + 0.40·M(p,Hard)
    Overall:    S_overall = (S_C + S_I + S_L) / 3
    Delta int:  Δ_int = S_I − S_C
    Delta rec:  Δ_rec = S_L − S_I
    """

    clutter: ESDResult
    interaction: ESDResult
    clean: ESDResult

    @property
    def s_clutter(self) -> float:
        return self.clutter.weighted_score()

    @property
    def s_interaction(self) -> float:
        return self.interaction.weighted_score()

    @property
    def s_clean(self) -> float:
        return self.clean.weighted_score()

    @property
    def s_overall(self) -> float:
        return (self.s_clutter + self.s_interaction + self.s_clean) / 3.0

    @property
    def delta_int(self) -> float:
        """Interaction drop: S_I − S_C (negative = model degrades on interaction)."""
        return self.s_interaction - self.s_clutter

    @property
    def delta_rec(self) -> float:
        """Recovery: S_L − S_I (positive = model recovers after interaction)."""
        return self.s_clean - self.s_interaction

    def to_dict(self) -> Dict[str, float]:
        return {
            "s_clutter": self.s_clutter,
            "s_interaction": self.s_interaction,
            "s_clean": self.s_clean,
            "s_overall": self.s_overall,
            "delta_int": self.delta_int,
            "delta_rec": self.delta_rec,
        }


@dataclass
class DeploymentReadinessReport:
    """Aggregated deployment-readiness report for a model on a task."""

    task: str
    model_name: str
    weighted_phase_score: WeightedPhaseScore | None = None
    temporal_stability: TemporalStabilityResult | None = None
    state_transition: StateTransitionRobustnessResult | None = None
    geometric_coherence: StackGeometricCoherenceResult | None = None

    # --- Tier 1: hardware-agnostic model properties -----------------------
    params_m: float | None = None  # parameter count in millions
    flops_g: float | None = None  # FLOPs in giga-ops (batch=1)
    macs_g: float | None = None  # MACs (= FLOPs / 2)
    actmem_gb_fp16: float | None = None  # optional activation memory at FP16
    memory_traffic_gb: float | None = None  # estimated DRAM traffic (GB)
    arithmetic_intensity: float | None = None  # FLOPs / Bytes (FLOP/Byte)

    # --- Tier 2: roofline bounds ------------------------------------------
    roofline: Dict | None = None  # {gpu_name: RooflineBound.to_dict()}

    # --- Tier 3: measured (hardware-specific) ------------------------------
    latency_ms_per_sample: float | None = None  # wall-clock inference latency (p50)
    peak_memory_mb: float | None = None  # peak resident memory (device-agnostic)
    system_card: Dict | None = None  # SystemCard.to_dict()

    # --- Operating point (precision × accuracy × cost triple) -------------
    operating_point: "OperatingPoint | None" = None

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"task": self.task, "model": self.model_name}
        if self.weighted_phase_score:
            out.update(self.weighted_phase_score.to_dict())
        if self.temporal_stability:
            out["ts_score"] = self.temporal_stability.ts_score
        if self.state_transition:
            out["str_c_to_i"] = self.state_transition.str_c_to_i
            out["str_i_to_l"] = self.state_transition.str_i_to_l
        if self.geometric_coherence:
            out["sgc_score"] = self.geometric_coherence.sgc_score
        # Tier 1
        if self.params_m is not None:
            out["params_m"] = self.params_m
        if self.flops_g is not None:
            out["flops_g"] = self.flops_g
        if self.macs_g is not None:
            out["macs_g"] = self.macs_g
        if self.memory_traffic_gb is not None:
            out["memory_traffic_gb"] = self.memory_traffic_gb
        if self.arithmetic_intensity is not None:
            out["arithmetic_intensity"] = self.arithmetic_intensity
        # Tier 2
        if self.roofline is not None:
            out["roofline"] = self.roofline
        # Tier 3
        if self.latency_ms_per_sample is not None:
            out["latency_ms_per_sample"] = self.latency_ms_per_sample
        if self.peak_memory_mb is not None:
            out["peak_memory_mb"] = self.peak_memory_mb
        if self.system_card is not None:
            out["system_card"] = self.system_card
        return out


# ------------------------------------------------------------------ #
# Metric computation functions
# ------------------------------------------------------------------ #


def compute_esd(
    per_sample_metrics: List[Dict[str, float]],
    per_sample_difficulties: List[Difficulty | None],
    metric_key: str,
) -> ESDResult:
    """Compute per-difficulty metric averages from per-sample results.

    Args:
        per_sample_metrics: list of metric dicts, one per sample.
        per_sample_difficulties: difficulty label per sample (may be None).
        metric_key: which metric key to stratify (e.g. "absrel", "miou").

    Returns:
        ESDResult with easy/medium/hard averages.
    """
    buckets: Dict[Difficulty, List[float]] = {d: [] for d in Difficulty}

    for metrics, diff in zip(per_sample_metrics, per_sample_difficulties, strict=False):
        if diff is None or metric_key not in metrics:
            continue
        buckets[diff].append(metrics[metric_key])

    def mean_or_none(vals: List[float]) -> float | None:
        return float(np.mean(vals)) if vals else None

    return ESDResult(
        easy=mean_or_none(buckets[Difficulty.EASY]),
        medium=mean_or_none(buckets[Difficulty.MEDIUM]),
        hard=mean_or_none(buckets[Difficulty.HARD]),
        metric_key=metric_key,
    )


def compute_weighted_phase_score(
    per_sample_metrics: List[Dict[str, float]],
    per_sample_phases: List[Phase | None],
    per_sample_difficulties: List[Difficulty | None],
    metric_key: str,
) -> WeightedPhaseScore:
    """Compute the full weighted phase scoring table.

    Groups samples by (phase, difficulty) and computes:
        S_p = 0.25·M(p,Easy) + 0.35·M(p,Medium) + 0.40·M(p,Hard)
    for each phase, then overall score and transition deltas.
    """
    phase_sample_metrics: Dict[Phase, Tuple[List, List]] = {p: ([], []) for p in Phase}

    for m, ph, diff in zip(
        per_sample_metrics, per_sample_phases, per_sample_difficulties, strict=False
    ):
        if ph is None:
            continue
        phase_sample_metrics[ph][0].append(m)
        phase_sample_metrics[ph][1].append(diff)

    def esd_for_phase(ph: Phase) -> ESDResult:
        metrics_list, diff_list = phase_sample_metrics[ph]
        return compute_esd(metrics_list, diff_list, metric_key)

    return WeightedPhaseScore(
        clutter=esd_for_phase(Phase.CLUTTER),
        interaction=esd_for_phase(Phase.INTERACTION),
        clean=esd_for_phase(Phase.CLEAN),
    )


def compute_str(
    phase_scores: Dict[Phase, float],
) -> StateTransitionRobustnessResult:
    """Compute STR from per-phase aggregated scores.

    Args:
        phase_scores: dict mapping Phase → scalar metric value.
    """
    m_c = phase_scores.get(Phase.CLUTTER, 0.0)
    m_i = phase_scores.get(Phase.INTERACTION, 0.0)
    m_l = phase_scores.get(Phase.CLEAN, 0.0)
    return StateTransitionRobustnessResult(
        str_c_to_i=m_i - m_c,
        str_i_to_l=m_l - m_i,
        metric_clutter=m_c,
        metric_interaction=m_i,
        metric_clean=m_l,
    )


def compute_temporal_stability_seg(
    pred_masks: Sequence[np.ndarray],
    camera_poses: Sequence[np.ndarray | None],
) -> TemporalStabilityResult:
    """TS_seg = E[IoU(P_t, warp(P_{t+1}, ΔT))].

    When T265 pose data is available, we use the relative rotation to compensate
    for camera motion before computing IoU between adjacent frames.  Without
    pixel-accurate warping (which requires depth for backprojection), we apply
    a simplified affine proxy using the in-plane rotation component only.

    This gives a conservative lower-bound TS_seg that is still a meaningful
    stability signal when scenes have modest depth variation.

    Args:
        pred_masks: sequence of predicted segmentation masks (H×W int).
        camera_poses: per-frame 4×4 SE(3) matrices (camera-to-world), or None.

    Returns:
        TemporalStabilityResult.
    """
    if len(pred_masks) < 2:
        return TemporalStabilityResult(ts_score=1.0, num_pairs=0)

    per_pair = []
    for t in range(len(pred_masks) - 1):
        m_t = np.asarray(pred_masks[t], dtype=np.int32)
        m_t1 = np.asarray(pred_masks[t + 1], dtype=np.int32)

        # Attempt pose-compensated warp if poses are available
        if camera_poses[t] is not None and camera_poses[t + 1] is not None:
            m_t1_warped = _warp_mask_approx(m_t1, camera_poses[t], camera_poses[t + 1])
        else:
            m_t1_warped = m_t1

        # Per-class IoU then mean
        classes = np.unique(np.concatenate([m_t.flatten(), m_t1_warped.flatten()]))
        classes = classes[classes >= 0]
        if len(classes) == 0:
            per_pair.append(1.0)
            continue
        ious = []
        for c in classes:
            inter = float(((m_t == c) & (m_t1_warped == c)).sum())
            union = float(((m_t == c) | (m_t1_warped == c)).sum())
            ious.append(inter / union if union > 0 else 1.0)
        per_pair.append(float(np.mean(ious)))

    return TemporalStabilityResult(
        ts_score=float(np.mean(per_pair)),
        num_pairs=len(per_pair),
        per_pair=per_pair,
    )


def compute_temporal_stability_depth(
    pred_depths: Sequence[np.ndarray],
    camera_poses: Sequence[np.ndarray | None],
) -> TemporalStabilityResult:
    """TS_depth = E[||D_t − warp(D_{t+1}, ΔT)||_1 / N_valid].

    Normalised to [0,1] by dividing by the max depth range to give a
    higher-is-better stability score (TS = 1 − normalised_L1).
    """
    if len(pred_depths) < 2:
        return TemporalStabilityResult(ts_score=1.0, num_pairs=0)

    per_pair = []
    for t in range(len(pred_depths) - 1):
        d_t = np.asarray(pred_depths[t], dtype=np.float32)
        d_t1 = np.asarray(pred_depths[t + 1], dtype=np.float32)

        if camera_poses[t] is not None and camera_poses[t + 1] is not None:
            d_t1_warped = _warp_depth_approx(d_t1, camera_poses[t], camera_poses[t + 1])
        else:
            d_t1_warped = d_t1

        valid = (d_t > 0) & (d_t1_warped > 0)
        if valid.sum() == 0:
            per_pair.append(1.0)
            continue
        l1 = float(np.abs(d_t[valid] - d_t1_warped[valid]).mean())
        depth_range = max(float(d_t[valid].max() - d_t[valid].min()), 1e-3)
        ts = max(0.0, 1.0 - l1 / depth_range)
        per_pair.append(ts)

    return TemporalStabilityResult(
        ts_score=float(np.mean(per_pair)),
        num_pairs=len(per_pair),
        per_pair=per_pair,
    )


def compute_sgc(
    pred_masks: Sequence[np.ndarray],
    pred_depths: Sequence[np.ndarray],
    depth_gradient_threshold: float = 0.1,
    boundary_dilation: int = 2,
) -> StackGeometricCoherenceResult:
    """Stack-Level Geometric Coherence: boundary F-score between mask and depth edges.

    SGC = F-score(boundary(mask), boundary(depth_gradient > τ))

    A high SGC means segmentation boundaries are geometrically consistent with
    the depth discontinuities — indicating the model perceives coherent surfaces.

    Args:
        pred_masks: sequence of predicted segmentation masks (H×W int).
        pred_depths: sequence of predicted depth maps (H×W float32, metres).
        depth_gradient_threshold: τ for depth gradient thresholding.
        boundary_dilation: pixel tolerance for boundary matching.
    """
    if len(pred_masks) == 0:
        return StackGeometricCoherenceResult(
            sgc_score=0.0, precision=0.0, recall=0.0, num_samples=0
        )

    precisions, recalls = [], []
    for mask, depth in zip(pred_masks, pred_depths, strict=False):
        mask = np.asarray(mask, dtype=np.int32)
        depth = np.asarray(depth, dtype=np.float32)

        mask_boundary = _extract_boundary(mask, dilation=boundary_dilation)
        depth_boundary = _extract_depth_boundary(
            depth, threshold=depth_gradient_threshold, dilation=boundary_dilation
        )

        tp = float((mask_boundary & depth_boundary).sum())
        fp = float((mask_boundary & ~depth_boundary).sum())
        fn = float((~mask_boundary & depth_boundary).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        precisions.append(p)
        recalls.append(r)

    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))
    f1 = (2 * mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0

    return StackGeometricCoherenceResult(
        sgc_score=f1,
        precision=mean_p,
        recall=mean_r,
        num_samples=len(pred_masks),
    )


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _extract_boundary(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    """Boolean boundary map from a semantic mask using finite differences."""
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    if dilation > 0:
        boundary = _binary_dilate(boundary, dilation)
    return boundary


def _extract_depth_boundary(depth: np.ndarray, threshold: float, dilation: int = 2) -> np.ndarray:
    """Boolean boundary map from depth via Sobel gradient magnitude."""
    gy = np.zeros_like(depth)
    gx = np.zeros_like(depth)
    gy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2.0
    gx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    boundary = grad_mag > threshold
    if dilation > 0:
        boundary = _binary_dilate(boundary, dilation)
    return boundary


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Simple box-dilation of a boolean mask (avoids scipy dependency)."""
    result = mask.copy()
    for _ in range(radius):
        tmp = result.copy()
        tmp[1:, :] |= result[:-1, :]
        tmp[:-1, :] |= result[1:, :]
        tmp[:, 1:] |= result[:, :-1]
        tmp[:, :-1] |= result[:, 1:]
        result = tmp
    return result


def _relative_pose(pose_t: np.ndarray, pose_t1: np.ndarray) -> np.ndarray:
    """Compute relative SE(3): T_{t→t+1} = pose_t^{-1} @ pose_{t+1}."""
    return np.linalg.inv(pose_t) @ pose_t1


def _warp_mask_approx(
    mask: np.ndarray,
    pose_t: np.ndarray,
    pose_t1: np.ndarray,
) -> np.ndarray:
    """Approximate mask warp using in-plane rotation from relative pose.

    Full pixel-accurate warping requires dense depth for backprojection.
    This approximation applies the in-plane rotation (yaw component) from
    the relative SE(3) transform as a 2D affine warp, which is a reasonable
    proxy for planar/forward-facing camera motion.
    """
    try:
        import cv2

        T_rel = _relative_pose(pose_t, pose_t1)
        R_rel = T_rel[:3, :3]
        angle_rad = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
        h, w = mask.shape
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), float(np.degrees(angle_rad)), 1.0)
        warped = cv2.warpAffine(
            mask.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=-1,
        ).astype(np.int32)
        return warped
    except ImportError:
        return mask  # fall back to unwarped if cv2 not available


# ================================================================== #
# Operating point (precision × accuracy × cost triple, no composite)
# ================================================================== #


@dataclass
class OperatingPoint:
    """One (precision, accuracy, cost) triple for a model.

    A model can publish multiple operating points — e.g. fp32, fp16,
    int8 — each reported separately on the three RPX axes (task
    accuracy, scene-change robustness, compute cost). RPX does not
    combine these into a single composite score: the reader picks the
    operating point and the axis their deployment cares about.
    """

    precision: str  # "fp32", "fp16", "bf16"
    # Task performance (primary metric — δ1, mIoU, accuracy, etc.)
    task_metric: float
    task_metric_name: str  # e.g. "delta1", "miou"
    higher_is_better: bool
    # Robustness
    str_score: float  # STR value (near 0 = robust)
    # Hardware-agnostic cost (Tier 1)
    flops_g: float
    params_m: float
    memory_traffic_gb: float | None = None


def _warp_depth_approx(
    depth: np.ndarray,
    pose_t: np.ndarray,
    pose_t1: np.ndarray,
) -> np.ndarray:
    """Approximate depth warp (same in-plane rotation proxy as mask warp)."""
    try:
        import cv2

        T_rel = _relative_pose(pose_t, pose_t1)
        R_rel = T_rel[:3, :3]
        angle_rad = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
        h, w = depth.shape
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), float(np.degrees(angle_rad)), 1.0)
        warped = cv2.warpAffine(
            depth,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
        return warped
    except ImportError:
        return depth

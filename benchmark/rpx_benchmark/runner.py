"""Benchmark runner orchestrating model, data, and metrics."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .api import BenchmarkModel, Difficulty, Phase, TaskType, validate_prediction
from .deployment import (
    DeploymentReadinessReport,
    StackGeometricCoherenceResult,
    TemporalStabilityResult,
    compute_sgc,
    compute_str,
    compute_temporal_stability_depth,
    compute_temporal_stability_seg,
    compute_weighted_phase_score,
)
from .evaluators import BenchmarkResult, MetricSuite
from .exceptions import ModelError
from .loader import RPXDataset
from .logging_utils import get_logger
from .profiler import EfficiencyMetadata

log = get_logger(__name__)

ProgressCallback = Callable[[int, Optional[int], str], None]
"""Signature: ``progress(done, total, stage) -> None``.

``total`` may be ``None`` while the total is unknown (e.g., mid-setup).
``stage`` is a short free-form label like ``"setup"``, ``"predict"``,
``"postprocess"``.
"""


def _sample_meta(sample: Any) -> Dict[str, Any]:
    """Extract id/phase/difficulty/scene for attachment to per-sample metrics."""
    meta: Dict[str, Any] = {"id": sample.id}
    if sample.phase is not None:
        meta["phase"] = sample.phase.value
    if sample.difficulty is not None:
        meta["difficulty"] = sample.difficulty.value
    # Scene id is not part of the Sample contract, but the loader passes
    # it through via id-prefix or metadata when available.
    if sample.metadata and "scene" in sample.metadata:
        meta["scene"] = sample.metadata["scene"]
    return meta


def _sum_nested_counts(node: Any) -> int:
    """Walk arbitrarily-nested FlopCounterMode output and sum int leaves.

    Torch changed the shape of ``FlopCounterMode.get_flop_counts`` between
    versions: older releases returned ``{op_name: int}``, newer releases
    return ``{module_name: {op_name: int}}`` (often with a top-level
    "Global" key alongside module-scoped dicts).
    """
    if isinstance(node, dict):
        return sum(_sum_nested_counts(v) for v in node.values())
    try:
        return int(node)
    except (TypeError, ValueError):
        return 0


def _count_flops_of(fn, *args, **kwargs) -> Tuple[Optional[float], Any]:
    """Invoke ``fn`` inside torch's FlopCounterMode and return ``(g_flops, result)``.

    ``g_flops`` is ``None`` when torch is not installed or the counter
    errors; in both cases ``fn`` is still invoked so the caller always
    gets the real result. Note that torch reports **FLOPs, not MACs** —
    the value is directly comparable to published `G_FLOPs` numbers.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None, fn(*args, **kwargs)
    try:
        with FlopCounterMode(display=False) as fcm:
            result = fn(*args, **kwargs)
        total_flops = _sum_nested_counts(fcm.get_flop_counts())
        if total_flops <= 0:
            return None, result
        return total_flops / 1e9, result
    except Exception:
        return None, fn(*args, **kwargs)


class BenchmarkRunner:
    """Runs a benchmark end-to-end for a given model.

    Basic usage::

        runner = BenchmarkRunner(model=model, dataset=dataset)
        result = runner.run()
        print(result.aggregated)

    Phase-stratified usage (requires manifest with ``phase`` / ``difficulty`` fields)::

        runner = BenchmarkRunner(model=model, dataset=dataset)
        result, dr_report = runner.run_with_deployment_readiness(
            primary_metric="absrel",
            model_name="MyDepthModel",
        )
    """

    def __init__(
        self,
        model: BenchmarkModel,
        dataset: RPXDataset,
        metric_suite: MetricSuite | None = None,
        call_setup: bool = True,
    ):
        self.model = model
        self.dataset = dataset
        self.metric_suite = metric_suite or MetricSuite.for_task(model.task)
        self.call_setup = call_setup
        self._validate_task_alignment()

    def _validate_task_alignment(self) -> None:
        if not isinstance(self.model.task, TaskType):
            raise ModelError(
                "BenchmarkModel must expose `task` as a TaskType enum.",
                hint=(
                    "Subclass rpx_benchmark.api.BenchmarkModel or use "
                    "rpx_benchmark.adapters.BenchmarkableModel which sets "
                    "task correctly."
                ),
            )
        if self.model.task != self.dataset.task:
            raise ModelError(
                f"Task mismatch between model ({self.model.task.value}) "
                f"and dataset ({self.dataset.task.value}).",
                hint="Build the dataset and model for the same TaskType.",
            )

    def run(
        self,
        progress: Optional[ProgressCallback] = None,
    ) -> BenchmarkResult:
        """Run benchmark and return flat per-sample + aggregated metrics."""
        if self.call_setup:
            if progress:
                progress(0, None, "setup")
            self.model.setup()

        total = len(self.dataset)
        per_sample: List[dict] = []
        if progress:
            progress(0, total, "predict")
        for batch in self.dataset:
            predictions = self.model.predict(batch)
            if len(predictions) != len(batch):
                raise ModelError(
                    f"Model returned {len(predictions)} predictions for "
                    f"a batch of {len(batch)} samples — must return one "
                    "prediction per sample.",
                )
            for sample, pred in zip(batch, predictions):
                validate_prediction(self.model.task, pred, sample)
                metrics = self.metric_suite.evaluate(pred, sample.ground_truth)
                per_sample.append(metrics)
                if progress:
                    progress(len(per_sample), total, "predict")

        return self.metric_suite.build_result(per_sample)

    def run_with_deployment_readiness(
        self,
        primary_metric: str,
        model_name: str = "model",
        efficiency: EfficiencyMetadata | None = None,
        compute_ts: bool = True,
        compute_sgc_flag: bool = True,
        progress: Optional[ProgressCallback] = None,
    ) -> tuple[BenchmarkResult, DeploymentReadinessReport]:
        """Run benchmark and compute all deployment-readiness metrics.

        Args:
            primary_metric: metric key used for ESD/STR scoring (e.g. "absrel", "miou").
            model_name: display name for the report.
            efficiency: pre-computed EfficiencyMetadata (params, FLOPs).
            compute_ts: whether to compute Temporal Stability (needs sequential frames).
            compute_sgc_flag: whether to compute SGC (needs both seg + depth predictions).

        Returns:
            (BenchmarkResult, DeploymentReadinessReport)
        """
        if self.call_setup:
            if progress:
                progress(0, None, "setup")
            self.model.setup()

        total = len(self.dataset)
        per_sample_metrics: List[dict] = []
        per_sample_phases: List[Phase | None] = []
        per_sample_difficulties: List[Difficulty | None] = []
        per_sample_poses: List[Any] = []
        all_predictions: List[Any] = []
        all_samples: List[Any] = []

        # Per-sample wall-clock inference time (seconds). First batch is
        # recorded separately so callers can discard warmup from the median.
        per_sample_seconds: List[float] = []
        first_batch_flops_g: Optional[float] = None

        if progress:
            progress(0, total, "predict")

        first_batch = True
        for batch in self.dataset:
            t0 = time.perf_counter()
            if first_batch:
                flops_g, predictions = _count_flops_of(self.model.predict, batch)
                if flops_g is not None and len(batch) > 0:
                    first_batch_flops_g = flops_g / len(batch)
                first_batch = False
            else:
                predictions = self.model.predict(batch)
            batch_seconds = time.perf_counter() - t0

            if len(predictions) != len(batch):
                raise ModelError(
                    f"Model returned {len(predictions)} predictions for "
                    f"a batch of {len(batch)} samples — must return one "
                    "prediction per sample.",
                )
            per_sample_seconds.extend([batch_seconds / len(batch)] * len(batch))

            for sample, pred in zip(batch, predictions):
                validate_prediction(self.model.task, pred, sample)
                metrics = self.metric_suite.evaluate(pred, sample.ground_truth)
                metrics.update(_sample_meta(sample))
                per_sample_metrics.append(metrics)
                per_sample_phases.append(sample.phase)
                per_sample_difficulties.append(sample.difficulty)
                per_sample_poses.append(sample.camera_pose)
                all_predictions.append(pred)
                all_samples.append(sample)
                if progress:
                    progress(len(per_sample_metrics), total, "predict")

        result = self.metric_suite.build_result(per_sample_metrics)

        # Compute latency median after skipping the first batch (warmup).
        latency_ms: Optional[float] = None
        if per_sample_seconds:
            warm = per_sample_seconds[1:] if len(per_sample_seconds) > 1 else per_sample_seconds
            latency_ms = round(1000.0 * float(np.median(warm)), 3)

        # --- Weighted Phase Score + STR ---
        wps = compute_weighted_phase_score(
            per_sample_metrics=per_sample_metrics,
            per_sample_phases=per_sample_phases,
            per_sample_difficulties=per_sample_difficulties,
            metric_key=primary_metric,
        )

        str_result = compute_str({
            Phase.CLUTTER: wps.s_clutter,
            Phase.INTERACTION: wps.s_interaction,
            Phase.CLEAN: wps.s_clean,
        })

        # --- Temporal Stability ---
        ts_result: TemporalStabilityResult | None = None
        if compute_ts and len(all_predictions) >= 2:
            task = self.model.task
            if task == TaskType.OBJECT_SEGMENTATION:
                masks = [p.mask for p in all_predictions]
                ts_result = compute_temporal_stability_seg(masks, per_sample_poses)
            elif task == TaskType.MONOCULAR_DEPTH:
                depths = [p.depth_map for p in all_predictions]
                ts_result = compute_temporal_stability_depth(depths, per_sample_poses)

        # --- Stack-Level Geometric Coherence (requires seg + depth in metadata) ---
        sgc_result: StackGeometricCoherenceResult | None = None
        if compute_sgc_flag:
            # SGC requires both mask and depth predictions in the same run.
            # When running segmentation, check if depth maps are in sample metadata.
            if self.model.task == TaskType.OBJECT_SEGMENTATION:
                depth_maps = [
                    s.metadata.get("depth_map") if s.metadata else None
                    for s in all_samples
                ]
                if any(d is not None for d in depth_maps):
                    valid = [(p.mask, d) for p, d in zip(all_predictions, depth_maps)
                             if d is not None]
                    masks_sgc = [v[0] for v in valid]
                    depths_sgc = [np.asarray(v[1], dtype=np.float32) for v in valid]
                    sgc_result = compute_sgc(masks_sgc, depths_sgc)

        # Merge counted FLOPs + measured latency into the passed-in
        # efficiency object (caller usually pre-computed params_m).
        flops_g = (efficiency.flops_g if efficiency and efficiency.flops_g else
                   first_batch_flops_g)
        report = DeploymentReadinessReport(
            task=self.model.task.value,
            model_name=model_name,
            weighted_phase_score=wps,
            temporal_stability=ts_result,
            state_transition=str_result,
            geometric_coherence=sgc_result,
            params_m=efficiency.params_m if efficiency else None,
            flops_g=flops_g,
            actmem_gb_fp16=efficiency.actmem_gb_fp16 if efficiency else None,
            latency_ms_per_sample=latency_ms,
        )

        return result, report

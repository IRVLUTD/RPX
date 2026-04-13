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
    compute_embodied_readiness,
    compute_str,
    compute_weighted_phase_score,
)
from .evaluators import BenchmarkResult, MetricSuite
from .exceptions import ModelError
from .loader import RPXDataset
from .logging_utils import get_logger
from .profiler import EfficiencyMetadata, LatencyProfiler, MemoryProfiler

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

    def _task_spec_or_none(self):
        """Fetch the TaskSpec for the current model, or None if unregistered.

        The runner supports ad-hoc models whose task is not wired into
        the pluggable task registry (e.g. user-defined custom tasks in
        a notebook). In that case deployment-readiness hooks simply
        don't fire and the caller gets back a report without TS / SGC
        populated.
        """
        try:
            from .tasks.registry import get_task_spec  # noqa: PLC0415 — lazy
            return get_task_spec(self.model.task)
        except Exception:
            return None

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

        # Latency + memory profiling. Both backends skip gracefully on
        # devices / libraries that don't support them, so this adds no
        # hard dependency to the runner.
        latency = LatencyProfiler(warmup=1)
        memory = MemoryProfiler().reset()
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
            latency.add_batch_seconds(batch_seconds, len(batch))

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

        # Attach per-sample latency_ms so downstream analysis can group
        # timings with the metric values. LatencyProfiler.samples_ms()
        # produces one entry per sample (batch timing amortised evenly
        # across the batch's samples).
        per_sample_latencies = latency.samples_ms()
        for row, lat_ms in zip(per_sample_metrics, per_sample_latencies):
            row["latency_ms"] = float(lat_ms)

        result = self.metric_suite.build_result(per_sample_metrics)

        latency_percentiles = latency.percentiles()
        latency_ms = latency_percentiles["p50_ms"]  # back-compat: median
        memory_peaks = memory.peaks()

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

        # --- Deployment-readiness hooks dispatched via TaskSpec ---
        # The runner no longer branches on task identity. Each task
        # that wants Temporal Stability or Stack Geometric Coherence
        # registers its own implementation on its TaskSpec; tasks
        # without a registered hook silently skip the computation.
        spec = self._task_spec_or_none()

        ts_result: TemporalStabilityResult | None = None
        if (
            compute_ts
            and len(all_predictions) >= 2
            and spec is not None
            and spec.temporal_stability_fn is not None
        ):
            ts_result = spec.temporal_stability_fn(
                all_predictions, all_samples, per_sample_poses,
            )

        sgc_result: StackGeometricCoherenceResult | None = None
        if (
            compute_sgc_flag
            and spec is not None
            and spec.geometric_coherence_fn is not None
        ):
            sgc_result = spec.geometric_coherence_fn(all_predictions, all_samples)

        # Merge counted FLOPs + measured latency + memory into the
        # passed-in efficiency object (caller usually pre-computed
        # params_m). A fresh object is created when the caller did not
        # provide one so downstream reporting always has a target to
        # write measured values onto.
        eff = efficiency or EfficiencyMetadata()
        eff.flops_g = eff.flops_g or first_batch_flops_g
        eff.latency_ms_per_sample = latency_ms
        eff.latency_p50_ms = latency_percentiles["p50_ms"]
        eff.latency_p95_ms = latency_percentiles["p95_ms"]
        eff.latency_p99_ms = latency_percentiles["p99_ms"]
        eff.peak_cpu_mb = memory_peaks["cpu_mb"]
        eff.peak_cuda_mb = memory_peaks["cuda_mb"]
        eff.peak_mps_mb = memory_peaks["mps_mb"]

        # Peak resident memory across the three possible backends
        # (CPU / CUDA / MPS). We pick the max so the ERS has a single
        # "worst-case device memory" number regardless of where the
        # model actually ran.
        peak_mb_vals = [v for v in memory_peaks.values() if v is not None]
        peak_memory_mb = max(peak_mb_vals) if peak_mb_vals else None

        report = DeploymentReadinessReport(
            task=self.model.task.value,
            model_name=model_name,
            weighted_phase_score=wps,
            temporal_stability=ts_result,
            state_transition=str_result,
            geometric_coherence=sgc_result,
            params_m=eff.params_m,
            flops_g=eff.flops_g,
            actmem_gb_fp16=eff.actmem_gb_fp16,
            latency_ms_per_sample=eff.latency_ms_per_sample,
            peak_memory_mb=peak_memory_mb,
        )

        # Compose the Embodied Readiness Score. Look up the TaskSpec to
        # know whether the primary metric is higher-is-better — the ERS
        # normalisation flips direction on that.
        higher_is_better = bool(getattr(spec, "higher_is_better", True))
        try:
            report.embodied_readiness = compute_embodied_readiness(
                report, higher_is_better=higher_is_better,
            )
        except ValueError:
            # All ERS components were None — rare, but don't block the
            # run on a composite we can't compute.
            pass

        return result, report

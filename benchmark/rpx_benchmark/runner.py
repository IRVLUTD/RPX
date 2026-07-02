"""Benchmark runner orchestrating model, data, and metrics."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .api import (
    BenchmarkModel,
    DepthPrediction,
    Difficulty,
    Phase,
    TaskType,
    validate_prediction,
)
from .deployment import (
    DeploymentReadinessReport,
    StackGeometricCoherenceResult,
    TemporalStabilityResult,
    compute_str,
    compute_weighted_phase_score,
)
from .evaluators import BenchmarkResult, MetricSuite
from .exceptions import ModelError
from .loader import RPXDataset
from .logging_utils import get_logger
from .metrics.depth_alignment import align_pred_to_gt_pooled
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
    # it through via metadata. Manifests write "scene_id"; accept either
    # so the cell-log bucketing (keyed on "scene") never silently drops
    # rows. See cell_log.cells_from_per_sample.
    if sample.metadata:
        scene = sample.metadata.get("scene_id") or sample.metadata.get("scene")
        if scene is not None:
            meta["scene"] = scene
    return meta


def _align_monocular_depth_predictions_pooled(
    predictions: List[DepthPrediction],
    samples: List[Any],
    *,
    mode: str,
) -> List[DepthPrediction]:
    """Align raw relative-depth predictions once per scene/phase cell.

    The manifest may interleave cells, so grouping is by explicit metadata
    rather than iteration adjacency. Metric models never enter this helper.
    """
    groups: Dict[tuple[str, str], List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        meta = _sample_meta(sample)
        scene = meta.get("scene")
        phase = meta.get("phase")
        if scene is None or phase is None:
            raise ModelError(
                "Relative Image Depth evaluation requires scene and phase metadata "
                "for pooled alignment.",
                hint=(
                    "Use an RPX manifest whose samples include scene_id and phase; "
                    "per-frame alignment is intentionally not used."
                ),
            )
        groups[(str(scene), str(phase))].append(idx)

    aligned: List[DepthPrediction | None] = [None] * len(predictions)
    for indices in groups.values():
        pred_seq = np.stack(
            [np.asarray(predictions[i].depth_map, dtype=np.float32) for i in indices]
        )
        gt_seq = np.stack(
            [np.asarray(samples[i].ground_truth.depth_map, dtype=np.float32) for i in indices]
        )
        aligned_seq = align_pred_to_gt_pooled(pred_seq, gt_seq, mode=mode)
        for frame_idx, source_idx in enumerate(indices):
            aligned[source_idx] = DepthPrediction(
                depth_map=np.asarray(aligned_seq[frame_idx], dtype=np.float32)
            )
    return [pred for pred in aligned if pred is not None]


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


def _make_cuda_sync():
    """Return a callable that synchronises CUDA when available, else a no-op.

    Resolved once at runner setup so the per-batch hot loop only pays
    the function-call overhead (no per-call ``torch.cuda.is_available()``
    branch). On CPU-only systems or when torch is missing, returns a
    no-op so the runner stays usable in lightweight test environments.
    """
    try:
        import torch
    except ImportError:
        return lambda: None
    if not torch.cuda.is_available():
        return lambda: None
    return torch.cuda.synchronize


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
        result, report = runner.run_with_report(
            primary_metric="absrel",
            model_name="MyDepthModel",
        )

    The returned ``DeploymentReadinessReport`` carries the per-axis
    components (scene-change robustness + compute cost). RPX no longer
    combines those into a single composite — see
    ``benchmark/SHARED_CONTEXT.md`` for the three-axis policy.
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
        all_predictions: List[Any] = []
        all_samples: List[Any] = []
        is_relative_depth = (
            self.model.task is TaskType.MONOCULAR_DEPTH
            and getattr(self.model, "depth_output_kind", "metric") == "relative"
        )
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
            for sample, pred in zip(batch, predictions, strict=False):
                validate_prediction(self.model.task, pred, sample)
                all_predictions.append(pred)
                all_samples.append(sample)
                if not is_relative_depth:
                    metrics = self.metric_suite.evaluate(pred, sample.ground_truth)
                    per_sample.append(metrics)
                if progress:
                    progress(len(all_samples), total, "predict")

        if is_relative_depth:
            mode = getattr(self.model, "native_alignment", "ls_affine")
            all_predictions = _align_monocular_depth_predictions_pooled(
                all_predictions,
                all_samples,
                mode=mode,
            )
            per_sample = [
                self.metric_suite.evaluate(pred, sample.ground_truth)
                for sample, pred in zip(all_samples, all_predictions, strict=True)
            ]

        return self.metric_suite.build_result(per_sample)

    def run_with_report(
        self,
        primary_metric: str,
        model_name: str = "model",
        efficiency: EfficiencyMetadata | None = None,
        compute_ts: bool = True,
        compute_sgc_flag: bool = True,
        skip_flops: bool = False,
        progress: Optional[ProgressCallback] = None,
    ) -> tuple[BenchmarkResult, DeploymentReadinessReport]:
        """Run benchmark and compute all per-axis components.

        The returned report carries the scene-change-robustness and
        compute-cost components. RPX never combines these into a
        single composite — see ``benchmark/SHARED_CONTEXT.md`` for the
        three-axis policy.

        Args:
            primary_metric: metric key used for ESD/STR scoring (e.g. "absrel", "miou").
            model_name: display name for the report.
            efficiency: pre-computed EfficiencyMetadata (params, FLOPs).
            compute_ts: whether to compute Temporal Stability (needs sequential frames).
            compute_sgc_flag: whether to compute SGC (needs both seg + depth predictions).
            skip_flops: skip the first-batch FLOP counter.  Useful for
                high-VRAM models on small GPUs where the FLOP-counter
                dispatch metadata alone may cause OOM even though
                inference fits.

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
        spec = self._task_spec_or_none()
        is_relative_depth = (
            self.model.task is TaskType.MONOCULAR_DEPTH
            and getattr(self.model, "depth_output_kind", "metric") == "relative"
        )
        incremental_depth_ts = (
            self.model.task is TaskType.MONOCULAR_DEPTH
            and compute_ts
            and spec is not None
            and spec.temporal_stability_fn is not None
        )
        retain_for_hooks = (
            not incremental_depth_ts
            and (
                (compute_ts and spec is not None and spec.temporal_stability_fn is not None)
                or (
                    compute_sgc_flag
                    and spec is not None
                    and spec.geometric_coherence_fn is not None
                )
            )
        )
        depth_ts_scores: List[float] = []
        previous_depth_prediction: Any = None
        previous_depth_pose: Any = None
        previous_depth_key: tuple[str, str] | None = None
        pending_samples: List[Any] = []
        pending_predictions: List[DepthPrediction] = []
        pending_key: tuple[str, str] | None = None
        closed_keys: set[tuple[str, str]] = set()

        def _record_evaluation(sample: Any, pred: Any) -> None:
            nonlocal previous_depth_key, previous_depth_pose, previous_depth_prediction
            metrics = self.metric_suite.evaluate(pred, sample.ground_truth)
            metrics.update(_sample_meta(sample))
            per_sample_metrics.append(metrics)
            per_sample_phases.append(sample.phase)
            per_sample_difficulties.append(sample.difficulty)
            per_sample_poses.append(sample.camera_pose)
            if incremental_depth_ts:
                meta = _sample_meta(sample)
                current_key = (str(meta.get("scene")), str(meta.get("phase")))
                if previous_depth_prediction is not None and previous_depth_key == current_key:
                    pair_result = spec.temporal_stability_fn(
                        [previous_depth_prediction, pred],
                        [None, None],
                        [previous_depth_pose, sample.camera_pose],
                    )
                    if pair_result.per_pair:
                        depth_ts_scores.extend(float(value) for value in pair_result.per_pair)
                previous_depth_prediction = pred
                previous_depth_pose = sample.camera_pose
                previous_depth_key = current_key
            elif retain_for_hooks:
                all_predictions.append(pred)
                all_samples.append(sample)

        def _flush_relative_cell() -> None:
            if not pending_samples:
                return
            mode = getattr(self.model, "native_alignment", "ls_affine")
            aligned = _align_monocular_depth_predictions_pooled(
                pending_predictions,
                pending_samples,
                mode=mode,
            )
            for cell_sample, cell_pred in zip(pending_samples, aligned, strict=True):
                _record_evaluation(cell_sample, cell_pred)
            pending_samples.clear()
            pending_predictions.clear()

        # Latency + memory profiling. Both backends skip gracefully on
        # devices / libraries that don't support them, so this adds no
        # hard dependency to the runner.
        latency = LatencyProfiler(warmup=1)
        memory = MemoryProfiler().reset()
        first_batch_flops_g: Optional[float] = None

        # CUDA kernels are async — ``model.predict`` returns when work
        # is *queued*, not when it *completes*. Without a synchronise
        # before stopping the timer, latency_ms in the cell log
        # silently undercounts every GPU-bound model. Sync once before
        # t0 (so kernels from outside the loop don't bleed into our
        # measurement) and once after predict (so we measure to GPU
        # completion). On CPU-only runs this is a no-op.
        _cuda_sync = _make_cuda_sync()

        if progress:
            progress(0, total, "predict")

        first_batch = True
        prediction_count = 0
        for batch in self.dataset:
            _cuda_sync()
            t0 = time.perf_counter()
            if first_batch and not skip_flops:
                # Count FLOPs on a single sample to avoid scaling
                # the FLOP-counter dispatch overhead by batch size.
                # FLOPs are per-sample, so counting on one is enough.
                single = [batch[0]]
                flops_g, _ = _count_flops_of(self.model.predict, single)
                first_batch_flops_g = flops_g  # already per-sample
                # Now run the full batch normally (timed)
                predictions = self.model.predict(batch)
                first_batch = False
            elif first_batch and skip_flops:
                predictions = self.model.predict(batch)
                first_batch = False
            else:
                predictions = self.model.predict(batch)
            _cuda_sync()
            batch_seconds = time.perf_counter() - t0

            if len(predictions) != len(batch):
                raise ModelError(
                    f"Model returned {len(predictions)} predictions for "
                    f"a batch of {len(batch)} samples — must return one "
                    "prediction per sample.",
                )
            latency.add_batch_seconds(batch_seconds, len(batch))

            for sample, pred in zip(batch, predictions, strict=False):
                validate_prediction(self.model.task, pred, sample)
                if is_relative_depth:
                    meta = _sample_meta(sample)
                    if meta.get("scene") is None or meta.get("phase") is None:
                        raise ModelError(
                            "Relative Image Depth evaluation requires scene and phase "
                            "metadata for pooled alignment.",
                            hint="Use an RPX manifest with scene_id and phase fields.",
                        )
                    key = (str(meta["scene"]), str(meta["phase"]))
                    if pending_key is None:
                        pending_key = key
                    elif key != pending_key:
                        _flush_relative_cell()
                        closed_keys.add(pending_key)
                        if key in closed_keys:
                            raise ModelError(
                                f"Manifest cell {key!r} is non-contiguous; pooled alignment "
                                "cannot be streamed safely.",
                                hint="Sort manifest samples by scene_id, phase and frame_idx.",
                            )
                        pending_key = key
                    pending_samples.append(sample)
                    pending_predictions.append(pred)
                else:
                    _record_evaluation(sample, pred)
                prediction_count += 1
                if progress:
                    progress(prediction_count, total, "predict")

        if is_relative_depth:
            log.info(
                "model %s is relative-depth: applying %s once per streamed scene/phase cell",
                model_name,
                getattr(self.model, "native_alignment", "ls_affine"),
            )
            _flush_relative_cell()

        # Attach per-sample latency_ms so downstream analysis can group
        # timings with the metric values. LatencyProfiler.samples_ms()
        # produces one entry per sample (batch timing amortised evenly
        # across the batch's samples).
        per_sample_latencies = latency.samples_ms()
        for row, lat_ms in zip(per_sample_metrics, per_sample_latencies, strict=False):
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

        str_result = compute_str(
            {
                Phase.CLUTTER: wps.s_clutter,
                Phase.INTERACTION: wps.s_interaction,
                Phase.CLEAN: wps.s_clean,
            }
        )

        # --- Deployment-readiness hooks dispatched via TaskSpec ---
        # The runner no longer branches on task identity. Each task
        # that wants Temporal Stability or Stack Geometric Coherence
        # registers its own implementation on its TaskSpec; tasks
        # without a registered hook silently skip the computation.
        ts_result: TemporalStabilityResult | None = None
        if incremental_depth_ts:
            ts_result = TemporalStabilityResult(
                ts_score=float(np.mean(depth_ts_scores)) if depth_ts_scores else 1.0,
                num_pairs=len(depth_ts_scores),
                per_pair=depth_ts_scores,
            )
        elif (
            compute_ts
            and len(all_predictions) >= 2
            and spec is not None
            and spec.temporal_stability_fn is not None
        ):
            ts_result = spec.temporal_stability_fn(
                all_predictions,
                all_samples,
                per_sample_poses,
            )

        sgc_result: StackGeometricCoherenceResult | None = None
        if compute_sgc_flag and spec is not None and spec.geometric_coherence_fn is not None:
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

        # Ensure derived Tier-1 fields (MACs, traffic, AI) are populated
        # now that flops_g may have been filled by the FLOP counter above.
        eff.derive_tier1()

        # Compute Tier-2 roofline bounds if we have enough data
        if eff.flops_g is not None and eff.roofline is None:
            eff.compute_roofline()

        # Serialise roofline bounds for the report
        roofline_dict = None
        if eff.roofline:
            roofline_dict = {
                name: {
                    "compute_ms": b.compute_ms,
                    "memory_ms": b.memory_ms,
                    "latency_ms": b.latency_ms,
                    "bottleneck": b.bottleneck,
                }
                for name, b in eff.roofline.items()
            }

        # Serialise system card if present
        system_card_dict = eff.system_card.to_dict() if eff.system_card else None

        report = DeploymentReadinessReport(
            task=self.model.task.value,
            model_name=model_name,
            weighted_phase_score=wps,
            temporal_stability=ts_result,
            state_transition=str_result,
            geometric_coherence=sgc_result,
            # Tier 1
            params_m=eff.params_m,
            flops_g=eff.flops_g,
            macs_g=eff.macs_g,
            actmem_gb_fp16=eff.actmem_gb_fp16,
            memory_traffic_gb=eff.memory_traffic_gb,
            arithmetic_intensity=eff.arithmetic_intensity,
            # Tier 2
            roofline=roofline_dict,
            # Tier 3
            latency_ms_per_sample=eff.latency_ms_per_sample,
            peak_memory_mb=peak_memory_mb,
            system_card=system_card_dict,
        )

        # Build an OperatingPoint for this run (the precision × accuracy ×
        # cost triple). RPX reports operating points but never combines
        # them into a composite score — see SHARED_CONTEXT.md.
        higher_is_better = bool(getattr(spec, "higher_is_better", True))
        from .deployment import OperatingPoint  # noqa: PLC0415

        if wps is not None and eff.flops_g is not None:
            # Precision: prefer the model's declared native_precision,
            # fall back to the system card, default to fp32.
            precision = getattr(self.model, "native_precision", None) or (
                eff.system_card.get("precision", "fp32")
                if isinstance(eff.system_card, dict)
                else getattr(eff.system_card, "precision", "fp32")
                if eff.system_card
                else "fp32"
            )
            str_val = 0.0
            if str_result is not None:
                str_val = str_result.str_c_to_i or 0.0
            report.operating_point = OperatingPoint(
                precision=precision,
                task_metric=wps.s_overall,
                task_metric_name=primary_metric,
                higher_is_better=higher_is_better,
                str_score=str_val,
                flops_g=eff.flops_g,
                params_m=eff.params_m or 0.0,
                memory_traffic_gb=eff.memory_traffic_gb,
            )

        return result, report

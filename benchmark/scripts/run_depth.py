"""Run a monocular-metric-depth model end-to-end against an RPX split.

By default emits aggregated metrics + a markdown summary into
``./rpx_results/<model>/<split>/`` (community-friendly: metrics only).

Two opt-in features for our team's analytics workflow:

* ``--save-predictions`` — also save each per-frame depth prediction as a
  compressed ``.npz`` under ``./rpx_results/<model>/<split>/predictions/``.
* ``--upload-to-box`` — after the run, ship the entire result dir to UTD Box
  under ``<box_folder_id>/monocular_depth/<model>/<split>/``. Requires
  ``BOX_DEVELOPER_TOKEN``. Off by default; community users keep results local.

Two manifest paths:

* Default: use the official pinned HF manifest via
  :func:`run_monocular_depth` (download → load → metrics → report).
* Explicit legacy fallback: build a local manifest from the cached frames
  Parquet (``local_manifest.py``).

Usage
-----
    # community-friendly default: just aggregated metrics
    PYTHONPATH=. python scripts/run_depth.py --model zoedepth --split easy

    # team analytics workflow: per-frame predictions + Box upload
    PYTHONPATH=. python scripts/run_depth.py --model zoedepth --split easy \\
        --save-predictions --upload-to-box
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DATASET_REPO = "anonymous/RPX"
PINNED_DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"


def _human_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}" if isinstance(n, float) else f"{n} {u}"
        n = n / 1024
    return f"{n:.1f} PB"


# Make ./scripts importable so `from depth_models.* import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# BatchedDepthBenchmarkModel lives in the toolkit so every adapter (here
# and in any future model zoo) imports from a stable location.
from rpx_benchmark.adapters.batched_depth import BatchedDepthBenchmarkModel  # noqa: E402
from rpx_benchmark.metrics.depth_paper import (  # noqa: E402
    FastPaperDepthMetricSuite,
    PaperDepthMetricSuite,
)


def _stage_timing(
    adapter,
    dataset,
    max_samples: int | None = None,
    alignment: str = "none",
) -> dict:
    """Time three stages independently per sample: data loading, model run,
    and metric calculation (a single AbsRel pass — proxy for the full basket).

    Returns ``{stage: {"mean": ms, "median": ms, "p95": ms, "n": int}, ...}``
    where stages are ``data_load``, ``model_run``, ``metric_calc``.
    Synchronises CUDA between stages so model-run numbers reflect the GPU
    work, not async kernel launch.
    """
    import time as _t

    import numpy as _np

    try:
        import torch as _torch

        _has_cuda = _torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        _torch, _has_cuda = None, False

    def _sync():
        if _has_cuda:
            _torch.cuda.synchronize()

    from stat_utils import summarize_with_ci as _summary  # noqa: PLC0415

    # RPXDataset is iter-only (yields batches), not subscriptable. Use
    # `_load_sample` against the underlying entries list so we can time one
    # sample at a time without batch granularity getting in the way.
    entries = list(dataset.samples)
    if max_samples is not None:
        entries = entries[:max_samples]
    n = len(entries)

    # Stage 1: data loading (decoded sample → np.ndarray RGB ready for the model)
    samples, gts, sample_objs, data_ms = [], [], [], []
    for entry in entries:
        t0 = _t.perf_counter()
        s = dataset._load_sample(entry)
        rgb = _np.asarray(s.rgb, dtype=_np.uint8)
        # Sample.ground_truth is the task-specific GT (DepthGroundTruth here).
        gt_obj = s.ground_truth
        gt_arr = getattr(gt_obj, "depth_map", None)
        if gt_arr is None:
            gt_arr = getattr(gt_obj, "depth", None)
        gt = (
            _np.asarray(gt_arr, dtype=_np.float32)
            if gt_arr is not None
            else _np.zeros_like(rgb[..., 0], dtype=_np.float32)
        )
        data_ms.append((_t.perf_counter() - t0) * 1000.0)
        samples.append(rgb)
        gts.append(gt)
        sample_objs.append(s)

    # Warm up — first 1–3 forwards include CUDA kernel selection / cuDNN
    # autotune / lazy weight uploads that distort the latency CI.
    if samples:
        n_warmup = min(3, len(samples))
        for _ in range(n_warmup):
            _sync()
            adapter(samples[0])
            _sync()

    # Stage 2: model forward
    preds, model_ms = [], []
    for rgb in samples:
        _sync()
        t0 = _t.perf_counter()
        p = adapter(rgb)
        _sync()
        model_ms.append((_t.perf_counter() - t0) * 1000.0)
        preds.append(_np.asarray(p, dtype=_np.float32))

    if alignment != "none" and preds:
        from collections import defaultdict as _defaultdict

        from rpx_benchmark.metrics.depth_alignment import align_pred_to_gt_pooled

        groups = _defaultdict(list)
        for i, sample in enumerate(sample_objs):
            meta = sample.metadata or {}
            groups[(meta.get("scene_id"), sample.phase)].append(i)
        for indices in groups.values():
            aligned = align_pred_to_gt_pooled(
                _np.stack([preds[i] for i in indices]),
                _np.stack([gts[i] for i in indices]),
                mode=alignment,
            )
            for cell_idx, source_idx in enumerate(indices):
                preds[source_idx] = aligned[cell_idx]

    # Stage 3: metric calculation — full per-sample depth basket
    # (AbsRel, SqRel, RMSE, RMSElog, MAE, δ1, δ2, δ3). Each sample's
    # metric pass is timed in its entirety; the resulting per-sample
    # values are also returned so the caller can CI-aggregate them.
    metric_ms = []
    per_sample_depth = []
    for p, gt in zip(preds, gts, strict=False):
        valid = _np.isfinite(gt) & (gt > 0) & _np.isfinite(p)
        if not _np.any(valid):
            metric_ms.append(0.0)
            continue
        t0 = _t.perf_counter()
        gv = gt[valid]
        pv = p[valid]
        rel = _np.abs(pv - gv) / gv
        absrel = float(rel.mean())
        sq_rel = float(((pv - gv) ** 2 / gv).mean())
        rmse = float(_np.sqrt(((pv - gv) ** 2).mean()))
        log_p = _np.log(_np.maximum(pv, 1e-9))
        log_g = _np.log(_np.maximum(gv, 1e-9))
        rmselog = float(_np.sqrt(((log_p - log_g) ** 2).mean()))
        mae = float(_np.abs(pv - gv).mean())
        ratio = _np.maximum(pv / gv, gv / pv)
        d1 = float((ratio < 1.25).mean())
        d2 = float((ratio < 1.25**2).mean())
        d3 = float((ratio < 1.25**3).mean())
        metric_ms.append((_t.perf_counter() - t0) * 1000.0)
        per_sample_depth.append(
            {
                "absrel": absrel,
                "sq_rel": sq_rel,
                "rmse": rmse,
                "rmse_log": rmselog,
                "mae": mae,
                "delta1": d1,
                "delta2": d2,
                "delta3": d3,
            }
        )

    # Aggregate the depth basket with 95% CIs across the full split.
    from stat_utils import aggregate_per_sample_with_ci  # noqa: PLC0415

    metrics_with_ci = aggregate_per_sample_with_ci(per_sample_depth, drop_keys=())

    return {
        "data_load": _summary(data_ms),
        "model_run": _summary(model_ms),
        "metric_calc": _summary(metric_ms),
        "metrics_with_ci": metrics_with_ci,
        "n_warmup": n_warmup if samples else 0,
        "unit": "ms",
    }


def _augment_result_with_timing(json_path, *, timing) -> None:
    """Add the per-stage timing breakdown to result.json.

    Efficiency fields are no longer injected here — the runner's
    `DeploymentReadinessReport` carries the full Tier 1/2/3 picture
    natively, and `write_json` serialises all of it under the
    `compute_cost` and `robustness` top-level blocks (see
    `benchmark/README.md` for the three-axis policy). This helper is
    now timing-only.
    """
    import json as _json

    payload = _json.loads(json_path.read_text())
    payload["timing"] = timing
    json_path.write_text(_json.dumps(payload, indent=2))


def _find_torch_module(adapter):
    """Walk into the adapter to find the underlying torch nn.Module.

    Adapters wrap a callable; the actual nn.Module hangs off
    `.torch_module` (preferred) or HF-pipeline fallbacks
    (`._pipe.model`, `._model`, `.model`). Returns the first object
    that exposes a `.parameters()` method, or None.
    """
    for path in ("torch_module", "_pipe.model", "_model", "model"):
        cur = adapter
        for part in path.split("."):
            cur = getattr(cur, part, None) if cur is not None else None
            if cur is None:
                break
        if cur is not None and hasattr(cur, "parameters"):
            return cur
    return None


def _build_model(
    name: str,
    device: str,
    batch_size: int = 1,
    *,
    acknowledge_unverified: bool = False,
    precision: str = "auto",
):
    """Build ``(BenchmarkableModel placeholder, raw_adapter)``.

    The placeholder model is only used for its ``.name`` attribute — the
    real predict path goes through ``BatchedDepthBenchmarkModel`` which
    is constructed inside ``_run_via_local_manifest`` once the runner's
    batch size is known.
    """
    from depth_models import MODEL_DISPLAY_NAMES, MODEL_REGISTRY, resolve_model_key

    import rpx_benchmark as rpx

    # Resolve canonical roster keys (kebab-case, e.g. ``da-v2-large``)
    # OR legacy registry keys (snake_case, e.g. ``da_v2_metric_indoor``)
    # to the registry factory. ``resolve_model_key`` raises SystemExit
    # with a friendly error if neither resolves.
    registry_key = resolve_model_key(name)
    # Only pass acknowledge_unverified to factories that accept it
    # (currently only the FE2E safety-rail adapter). Other factories
    # don't have the kwarg and would TypeError if we always passed it.
    factory = MODEL_REGISTRY[registry_key]
    kwargs = {"device": device, "batch_size": batch_size}
    if registry_key in {"da_v2_metric_indoor", "da_v2_metric_outdoor"} and precision != "auto":
        import torch

        kwargs["dtype"] = torch.float16 if precision == "fp16" else torch.float32
    if acknowledge_unverified:
        kwargs["acknowledge_unverified"] = True
    try:
        adapter = factory(**kwargs)
    except TypeError as e:
        if "acknowledge_unverified" in str(e):
            # Factory doesn't accept the flag — drop it and retry.
            kwargs.pop("acknowledge_unverified", None)
            adapter = factory(**kwargs)
        else:
            raise
    # Prefer the canonical roster's display name when the caller used the
    # canonical key; otherwise fall back to the legacy registry's display name.
    from rpx_benchmark.adapters.depth_scaffold import DEPTH_MODEL_CARDS

    display_name = (
        DEPTH_MODEL_CARDS[name].name
        if name in DEPTH_MODEL_CARDS
        else MODEL_DISPLAY_NAMES.get(registry_key, registry_key)
    )
    native_alignment = getattr(adapter, "native_alignment", "none")
    return rpx.make_numpy_depth_model(
        adapter,
        name=display_name,
        depth_output_kind=("relative" if native_alignment != "none" else "metric"),
        native_alignment=native_alignment,
    ), adapter


def _run_via_official_pipeline(
    *,
    model,
    adapter,
    split: str,
    repo_id: str,
    device: str,
    output_dir: str | None,
    batch_size: int,
    manifest_path: str | None = None,
    revision: str | None = None,
    max_samples: int | None = None,
    save_predictions: bool = False,
    require_cuda: bool = True,
    skip_flops: bool = False,
    cache_dir: str | None = None,
    resume_predictions: bool = False,
    paper_protocol: bool = False,
    defer_fscore: bool = False,
):
    """Path A: toolkit's `run_monocular_depth`.

    Two sub-paths:

    * ``manifest_path is not None`` — load that manifest from disk
      directly. Use for runs against a locally-staged lossless
      (v2-webp) tree before the HF upload.
    * ``manifest_path is None`` — pull ``manifests/<task>/<split>.json``
      from the HF repo (the canonical post-publish path).
    """
    from rpx_benchmark import MonocularDepthRunConfig, run_monocular_depth

    name = getattr(model, "name", "model").replace("/", "__")
    out_dir = Path(output_dir or f"./rpx_results/{name}/{split}")
    pred_dir = out_dir / "predictions"
    benchmark_model = BatchedDepthBenchmarkModel(
        adapter,
        name=name,
        save_dir=(pred_dir if save_predictions else None),
        native_alignment=getattr(adapter, "native_alignment", "none"),
        native_precision=getattr(adapter, "native_precision", None),
        resume_predictions=resume_predictions,
        allow_nonpositive_predictions=paper_protocol,
    )
    cfg = MonocularDepthRunConfig(
        model=benchmark_model,
        split=split,
        repo_id=repo_id,
        device=device,
        output_dir=output_dir,
        batch_size=batch_size,
        manifest_path=manifest_path,
        revision=revision,
        max_samples=max_samples,
        require_cuda=require_cuda,
        skip_flops=skip_flops,
        cache_dir=cache_dir,
        metric_suite=(
            FastPaperDepthMetricSuite()
            if paper_protocol and defer_fscore
            else PaperDepthMetricSuite() if paper_protocol else None
        ),
        compute_temporal_stability=not paper_protocol,
    )
    result, report, paths = run_monocular_depth(cfg)
    if save_predictions:
        paths["predictions_dir"] = pred_dir
        paths["prediction_stats"] = dict(benchmark_model.resume_stats)
    return result, report, paths


def _run_via_local_manifest(
    *,
    model,
    adapter,
    split: str,
    repo_id: str,
    device: str,
    output_dir: str | None,
    batch_size: int,
    max_samples: int | None,
    save_predictions: bool,
    revision: str | None,
    require_cuda: bool,
    skip_flops: bool,
    resume_predictions: bool = False,
    paper_protocol: bool = False,
    defer_fscore: bool = False,
):
    """Path B: build a manifest locally from the cached frames Parquet, then
    call BenchmarkRunner directly. Mirrors `_pipeline.run_pipeline` minus the
    download step. Optionally wraps the model to also save per-frame .npz."""
    from local_manifest import build_local_manifest

    from rpx_benchmark.api import TaskType
    from rpx_benchmark.loader import RPXDataset
    from rpx_benchmark.metrics.registry import MetricSuite
    from rpx_benchmark.profiler import (
        EfficiencyMetadata,
        SystemCard,
        count_parameters,
        estimate_memory_traffic_gb,
    )
    from rpx_benchmark.reports import format_markdown_summary, write_json
    from rpx_benchmark.runner import BenchmarkRunner
    from rpx_benchmark.tasks._pipeline import resolve_device

    device = resolve_device(device, require_cuda=require_cuda)
    print(f"[local-pipeline] task=monocular_depth split={split} device={device}")

    res = build_local_manifest(
        task="monocular_depth",
        split=split,
        repo_id=repo_id,
        revision=revision,
        max_samples=max_samples,
    )
    print(f"[local-pipeline] manifest: {res.manifest_path}  ({res.n_samples} samples)")

    name = getattr(model, "name", "model").replace("/", "__")
    out_dir = Path(output_dir or f"./rpx_results/{name}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"

    # Replace the per-sample BenchmarkableModel with our true-batch wrapper
    # so the adapter sees the entire batch at once (HF pipeline batches
    # the GPU forward natively). The dataset's batch_size dictates the
    # batch size; the adapter/model semantics are unchanged. Adapter's
    # `native_alignment` is propagated so the runner / post-processor
    # apply the right alignment by default (metric → none, diffusion → ls_affine).
    model = BatchedDepthBenchmarkModel(
        adapter,
        name=name,
        save_dir=(pred_dir if save_predictions else None),
        native_alignment=getattr(adapter, "native_alignment", "none"),
        native_precision=getattr(adapter, "native_precision", None),
        resume_predictions=resume_predictions,
        allow_nonpositive_predictions=paper_protocol,
    )
    dataset = RPXDataset.from_manifest(res.manifest_path, batch_size=batch_size)
    print(
        f"[local-pipeline] batch_size={batch_size}  predictions_dir="
        f"{'<scene>/<phase>/<frame>.npz layout' if save_predictions else 'n/a'}"
    )

    model.setup()
    # Caller side of the 3-tier efficiency contract (the runner now
    # owns derive_tier1 + compute_roofline + flops counting). We only
    # provide the bits the runner can't infer on its own:
    #   - params_m       (count_parameters needs the torch module)
    #   - memory_traffic (needs the torch module + a forward shape)
    #   - system_card    (needs to read host CUDA / OS fields)
    # Everything else (FLOPs, MACs, AI, roofline, latency, peak memory)
    # is computed inside `runner.run_with_report` and serialised under
    # `result.json["compute_cost"]` (three-axis schema).
    torch_mod = _find_torch_module(adapter)
    eff = EfficiencyMetadata(
        model_type="local",
        notes=f"{getattr(model, 'name', 'model')} @ 480x640",
    )
    if torch_mod is not None:
        try:
            eff.params_m = count_parameters(torch_mod)
        except Exception as e:  # noqa: BLE001
            print(f"[profiler] params count failed: {type(e).__name__}: {e}")
        try:
            eff.memory_traffic_gb = estimate_memory_traffic_gb(
                torch_mod,
                (3, 480, 640),
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[profiler] memory-traffic estimate failed: {type(e).__name__}: {e}")
    try:
        eff.system_card = SystemCard.auto_detect(input_resolution="640x480")
    except Exception as e:  # noqa: BLE001
        print(f"[profiler] system card detection failed: {type(e).__name__}: {e}")

    runner = BenchmarkRunner(
        model=model,
        dataset=dataset,
        metric_suite=(
            FastPaperDepthMetricSuite()
            if paper_protocol and defer_fscore
            else PaperDepthMetricSuite()
            if paper_protocol
            else MetricSuite.for_task(TaskType.MONOCULAR_DEPTH)
        ),
        call_setup=False,
    )
    bench_result, dr_report = runner.run_with_report(
        primary_metric="absrel",
        model_name=name,
        efficiency=eff,
        compute_ts=not paper_protocol,
        compute_sgc_flag=False,
        skip_flops=skip_flops,
    )
    # ↑ At this point dr_report carries the full Tier 1/2/3 efficiency
    # picture (params, flops, macs, mem-traffic, arithmetic intensity,
    # roofline bounds for A100/4090/Orin, measured latency, peak memory,
    # system card). `write_json` serialises all of it under
    # `result.json["compute_cost"]` (plus `["robustness"]` for the
    # phase-aware scores) — no post-hoc augmentation needed.

    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    write_json(
        json_path,
        task="monocular_depth",
        model_name=name,
        split=split,
        repo_id=repo_id,
        result=bench_result,
        dr_report=dr_report,
    )
    md_path.write_text(
        format_markdown_summary(
            task="monocular_depth",
            model_name=name,
            split=split,
            repo_id=repo_id,
            result=bench_result,
            dr_report=dr_report,
        ),
        encoding="utf-8",
    )

    artefacts = {"json": json_path, "markdown": md_path, "out_dir": out_dir}
    if save_predictions:
        artefacts["predictions_dir"] = pred_dir
        artefacts["prediction_stats"] = dict(model.resume_stats)
    return bench_result, dr_report, artefacts


def main() -> None:
    from rpx_benchmark.cleanup import install_signal_cleanup
    install_signal_cleanup()

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 2)[0])
    ap.add_argument("--model", default="zoedepth", help="adapter to use")
    ap.add_argument("--split", default="easy", help="easy | medium | hard")
    ap.add_argument("--repo", default=DEFAULT_DATASET_REPO, help="HuggingFace dataset repo")
    ap.add_argument(
        "--revision",
        default=PINNED_DATASET_REVISION,
        help="immutable Hugging Face dataset commit",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--precision",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="explicit torch dtype for DA-V2; production paper runs use fp16",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache root; defaults to HF_HOME/huggingface_hub defaults",
    )
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow an explicit CPU diagnostic; depth runs are GPU-only by default",
    )
    ap.add_argument("--output-dir", default=None, help="default: ./rpx_results/<model>/<split>/")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Runner batch size + adapter dispatch batch size. "
        "ZoeDepth at 480x640 fp32 fits batch=8 in 8GB VRAM; "
        "drop to 1 if your card OOMs.",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="cap (smoke testing)")
    ap.add_argument(
        "--skip-flops",
        action="store_true",
        help="skip first-sample FLOP instrumentation (recommended for smoke/OOM safety)",
    )
    manifest_group = ap.add_mutually_exclusive_group()
    manifest_group.add_argument(
        "--use-official",
        dest="use_official",
        action="store_true",
        default=True,
        help="use the official pinned HF manifest (default)",
    )
    manifest_group.add_argument(
        "--use-local-cache-manifest",
        dest="use_official",
        action="store_false",
        help="legacy fallback: build a manifest from cached frames_v1.parquet",
    )
    ap.add_argument(
        "--acknowledge-unverified",
        action="store_true",
        help="Deprecated compatibility flag. FE2E now uses the verified "
        "AMAP-ML release and does not require acknowledgement.",
    )
    ap.add_argument(
        "--manifest-path",
        default=None,
        help="Path to a pre-built manifest JSON (e.g. "
        "<staging>/manifests/monocular_depth/<split>.json from "
        "`python -m rpx_benchmark.dataset_hub.cli manifest`). Used to "
        "benchmark against a locally-staged lossless v2-webp tree "
        "before HF upload. Skips both --use-official and the local "
        "Parquet fallback.",
    )
    ap.add_argument(
        "--save-predictions",
        action="store_true",
        help="also save each per-frame .npz prediction (default: off; "
        "metrics only). Opt in for post-hoc analytics.",
    )
    ap.add_argument(
        "--resume-predictions",
        action="store_true",
        help="reuse only validated saved predictions and infer missing/corrupt frames",
    )
    ap.add_argument(
        "--paper-protocol",
        action="store_true",
        help="use the strict six-metric RPX D1-F protocol (640x480 only)",
    )
    ap.add_argument(
        "--defer-fscore",
        action="store_true",
        help="with --paper-protocol, compute the five exact image-space metrics now and "
        "defer F-Score@5cm to evaluate_depth_paper_predictions.py",
    )
    ap.add_argument(
        "--comprehensive-metrics",
        action="store_true",
        help="after the run, compute the full comprehensive metric basket "
        "(SqRel/RMSElog/SIlog/log10/MAE/iRMSE/iMAE + boundary "
        "F-score + ORD + depth-band & in-mask stratification) "
        "and append to result.json. Auto-enables --save-predictions.",
    )
    ap.add_argument(
        "--alignment",
        default="auto",
        choices=["auto", "none", "median", "ls_affine", "ls_disparity", "ls_log"],
        help="alignment mode for comprehensive metrics. 'auto' (default) "
        "reads the adapter's `native_alignment` (metric models → "
        "'none'; diffusion / up-to-scale → 'ls_affine'). Override "
        "explicitly to compare a model under a non-default alignment.",
    )
    ap.add_argument(
        "--upload-to-box",
        action="store_true",
        help="after the run, upload the result dir (and predictions/ "
        "if --save-predictions) to Box under "
        "<box_folder_id>/<task>/<model>/<split>/. Off by default; "
        "team-internal feature. Requires BOX_DEVELOPER_TOKEN.",
    )
    ap.add_argument(
        "--box-folder-id",
        default="380510613151",
        help="Box folder id to root uploads under (default: the team's "
        "RPX-Outputs folder). Override per environment.",
    )
    args = ap.parse_args()
    if args.resume_predictions and not args.save_predictions and not args.comprehensive_metrics:
        ap.error("--resume-predictions requires --save-predictions")
    if args.defer_fscore and not args.paper_protocol:
        ap.error("--defer-fscore requires --paper-protocol")
    if args.max_samples is not None and args.max_samples < 1:
        ap.error("--max-samples must be >= 1")

    from rpx_benchmark.tasks._pipeline import resolve_device

    args.device = resolve_device(args.device, require_cuda=not args.allow_cpu)

    from rpx_benchmark import cli_ux

    cli_ux.setup("run-depth")
    cli_ux.banner(
        "Monocular depth benchmark",
        f"model: {args.model}  ·  split: {args.split}",
    )
    cli_ux.config(
        {
            "model":              args.model,
            "split":              args.split,
            "repo":               args.repo,
            "revision":           args.revision,
            "device":             args.device,
            "precision":          args.precision,
            "cache-dir":          args.cache_dir or "(HF default)",
            "require-cuda":       not args.allow_cpu,
            "skip-flops":         args.skip_flops,
            "batch-size":         args.batch_size,
            "max-samples":        args.max_samples or "(all)",
            "alignment":          args.alignment,
            "save-predictions":   args.save_predictions,
            "resume-predictions": args.resume_predictions,
            "paper-protocol":     args.paper_protocol,
            "defer-fscore":       args.defer_fscore,
            "comprehensive":      args.comprehensive_metrics,
            "use-official":       args.use_official,
            "upload-to-box":      args.upload_to_box,
        }
    )

    with cli_ux.working(f"Loading adapter '{args.model}' (may fetch weights)"):
        model, adapter = _build_model(
            args.model,
            device=args.device,
            batch_size=args.batch_size,
            acknowledge_unverified=args.acknowledge_unverified,
            precision=args.precision,
        )
    if args.model == "da-v2-large" and args.precision == "fp16":
        from rpx_benchmark.exceptions import ConfigError

        expected_checkpoint = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        if getattr(adapter, "model_id", None) != expected_checkpoint:
            raise ConfigError(
                f"DA-V2 paper run expected {expected_checkpoint}; got "
                f"{getattr(adapter, 'model_id', None)}."
            )
        if getattr(adapter, "actual_torch_dtype", None) != "torch.float16":
            raise ConfigError(
                "DA-V2 paper run did not load FP16 parameters; "
                f"got {getattr(adapter, 'actual_torch_dtype', None)}."
            )

    # Comprehensive metrics need predictions on disk
    if args.comprehensive_metrics:
        args.save_predictions = True

    cli_ux.section("Run")
    if args.manifest_path or args.use_official:
        result, dr_report, paths = _run_via_official_pipeline(
            model=model,
            adapter=adapter,
            split=args.split,
            repo_id=args.repo,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            manifest_path=args.manifest_path,
            revision=args.revision,
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            require_cuda=not args.allow_cpu,
            skip_flops=args.skip_flops,
            cache_dir=args.cache_dir,
            resume_predictions=args.resume_predictions,
            paper_protocol=args.paper_protocol,
            defer_fscore=args.defer_fscore,
        )
    else:
        result, dr_report, paths = _run_via_local_manifest(
            model=model,
            adapter=adapter,
            split=args.split,
            repo_id=args.repo,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            revision=args.revision,
            require_cuda=not args.allow_cpu,
            skip_flops=args.skip_flops,
            resume_predictions=args.resume_predictions,
            paper_protocol=args.paper_protocol,
            defer_fscore=args.defer_fscore,
        )

    if args.comprehensive_metrics:
        import json as _json

        from comprehensive_depth_metrics import compute_run
        if args.manifest_path:
            manifest_path = Path(args.manifest_path)
        elif args.use_official:
            from rpx_benchmark.api import TaskType
            from rpx_benchmark.hub import download_split

            manifest_path = download_split(
                task=TaskType.MONOCULAR_DEPTH,
                split=args.split,
                repo_id=args.repo,
                revision=args.revision,
            )
        else:
            from local_manifest import _hf_snapshot_root

            snap = _hf_snapshot_root(args.repo, revision=args.revision)
            manifest_path = (
                snap / "extracted" / "manifests" / "monocular_depth" / f"{args.split}.json"
            )
        pred_dir = paths["predictions_dir"]
        # Resolve 'auto' to the adapter's native alignment.
        chosen_alignment = args.alignment
        if chosen_alignment == "auto":
            chosen_alignment = getattr(adapter, "native_alignment", "none")
        cli_ux.section(f"Comprehensive metrics (alignment={chosen_alignment})")
        with cli_ux.working("computing per-sample errors + CIs + stratifications"):
            manifest_payload = _json.loads(manifest_path.read_text())
            snapshot_root = (
                snap
                if not args.manifest_path and not args.use_official
                else Path(manifest_payload["root"])
            )
            extras = compute_run(
                pred_dir,
                manifest_path,
                alignment=chosen_alignment,
                snapshot_root=snapshot_root,
            )
        out = paths["out_dir"] / "comprehensive_metrics.json"
        out.write_text(_json.dumps(extras, indent=2))
        cli_ux.step(f"wrote {out}  ({len(extras['per_sample'])} samples)")
        for k in sorted(extras["aggregated"]):
            cli_ux.kv(k, f"{extras['aggregated'][k]:.4f}")

    if args.upload_to_box:
        from rpx_benchmark.box_upload import upload_run_dir

        out_dir = paths.get("out_dir")
        if out_dir is None:
            from rpx_benchmark.exceptions import ConfigError

            raise ConfigError(
                "upload requested but no out_dir in artefacts",
                hint="The chosen pipeline path didn't return out_dir. Check "
                "_run_via_official_pipeline / _run_via_local_manifest.",
            )
        name = getattr(model, "name", "model")
        cli_ux.section(f"Mirroring to Box: monocular_depth/{name}/{args.split}")
        with cli_ux.working(f"uploading {out_dir}"):
            upl = upload_run_dir(
                out_dir,
                task="monocular_depth",
                model_name=name,
                split=args.split,
                root_folder_id=args.box_folder_id,
                verbose=False,
            )
        cli_ux.kv("uploaded", f"{upl['uploaded']} files ({cli_ux.fmt_bytes(upl['bytes_uploaded'])})")
        cli_ux.kv("skipped",  f"{upl['skipped']} files (already on Box)")
        cli_ux.kv("remote folder id", upl["remote_folder_id"])

    # Direct benchmark runs carry the same minimum provenance contract as
    # smoke-gate runs. This file is intentionally written after all optional
    # post-processing so its presence means the command reached completion.
    import json as _metadata_json
    import os as _metadata_os
    import subprocess as _metadata_subprocess
    from datetime import datetime as _metadata_datetime
    from datetime import timezone as _metadata_timezone

    from rpx_benchmark.metrics.depth_paper import (
        D1_CALIBRATION,
        FAST_PAPER_METRIC_KEYS,
        PAPER_METRIC_KEYS,
        PREDICTION_EVALUATION_POLICY,
    )

    try:
        _git_sha = _metadata_subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=_metadata_subprocess.DEVNULL,
        ).strip()
    except (OSError, _metadata_subprocess.CalledProcessError):
        _git_sha = _metadata_os.environ.get("RPX_GIT_SHA", "unknown")
    _torch_module = _find_torch_module(adapter)
    _first_parameter = next(_torch_module.parameters(), None) if _torch_module is not None else None
    _metadata = {
        "schema_version": "rpx-depth-run-v1",
        "completed_utc": _metadata_datetime.now(_metadata_timezone.utc).isoformat(),
        "model": args.model,
        "model_checkpoint": getattr(adapter, "model_id", None),
        "split": args.split,
        "num_samples": len(result.per_sample),
        "dataset_repo": args.repo,
        "dataset_revision": args.revision,
        "cache_dir": args.cache_dir,
        "alignment": (
            "none"
            if args.model == "da-v2-large"
            else (
                getattr(adapter, "native_alignment", "none")
                if args.alignment == "auto"
                else args.alignment
            )
        ),
        "requested_precision": args.precision,
        "actual_torch_dtype": getattr(
            adapter,
            "actual_torch_dtype",
            str(_first_parameter.dtype) if _first_parameter is not None else None,
        ),
        "parameter_dtypes": getattr(adapter, "parameter_dtypes", None),
        "batch_size": args.batch_size,
        "metrics": (
            list(FAST_PAPER_METRIC_KEYS if args.defer_fscore else PAPER_METRIC_KEYS)
            if args.paper_protocol
            else sorted(result.aggregated)
        ),
        "calibration": D1_CALIBRATION.to_dict() if args.paper_protocol else None,
        "prediction_evaluation_policy": (
            PREDICTION_EVALUATION_POLICY if args.paper_protocol else None
        ),
        "paper_protocol": args.paper_protocol,
        "fscore_status": (
            "deferred"
            if args.defer_fscore
            else "complete" if args.paper_protocol else "not_applicable"
        ),
        "prediction_resume": paths.get("prediction_stats", {}),
        "git_sha": _git_sha,
        "docker_digest": _metadata_os.environ.get("RPX_DOCKER_DIGEST", "unknown"),
    }
    _metadata_path = paths["out_dir"] / "run_metadata.json"
    _metadata_path.write_text(_metadata_json.dumps(_metadata, indent=2, sort_keys=True) + "\n")
    paths["run_metadata"] = _metadata_path

    cli_ux.summary(
        {k: (f"{v:.4f}" if isinstance(v, float) else v)
         for k, v in (result.aggregated or {}).items()},
        title="Aggregated metrics",
    )
    cli_ux.summary({str(k): str(p) for k, p in paths.items()}, title="Artefacts")


if __name__ == "__main__":
    main()

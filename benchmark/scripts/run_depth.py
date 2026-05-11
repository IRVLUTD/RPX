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

* If the HF repo has ``manifests/<task>/<split>.json`` published, use the
  toolkit's :func:`run_monocular_depth` (download → load → metrics → report).
* If not (the case for ``itaykadosh/rpx-test`` today), fall back to building a
  local manifest from the cached frames Parquet (``local_manifest.py``).

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


def _stage_timing(adapter, dataset, max_samples: int | None = None) -> dict:
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
    samples, gts, data_ms = [], [], []
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
    natively, and `write_json` serialises all of it under
    `payload["deployment_readiness"]`. This helper is now timing-only.
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


def _build_model(name: str, device: str, batch_size: int = 1):
    """Build ``(BenchmarkableModel placeholder, raw_adapter)``.

    The placeholder model is only used for its ``.name`` attribute — the
    real predict path goes through ``BatchedDepthBenchmarkModel`` which
    is constructed inside ``_run_via_local_manifest`` once the runner's
    batch size is known.
    """
    from depth_models import MODEL_DISPLAY_NAMES, MODEL_REGISTRY, list_models

    import rpx_benchmark as rpx

    if name not in MODEL_REGISTRY:
        raise SystemExit(
            f"unknown --model {name!r}. Available: {list_models()}. "
            "See scripts/depth_models/__init__.py to add a new adapter."
        )
    adapter = MODEL_REGISTRY[name](device=device, batch_size=batch_size)
    display_name = MODEL_DISPLAY_NAMES.get(name, name)
    return rpx.make_numpy_depth_model(adapter, name=display_name), adapter


def _run_via_official_pipeline(
    *, model, split: str, repo_id: str, device: str, output_dir: str | None, batch_size: int
):
    """Path A: toolkit's `run_monocular_depth`. Works once the per-task manifests
    are published on the HF repo."""
    from rpx_benchmark import MonocularDepthRunConfig, run_monocular_depth

    cfg = MonocularDepthRunConfig(
        model=model,
        split=split,
        repo_id=repo_id,
        device=device,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    return run_monocular_depth(cfg)


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

    device = resolve_device(device)
    print(f"[local-pipeline] task=monocular_depth split={split} device={device}")

    res = build_local_manifest(
        task="monocular_depth",
        split=split,
        repo_id=repo_id,
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
    # is computed inside `runner.run_with_deployment_readiness` and
    # serialised under `result.json["deployment_readiness"]`.
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
        metric_suite=MetricSuite.for_task(TaskType.MONOCULAR_DEPTH),
        call_setup=False,
    )
    bench_result, dr_report = runner.run_with_deployment_readiness(
        primary_metric="absrel",
        model_name=name,
        efficiency=eff,
        compute_ts=True,
        compute_sgc_flag=False,
    )
    # ↑ At this point dr_report carries the full Tier 1/2/3 efficiency
    # picture (params, flops, macs, mem-traffic, arithmetic intensity,
    # roofline bounds for A100/4090/Orin, measured latency, peak memory,
    # system card). `write_json` serialises all of it under
    # `result.json["deployment_readiness"]` — no post-hoc augmentation
    # of efficiency fields needed.

    # ---- per-stage timing breakdown (data load / model run / metric calc) ---
    # The runner's `latency_ms` is end-to-end; this gives us where the time
    # actually goes so we can compare adapters fairly. Re-runs the dataset
    # without the deployment-readiness wrapper to keep the numbers clean.
    timing = _stage_timing(adapter, dataset)
    print(
        "[timing] data_load_ms (mean/median/p95): "
        f"{timing['data_load']['mean']:.1f}/"
        f"{timing['data_load']['median']:.1f}/"
        f"{timing['data_load']['p95']:.1f}"
    )
    print(
        "[timing] model_run_ms (mean/median/p95): "
        f"{timing['model_run']['mean']:.1f}/"
        f"{timing['model_run']['median']:.1f}/"
        f"{timing['model_run']['p95']:.1f}"
    )
    print(
        "[timing] metric_calc_ms (mean/median/p95): "
        f"{timing['metric_calc']['mean']:.1f}/"
        f"{timing['metric_calc']['median']:.1f}/"
        f"{timing['metric_calc']['p95']:.1f}"
    )

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

    # Per-stage timing isn't part of the dr_report, so it's added here.
    _augment_result_with_timing(json_path, timing=timing)

    artefacts = {"json": json_path, "markdown": md_path, "out_dir": out_dir}
    if save_predictions:
        artefacts["predictions_dir"] = pred_dir
    return bench_result, dr_report, artefacts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 2)[0])
    ap.add_argument("--model", default="zoedepth", help="adapter to use")
    ap.add_argument("--split", default="easy", help="easy | medium | hard")
    ap.add_argument("--repo", default="itaykadosh/rpx-test", help="HuggingFace dataset repo")
    ap.add_argument("--device", default="cuda")
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
        "--use-official",
        action="store_true",
        help="use download_split (requires manifests/ on HF). "
        "Default: build manifest locally from cached Parquet.",
    )
    ap.add_argument(
        "--save-predictions",
        action="store_true",
        help="also save each per-frame .npz prediction (default: off; "
        "metrics only). Opt in for post-hoc analytics.",
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
        choices=["auto", "none", "median", "ls_affine", "ls_disparity"],
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

    model, adapter = _build_model(args.model, device=args.device, batch_size=args.batch_size)

    # Comprehensive metrics need predictions on disk
    if args.comprehensive_metrics:
        args.save_predictions = True

    if args.use_official:
        result, dr_report, paths = _run_via_official_pipeline(
            model=model,
            split=args.split,
            repo_id=args.repo,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
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
        )

    if args.comprehensive_metrics:
        from comprehensive_depth_metrics import compute_run
        from local_manifest import _hf_snapshot_root

        # Find the manifest we just used (build_local_manifest writes a known path)
        snap = _hf_snapshot_root(args.repo)
        manifest_path = snap / "extracted" / "manifests" / "monocular_depth" / f"{args.split}.json"
        pred_dir = paths["predictions_dir"]
        # Resolve 'auto' to the adapter's native alignment.
        chosen_alignment = args.alignment
        if chosen_alignment == "auto":
            chosen_alignment = getattr(adapter, "native_alignment", "none")
        print(f"\n=== comprehensive metrics (alignment={chosen_alignment}) ===")
        extras = compute_run(pred_dir, manifest_path, alignment=chosen_alignment)
        out = paths["out_dir"] / "comprehensive_metrics.json"
        import json as _json

        out.write_text(_json.dumps(extras, indent=2))
        print(f"  wrote {out}  ({len(extras['per_sample'])} samples)")
        for k in sorted(extras["aggregated"]):
            print(f"    {k:>32}: {extras['aggregated'][k]:.4f}")

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
        # Box layout: <root>/monocular_depth/<model>/<split>/
        name = getattr(model, "name", "model")
        print(f"\n=== uploading {out_dir} → Box:monocular_depth/{name}/{args.split} ===")
        summary = upload_run_dir(
            out_dir,
            task="monocular_depth",
            model_name=name,
            split=args.split,
            root_folder_id=args.box_folder_id,
        )
        print(
            f"  uploaded: {summary['uploaded']} files ({_human_bytes(summary['bytes_uploaded'])})"
        )
        print(f"  skipped:  {summary['skipped']} files (already on Box)")
        print(f"  remote folder id: {summary['remote_folder_id']}")

    print()
    print("=== aggregated metrics ===")
    for k, v in (result.aggregated or {}).items():
        if isinstance(v, float):
            print(f"  {k:>22}: {v:.4f}")
        else:
            print(f"  {k:>22}: {v}")
    print()
    print("=== artefacts ===")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()

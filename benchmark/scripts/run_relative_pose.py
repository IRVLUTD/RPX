"""Run a Relative Camera Pose model end-to-end against an RPX paired split.

Sister of ``run_depth.py``. Wires the registered pose adapters in
``scripts/pose_models/`` to the toolkit's loader + runner + Box upload
pipeline. Per-pair predictions land in a single CSV
(``predictions.csv``); aggregated metrics + DRS OperatingPoint go in
``result.json``; full pose-error basket with 95% CIs (rotation,
translation L2 + angular, AUC@5°/10°/20°, per-stride breakdown) lands
in ``pose_comprehensive_metrics.json``.

Usage
-----
    PYTHONPATH=. python scripts/run_relative_pose.py --model loftr --split easy
    PYTHONPATH=. python scripts/run_relative_pose.py --model reloc3r --split easy \
        --save-predictions --comprehensive-metrics --upload-to-box
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _human_bytes(n: int | float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}" if isinstance(n, float) else f"{n} {u}"
        n /= 1024
    return f"{n:.1f} PB"


# Make ./scripts importable so `from pose_models import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _build_model(name: str, device: str, batch_size: int = 1):
    """Resolve a registry name → (BenchmarkableModel placeholder, raw adapter)."""
    from pose_models import MODEL_DISPLAY_NAMES, MODEL_REGISTRY, list_models

    import rpx_benchmark as rpx

    if name not in MODEL_REGISTRY:
        from rpx_benchmark.exceptions import ConfigError

        raise ConfigError(
            f"unknown --model {name!r}",
            hint=f"Available: {list_models()}. See scripts/pose_models/__init__.py.",
        )
    adapter = MODEL_REGISTRY[name](device=device, batch_size=batch_size)
    display_name = MODEL_DISPLAY_NAMES.get(name, name)
    # The placeholder BenchmarkableModel is just for the .name attribute.
    placeholder = rpx.make_numpy_pose_model(adapter, name=display_name)
    return placeholder, adapter


def _run_via_local_manifest(
    *,
    adapter,
    name: str,
    split: str,
    repo_id: str,
    device: str,
    output_dir: str | None,
    batch_size: int,
    max_samples: int | None,
    save_predictions: bool,
):
    from local_manifest import build_local_manifest

    from rpx_benchmark.adapters import BatchedRelativePoseBenchmarkModel
    from rpx_benchmark.api import TaskType
    from rpx_benchmark.evaluators import MetricSuite
    from rpx_benchmark.loader import RPXDataset
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
    print(f"[pose-pipeline] task=relative_pose split={split} device={device}")

    # Build a paired manifest. local_manifest currently routes through the
    # same _RelativePoseSpec writer the dataset_hub uses, so the entry
    # shape (rgb, rgb_b, pose_a, pose_b, metadata) is loader-correct.
    res = build_local_manifest(
        task="relative_pose",
        split=split,
        repo_id=repo_id,
        max_samples=max_samples,
    )
    print(f"[pose-pipeline] manifest: {res.manifest_path}  ({res.n_samples} pairs)")

    out_dir = Path(output_dir or f"./rpx_results/{name}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir if save_predictions else None  # CSV lands at out_dir/predictions.csv

    model = BatchedRelativePoseBenchmarkModel(
        adapter,
        name=name,
        save_dir=pred_dir,
    )
    dataset = RPXDataset.from_manifest(res.manifest_path, batch_size=batch_size)
    print(
        f"[pose-pipeline] batch_size={batch_size}  predictions_csv="
        f"{(out_dir / 'predictions.csv') if save_predictions else 'n/a'}"
    )

    model.setup()

    # Lightweight efficiency profiling (analogous to run_depth.py).
    torch_mod = _find_torch_module(adapter)
    eff = EfficiencyMetadata(
        model_type="local",
        notes=f"{name} @ pose pair",
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
        print(f"[profiler] system_card failed: {type(e).__name__}: {e}")

    runner = BenchmarkRunner(
        model=model,
        dataset=dataset,
        metric_suite=MetricSuite.for_task(TaskType.RELATIVE_CAMERA_POSE),
        call_setup=False,
    )
    bench_result, dr_report = runner.run_with_deployment_readiness(
        primary_metric="rotation_error_deg",
        model_name=name,
        efficiency=eff,
        compute_ts=False,
        compute_sgc_flag=False,
    )

    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    write_json(
        json_path,
        task="relative_pose",
        model_name=name,
        split=split,
        repo_id=repo_id,
        result=bench_result,
        dr_report=dr_report,
    )
    md_path.write_text(
        format_markdown_summary(
            task="relative_pose",
            model_name=name,
            split=split,
            repo_id=repo_id,
            result=bench_result,
            dr_report=dr_report,
        ),
        encoding="utf-8",
    )

    artefacts: dict = {"json": json_path, "markdown": md_path, "out_dir": out_dir}
    if save_predictions:
        artefacts["predictions_csv"] = out_dir / "predictions.csv"
    return bench_result, dr_report, artefacts


def _find_torch_module(adapter):
    for path in ("torch_module", "_model", "_pipe.model", "model"):
        cur = adapter
        for part in path.split("."):
            cur = getattr(cur, part, None) if cur is not None else None
            if cur is None:
                break
        if cur is not None and hasattr(cur, "parameters"):
            return cur
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="opencv_baseline", help="adapter to use")
    ap.add_argument("--split", default="easy", help="easy | medium | hard")
    ap.add_argument("--repo", default="itaykadosh/rpx-test", help="HuggingFace dataset repo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default=None, help="default: ./rpx_results/<model>/<split>/")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Pose adapters mostly process one pair at a time today; "
        "raise to N if your adapter supports vectorised inference.",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="cap (smoke testing)")
    ap.add_argument(
        "--save-predictions",
        action="store_true",
        help="write per-pair predictions to predictions.csv "
        "(scene, phase, frame_a, frame_b, R[9], t[3]).",
    )
    ap.add_argument(
        "--comprehensive-metrics",
        action="store_true",
        help="after the run, compute the full pose basket "
        "(rotation_error_deg, translation_l2, "
        "translation_angular_deg, AUC@5°/10°/20°, "
        "per-stride breakdown). Auto-enables --save-predictions.",
    )
    ap.add_argument(
        "--upload-to-box",
        action="store_true",
        help="ship the result dir to UTD Box under "
        "<box_folder_id>/relative_pose/<model>/<split>/. "
        "Requires BOX_DEVELOPER_TOKEN.",
    )
    ap.add_argument(
        "--box-folder-id",
        default="380510613151",
        help="Box folder id (default: team's RPX-Outputs).",
    )
    args = ap.parse_args()

    placeholder, adapter = _build_model(args.model, device=args.device, batch_size=args.batch_size)
    name = placeholder.name

    if args.comprehensive_metrics:
        args.save_predictions = True

    result, dr_report, paths = _run_via_local_manifest(
        adapter=adapter,
        name=name,
        split=args.split,
        repo_id=args.repo,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_predictions=args.save_predictions,
    )

    if args.comprehensive_metrics:
        from local_manifest import _hf_snapshot_root
        from pose_comprehensive_metrics import compute_run

        snap = _hf_snapshot_root(args.repo)
        manifest_path = snap / "extracted" / "manifests" / "relative_pose" / f"{args.split}.json"
        csv_path = paths["predictions_csv"]
        print(f"\n=== comprehensive pose metrics ===")
        extras = compute_run(csv_path, manifest_path)
        out = paths["out_dir"] / "pose_comprehensive_metrics.json"
        import json as _json

        out.write_text(_json.dumps(extras, indent=2, default=str))
        print(f"  wrote {out}  ({len(extras['per_pair'])} pairs)")
        for k in sorted(extras.get("aggregated") or {}):
            print(f"  {k:>32}: {extras['aggregated'][k]:.4f}")

    if args.upload_to_box:
        from box_fetch import upload_tree

        out_dir = paths.get("out_dir")
        if out_dir is None:
            from rpx_benchmark.exceptions import ConfigError

            raise ConfigError("upload requested but no out_dir in artefacts")
        remote = f"relative_pose/{name}/{args.split}"
        print(f"\n=== uploading {out_dir} → Box:{remote} ===")
        summary = upload_tree(Path(out_dir), remote_path=remote, root_folder_id=args.box_folder_id)
        print(
            f"  uploaded: {summary['uploaded']} files ({_human_bytes(summary['bytes_uploaded'])})"
        )
        print(f"  skipped:  {summary['skipped']} files (already on Box)")
        print(f"  remote folder id: {summary['remote_folder_id']}")

    print()
    print("=== aggregated metrics ===")
    for k, v in (result.aggregated or {}).items():
        if isinstance(v, float):
            print(f"  {k:>26}: {v:.4f}")
        else:
            print(f"  {k:>26}: {v}")
    print()
    print("=== artefacts ===")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

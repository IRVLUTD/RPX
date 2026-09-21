"""Run a Relative Camera Pose model end-to-end against an RPX paired split.

Sister of ``run_depth.py``. Wires the registered pose adapters in
``scripts/pose_models/`` to the toolkit's loader + runner + Box upload
pipeline. Per-pair predictions land in a single CSV
(``predictions.csv``); aggregated metrics + DRS OperatingPoint go in
``result.json``; full pose-error basket with 95% CIs (rotation,
translation L2 + angular, AUC@5°/10°/20°, per-stride breakdown) lands
in ``pose_comprehensive_metrics.json``.

Pair sources
------------
- ``--pairs-source manifest`` (default): use a pre-saved JSON manifest
  from ``--pairs-manifest`` or the canonical HF path.
- ``--pairs-source on_the_fly``: use ``PosePairGenerator`` for
  deterministic on-the-fly pair generation with stratified rotation
  bins, cross-phase pairs, and temporal chains.  No manifest file
  needed — pairs are regenerated from seed every time.

Usage
-----
    PYTHONPATH=. python scripts/run_relative_pose.py --model loftr --split easy

    # On-the-fly stratified pairs (novel RPX-RCPE protocol):
    PYTHONPATH=. python scripts/run_relative_pose.py --model reloc3r --split easy \\
        --pairs-source on_the_fly --save-predictions --comprehensive-metrics
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


def _stratified_cap(samples: list[dict], limit: int) -> list[dict]:
    """Deterministically cap a gate run while retaining protocol coverage."""
    if limit >= len(samples):
        return samples
    priorities = [
        ("intra_phase", 0),
        ("intra_phase", 2),
        ("temporal_chain", 0),
        ("temporal_chain", 2),
    ]
    selected = []
    selected_ids = set()
    for pair_type, phase in priorities:
        match = next(
            (
                sample
                for sample in samples
                if sample.get("pair_type") == pair_type
                and sample.get("phase") == phase
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_ids.add(match["id"])
            if len(selected) == limit:
                return selected
    buckets: dict[tuple, list[dict]] = {}
    for sample in samples:
        if sample["id"] in selected_ids:
            continue
        key = (
            sample.get("pair_type"),
            sample.get("phase"),
            sample.get("rotation_bin"),
        )
        buckets.setdefault(key, []).append(sample)
    bucket_values = list(buckets.values())
    offset = 0
    while len(selected) < limit:
        added = False
        for bucket in bucket_values:
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


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
    pairs_manifest: Path | None = None,
    revision: str | None = None,
):
    import json as _json

    from local_manifest import _hf_snapshot_root

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

    # The canonical manifest shipped from HF (`manifests/relative_pose/<split>.json`)
    # already has the paired structure (`rgb`, `rgb_b`, `pose_a`, `pose_b`,
    # `metadata`) + the correct TaskType name (`relative_camera_pose`),
    # produced by the dataset_hub `_RelativePoseSpec` writer. Use it directly:
    # paths inside are relative to <snap>/, so we pass `root=<snap>` to
    # `RPXDataset.from_dict`.
    snap = _hf_snapshot_root(repo_id, revision=revision)
    if pairs_manifest is not None:
        manifest_path = Path(pairs_manifest)
        if not manifest_path.is_file():
            from rpx_benchmark.exceptions import DatasetError

            raise DatasetError(
                f"--pairs-manifest path not found: {manifest_path}",
                hint="Pass the absolute path to a pose manifest produced by "
                "scripts/generate_pose_pairs.py (or any file matching the "
                "canonical `relative_camera_pose` schema).",
            )
    else:
        manifest_path = snap / "manifests" / "relative_pose" / f"{split}.json"
        if not manifest_path.is_file():
            from rpx_benchmark.exceptions import DatasetError

            raise DatasetError(
                f"missing canonical pose manifest at {manifest_path}",
                hint=f"Run `rpx.load('relative_pose', '{split}')` first to "
                "populate the HF snapshot, or run "
                "`python -m rpx_benchmark.dataset_hub.cli manifest --tasks relative_pose`.",
            )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = _json.load(f)
    if max_samples is not None:
        manifest["samples"] = manifest["samples"][:max_samples]
    manifest["root"] = str(snap)
    sampler = manifest.get("_sampler", {}).get("name", "stride")
    print(
        f"[pose-pipeline] manifest: {manifest_path}  "
        f"({len(manifest['samples'])} pairs, sampler={sampler}"
        f"{f', capped at --max-samples={max_samples}' if max_samples else ''})"
    )

    out_dir = Path(output_dir or f"./rpx_results/{name}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir if save_predictions else None  # CSV lands at out_dir/predictions.csv

    model = BatchedRelativePoseBenchmarkModel(
        adapter,
        name=name,
        save_dir=pred_dir,
    )
    dataset = RPXDataset.from_dict(manifest, batch_size=batch_size)
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
    bench_result, dr_report = runner.run_with_report(
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


def _run_on_the_fly(
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
    intra_pairs_per_bin: int = 50,
    cross_pairs_per_bin: int = 0,
    skip_flops: bool = False,
    revision: str | None = None,
):
    """Run using PosePairGenerator — deterministic on-the-fly pairs."""
    import json as _json

    from local_manifest import _hf_snapshot_root

    from rpx_benchmark.adapters import BatchedRelativePoseBenchmarkModel
    from rpx_benchmark.api import TaskType
    from rpx_benchmark.cell_log import write_cells
    from rpx_benchmark.evaluators import MetricSuite
    from rpx_benchmark.phi_jedi_summary import summarize_phi_jedi
    from rpx_benchmark.pose_metrics import (
        CANONICAL_POSE_METRICS,
        build_rcpe_cells,
        evaluate_rcpe,
    )
    from rpx_benchmark.pose_pairs import PairConfig, PosePairGenerator
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
    snap = _hf_snapshot_root(repo_id, revision=revision)

    cfg = PairConfig(
        intra_pairs_per_bin=intra_pairs_per_bin,
        cross_pairs_per_bin=0,
    )
    gen = PosePairGenerator(
        extracted_root=snap / "extracted",
        parquet_path=snap / "manifest" / "frames_v1.parquet",
        split=split,
        config=cfg,
        snapshot_root=snap,
        repo_id=repo_id,
    )
    print(gen.summary())

    manifest = gen.manifest()
    if max_samples is not None:
        manifest["samples"] = _stratified_cap(manifest["samples"], max_samples)
    print(
        f"[pose-pipeline] on_the_fly: {len(manifest['samples'])} pairs, "
        f"device={device}, batch_size={batch_size}"
    )

    out_dir = Path(output_dir or f"./rpx_results/{name}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_manifest_path = out_dir / "pairs_manifest.json"
    pairs_manifest_path.write_text(
        _json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pred_dir = out_dir if save_predictions else None

    model = BatchedRelativePoseBenchmarkModel(
        adapter, name=name, save_dir=pred_dir,
    )
    from rpx_benchmark.loader import RPXDataset

    gen.ensure_pairs_extracted(manifest["samples"])
    dataset = RPXDataset.from_dict(manifest, batch_size=batch_size)

    model.setup()

    # Efficiency profiling
    torch_mod = _find_torch_module(adapter)
    eff = EfficiencyMetadata(model_type="local", notes=f"{name} @ pose pair")
    if torch_mod is not None:
        try:
            eff.params_m = count_parameters(torch_mod)
        except Exception as e:  # noqa: BLE001
            print(f"[profiler] params count failed: {type(e).__name__}: {e}")
        try:
            eff.memory_traffic_gb = estimate_memory_traffic_gb(
                torch_mod, (3, 480, 640), device=device,
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
    bench_result, dr_report = runner.run_with_report(
        primary_metric="rotation_error_deg",
        model_name=name,
        efficiency=eff,
        compute_ts=False,
        compute_sgc_flag=False,
        skip_flops=skip_flops,
    )

    metric_translation_available = (
        getattr(adapter, "native_alignment", "none") == "none"
    )
    if not metric_translation_available:
        # Up-to-scale models have no meaningful metric translation error.
        # Keep their scale-invariant translation-angle metric, but do not emit
        # a misleading metre value in the standard result/summary.
        bench_result.aggregated.pop("translation_error_m", None)
        for row in bench_result.per_sample:
            row.pop("translation_error_m", None)

    # The generic deployment report assumes all three RPX phases. RCPE is
    # intentionally phase 0/2 only, so its three-phase WPS and STR would insert
    # a fictitious zero-valued phase 1. Φ/JEDI below is the valid phase report.
    dr_report.weighted_phase_score = None
    dr_report.state_transition = None

    # ── Standard output ───────────────────────────────────────────────
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    write_json(
        json_path, task="relative_pose", model_name=name,
        split=split, repo_id=repo_id,
        result=bench_result, dr_report=dr_report,
    )
    md_path.write_text(
        format_markdown_summary(
            task="relative_pose", model_name=name,
            split=split, repo_id=repo_id,
            result=bench_result, dr_report=dr_report,
        ),
        encoding="utf-8",
    )

    # ── Novel RPX-RCPE metrics ────────────────────────────────────────
    if save_predictions:
        from pose_comprehensive_metrics import compute_run

        comprehensive = compute_run(
            out_dir / "predictions.csv",
            pairs_manifest_path,
            snapshot_root=snap,
            metric_translation_available=metric_translation_available,
        )
        per_pair_enriched = comprehensive["per_pair"]
        comprehensive_path = out_dir / "pose_comprehensive_metrics.json"
        comprehensive_path.write_text(
            _json.dumps(comprehensive, indent=2, default=str), encoding="utf-8"
        )
    else:
        id_to_meta = {s["id"]: s.get("metadata", {}) for s in manifest["samples"]}
        per_pair_enriched = []
        for row in bench_result.per_sample:
            enriched = dict(row)
            meta = id_to_meta.get(row.get("id", ""), {})
            enriched.update(meta)
            enriched["metadata"] = meta
            per_pair_enriched.append(enriched)

    rcpe_results = evaluate_rcpe(
        per_pair_enriched,
        metric_translation_available=metric_translation_available,
    )

    # The active Φ/JEDI input is one locked K=3 angular row per
    # (scene, phase), computed only from exact-gap intra-phase pairs. Metric
    # M-AUC is skipped; temporal-chain samples remain diagnostic.
    cells, analysis_rows = build_rcpe_cells(
        per_pair_enriched,
        model_name=name,
        difficulty=split,
        metric_translation_available=metric_translation_available,
    )
    cells_result = write_cells(cells, out_dir / "cells.parquet")
    phase_coverage = sorted({row["phase"] for row in analysis_rows})
    if phase_coverage != ["clean", "clutter"]:
        phi_jedi = {
            "status": "insufficient_gate_coverage",
            "reason": "phases 0 (clutter) and 2 (clean) are required for D6 Phi/JEDI",
            "phase_coverage": phase_coverage,
            "metrics": list(CANONICAL_POSE_METRICS),
        }
    else:
        summary = summarize_phi_jedi(
            analysis_rows,
            metric_keys=CANONICAL_POSE_METRICS,
        )
        phi_jedi = {"status": "computed", **summary.to_dict()}
    phi_jedi_path = out_dir / "phi_jedi.json"
    phi_jedi_path.write_text(
        _json.dumps(phi_jedi, indent=2, default=str), encoding="utf-8"
    )

    # Attach efficiency summary — structured by tier so consumers know
    # which numbers are hardware-agnostic and which are not.
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

    rcpe_results["efficiency"] = {
        # Tier 1: hardware-agnostic (identical on any machine)
        "tier1_hardware_agnostic": {
            "params_m": eff.params_m,
            "flops_g": eff.flops_g,
            "macs_g": eff.macs_g,
            "memory_traffic_gb": eff.memory_traffic_gb,
            "arithmetic_intensity": eff.arithmetic_intensity,
        },
        # Tier 2: hardware-parametric (reproducible from Tier 1 + GPU spec)
        "tier2_roofline": roofline_dict,
        # Tier 3: measured (hardware-specific — interpret with system_card)
        "tier3_measured": {
            "latency_p50_ms": eff.latency_p50_ms,
            "latency_p95_ms": eff.latency_p95_ms,
            "latency_p99_ms": eff.latency_p99_ms,
            "peak_cpu_mb": eff.peak_cpu_mb,
            "peak_cuda_mb": eff.peak_cuda_mb,
            "peak_mps_mb": eff.peak_mps_mb,
            "system_card": eff.system_card.to_dict() if eff.system_card else None,
        },
    }

    rcpe_path = out_dir / "rcpe_metrics.json"
    rcpe_path.write_text(
        _json.dumps(rcpe_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[pose-pipeline] wrote {rcpe_path}")

    # Print novel metrics summary
    print(f"\n{'='*60}")
    print(f"  RPX-RCPE Novel Metrics — {name} / {split}")
    print(f"{'='*60}")
    agg = rcpe_results.get("aggregated", {})
    for k, v in agg.items():
        rendered = "unavailable (up-to-scale)" if v is None else f"{v:.4f}"
        print(f"  {k:>35}: {rendered}")
    print()
    sauc = rcpe_results.get("standard_auc", {})
    for k, v in sauc.items():
        print(f"  {k:>35}: {v:.4f}")
    print()
    print("  Active D6 K=3 (M-AUC skipped):")
    for k, v in rcpe_results.get("canonical", {}).items():
        rendered = "unavailable (up-to-scale)" if v is None else f"{v:.4f}"
        print(f"  {k:>35}: {rendered}")
    print()
    print("  Per rotation bin:")
    for b, m in rcpe_results.get("per_bin", {}).items():
        n = m.get('n_pairs', 0)
        re = m.get('rotation_error_deg', 0)
        te = m.get('translation_error_m')
        trans = "n/a (up-to-scale)" if te is None else f"{te*100:.1f}cm"
        print(f"    {b:>8}: n={n:.0f}  rot={re:.2f}°  trans={trans}")
    print()
    print("  Per pair type:")
    for t, m in rcpe_results.get("per_type", {}).items():
        n = m.get('n_pairs', 0)
        re = m.get('rotation_error_deg', 0)
        te = m.get('translation_error_m')
        trans = "n/a (up-to-scale)" if te is None else f"{te*100:.1f}cm"
        print(f"    {t:>15}: n={n:.0f}  rot={re:.2f}°  trans={trans}")
    print()
    cpd = rcpe_results.get("cross_phase_delta", {})
    if cpd:
        print("  Cross-phase Δ (positive = harder):")
        for k, v in cpd.items():
            print(f"    {k:>20}: {v:+.4f}")
    print()
    drift = rcpe_results.get("temporal_drift", {})
    if drift.get("n_chains"):
        print(f"  Temporal drift ({drift['n_chains']} chains):")
        print(f"    mean rot drift:   {drift['mean_drift_rot_deg']:.1f}°")
        trans_drift = drift.get("mean_drift_trans_m")
        if trans_drift is not None:
            print(f"    mean trans drift: {trans_drift*100:.1f} cm")
        else:
            print("    mean trans drift: n/a (up-to-scale)")

    artefacts: dict = {
        "json": json_path, "markdown": md_path,
        "rcpe_metrics": rcpe_path, "out_dir": out_dir,
        "pairs_manifest": pairs_manifest_path,
        "cells": cells_result.path,
        "phi_jedi": phi_jedi_path,
    }
    if save_predictions:
        artefacts["predictions_csv"] = out_dir / "predictions.csv"
        artefacts["pose_comprehensive_metrics"] = comprehensive_path

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
    from rpx_benchmark.cleanup import install_signal_cleanup
    install_signal_cleanup()

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="opencv_baseline", help="adapter to use")
    ap.add_argument("--split", default="easy", help="easy | medium | hard")
    ap.add_argument("--repo", default="anonymous/RPX", help="HuggingFace dataset repo")
    ap.add_argument(
        "--revision",
        default=None,
        help="exact cached Hugging Face dataset revision to use",
    )
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
        "--pairs-source",
        choices=("manifest", "on_the_fly"),
        default="manifest",
        help="'manifest' (default): load from --pairs-manifest or canonical HF path. "
        "'on_the_fly': use PosePairGenerator for deterministic stratified pairs "
        "(exact-gap intra-phase pairs + temporal chains).",
    )
    ap.add_argument(
        "--pairs-manifest",
        type=Path,
        default=None,
        help="override the canonical pair list (only used with --pairs-source manifest).",
    )
    ap.add_argument(
        "--intra-pairs-per-bin", type=int, default=50,
        help="intra-phase pairs per rotation bin per (scene, phase) "
        "(only with --pairs-source on_the_fly)",
    )
    ap.add_argument(
        "--cross-pairs-per-bin", type=int, default=0,
        help="deprecated; must remain 0 because captures have unrelated T265 worlds",
    )
    ap.add_argument(
        "--skip-flops",
        action="store_true",
        help="skip the FlopCounterMode on the first batch. Use for large "
        "models (MASt3R, DUSt3R) where the FLOP counter OOMs.",
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

    if args.cross_pairs_per_bin != 0:
        ap.error(
            "--cross-pairs-per-bin must be 0: phases 0 and 2 are separate "
            "captures with unrelated T265 local world frames"
        )

    placeholder, adapter = _build_model(args.model, device=args.device, batch_size=args.batch_size)
    name = placeholder.name

    if args.comprehensive_metrics:
        args.save_predictions = True

    if args.pairs_source == "on_the_fly":
        result, dr_report, paths = _run_on_the_fly(
            adapter=adapter,
            name=name,
            split=args.split,
            repo_id=args.repo,
            device=args.device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            save_predictions=args.save_predictions,
            intra_pairs_per_bin=args.intra_pairs_per_bin,
            cross_pairs_per_bin=args.cross_pairs_per_bin,
            skip_flops=args.skip_flops,
            revision=args.revision,
        )
    else:
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
            pairs_manifest=args.pairs_manifest,
            revision=args.revision,
        )

    if args.comprehensive_metrics:
        from local_manifest import _hf_snapshot_root
        from pose_comprehensive_metrics import compute_run

        snap = _hf_snapshot_root(args.repo, revision=args.revision)
        if args.pairs_source == "on_the_fly":
            # The generated manifest is the exact evaluated sample set.  The
            # canonical HF manifest is neither required nor necessarily
            # published for on-the-fly RCPE runs.
            manifest_path = paths["pairs_manifest"]
        elif args.pairs_manifest is not None:
            manifest_path = Path(args.pairs_manifest)
        else:
            manifest_path = snap / "manifests" / "relative_pose" / f"{args.split}.json"
        csv_path = paths["predictions_csv"]
        print(f"\n=== comprehensive pose metrics ===")
        extras = compute_run(csv_path, manifest_path, snapshot_root=snap)
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

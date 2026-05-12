"""Run a Novel View Synthesis model end-to-end against an RPX split.

Mirrors the structure of ``run_relative_pose.py`` for the NVS axis. Wires
the registered NVS adapters in ``scripts/nvs_models/`` to the
``NVSPairGenerator`` (on-the-fly stratified sample generator) and the
``evaluate_nvs`` aggregator, then writes a three-axis ``result.json``
into ``./rpx_results/<DisplayName>/<split>/``.

Per the RPX policy (see ``benchmark/SHARED_CONTEXT.md``), the output
reports each model on **three independent axes** — task performance,
scene-change robustness, compute cost — and never combines them into a
single composite score.

Output schema
-------------
``rpx_results/<display>/<split>/``:

* ``result.json`` — full three-axis report.
* ``summary.md``  — human-readable.
* ``predictions/<scene>/<phase>/<frame>.npz`` — rendered RGB + depth per
  sample, when ``--save-predictions`` is passed.

Usage
-----
::

    # Smoke run with the identity baseline (zero deps):
    PYTHONPATH=. python scripts/run_nvs.py \\
        --model identity_passthrough --split easy --max-samples 20

    # Real NVS adapter (raises AdapterError until the upstream is wired):
    PYTHONPATH=. python scripts/run_nvs.py --model depthsplat --split easy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ./scripts importable so `from nvs_models import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────────────────────────────────────────────
# Modality loaders — keep them minimal and dependency-light
# ─────────────────────────────────────────────────────────────────────────────


def _load_rgb(path: Path) -> "Any":
    """Load an RGB image as (H, W, 3) uint8."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_depth(path: Path) -> "Any":
    """Load a 16-bit depth PNG as (H, W) float32 in metres (D435 mm → m)."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    arr = np.array(Image.open(path))
    return (arr.astype(np.float32) / 1000.0) if arr.dtype == np.uint16 else arr.astype(np.float32)


def _load_pose(path: Path) -> "Any":
    """Load a T265 pose ``.npz`` → 4×4 SE(3) camera-to-world (float64).

    The RPX NPZ schema (see ``loader._load_pose``) stores **two arrays**,
    not a baked 4×4:

    * ``position``    — ``(3,)`` metres
    * ``orientation`` — ``(4,)`` quaternion in T265 ``[x, y, z, w]`` order

    We reconstruct the 4×4 here so the runner's pose semantics match the
    toolkit's canonical loader exactly. Mirrors the math in
    ``rpx_benchmark/loader.py:_quat_xyzw_to_rotmat`` / ``_load_pose``.
    """
    import numpy as np  # noqa: PLC0415

    data = np.load(path)
    position  = np.asarray(data["position"], dtype=np.float64)
    quat_xyzw = np.asarray(data["orientation"], dtype=np.float64)

    x, y, z, w = quat_xyzw / np.linalg.norm(quat_xyzw)
    rot = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3]  = position
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Model resolution
# ─────────────────────────────────────────────────────────────────────────────


def _build_model(name: str, device: str) -> Any:
    """Resolve a registry name → adapter callable. Raises AdapterError on
    unknown names; the registry's own builders raise AdapterError when
    their upstream isn't installed yet."""
    from nvs_models import MODEL_REGISTRY, list_models  # noqa: PLC0415

    if name not in MODEL_REGISTRY:
        from rpx_benchmark.exceptions import AdapterError  # noqa: PLC0415

        raise AdapterError(
            f"unknown NVS model: {name!r}",
            hint=f"registered models: {', '.join(list_models())}",
        )
    return MODEL_REGISTRY[name](device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample rendering + metric computation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _SampleResult:
    metrics: Dict[str, Any]                # per-sample metrics for evaluate_nvs
    pred_rgb: "Any"                        # (H, W, 3) uint8
    pred_depth: "Optional[Any]"            # (H, W) float32 or None
    latency_ms: float                      # adapter wall-clock per sample
    sample_id: str
    scene_id: str
    phase: int


def _evaluate_sample(
    sample: Any,
    rendered: Dict[str, Any],
    gt_rgb: "Any",
    gt_depth: "Optional[Any]",
) -> Dict[str, Any]:
    """Compute the per-sample metric dict consumed by ``evaluate_nvs``."""
    from rpx_benchmark.nvs_metrics import depth_metrics, psnr, ssim  # noqa: PLC0415

    pred_rgb = rendered.get("rgb")
    pred_depth = rendered.get("depth")

    row: Dict[str, Any] = {
        "sample_id":   sample.id,
        "scene_id":    sample.scene_id,
        "phase":       sample.phase,
        "n_context":   sample.n_context,
        "sample_type": sample.sample_type,
    }

    if pred_rgb is not None and gt_rgb is not None:
        row["psnr"] = psnr(pred_rgb, gt_rgb)
        row["ssim"] = ssim(pred_rgb, gt_rgb)

    if pred_depth is not None and gt_depth is not None:
        # depth_metrics already returns keys named exactly
        # ``depth_absrel`` / ``depth_rmse`` / ``depth_delta1`` — the same
        # keys evaluate_nvs aggregates on. Merge wholesale.
        row.update(depth_metrics(pred_depth, gt_depth))

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Compute-cost block
# ─────────────────────────────────────────────────────────────────────────────


def _compute_cost_block(
    adapter: Any,
    latencies_ms: List[float],
) -> Dict[str, Any]:
    """Build the ``compute_cost`` Axis-3 block from the adapter + measurements.

    Uses :class:`ModelProfiler` when the adapter exposes a ``torch_module``;
    otherwise reports params=0, flops=None and only the measured Tier-3
    numbers (latency, system card).
    """
    import numpy as np  # noqa: PLC0415

    block: Dict[str, Any] = {
        "params_m":              0.0,
        "flops_g":               None,
        "macs_g":                None,
        "memory_traffic_gb":     None,
        "arithmetic_intensity":  None,
        "roofline":              None,
        "latency_ms_per_sample": float(np.median(latencies_ms)) if latencies_ms else None,
        "peak_memory_mb":        None,
        "system_card":           None,
        "operating_point":       None,
    }

    torch_mod = getattr(adapter, "torch_module", None)
    if torch_mod is not None:
        try:
            from rpx_benchmark.model_profiler import ModelProfiler  # noqa: PLC0415

            profiler = ModelProfiler(torch_mod)
            eff = profiler.pre_run_profile()  # EfficiencyMetadata
            block.update(
                {
                    "params_m":             eff.params_m,
                    "flops_g":              eff.flops_g,
                    "macs_g":               eff.macs_g,
                    "memory_traffic_gb":    eff.memory_traffic_gb,
                    "arithmetic_intensity": eff.arithmetic_intensity,
                }
            )
        except Exception:
            # Profiling is best-effort — the rest of the report still ships.
            pass

    block["operating_point"] = {
        "precision":     getattr(adapter, "native_precision", "fp32"),
        "params_m":      block["params_m"],
        "flops_g":       block["flops_g"],
    }
    return block


# ─────────────────────────────────────────────────────────────────────────────
# Result-JSON assembly + write
# ─────────────────────────────────────────────────────────────────────────────


def _assemble_result(
    *,
    model_key:     str,
    display_name:  str,
    split:         str,
    per_sample:    List[Dict[str, Any]],
    cost_block:    Dict[str, Any],
    latencies_ms:  List[float],
    wall_seconds:  float,
) -> Dict[str, Any]:
    """Glue ``evaluate_nvs`` + cost block into the canonical three-axis layout."""
    from rpx_benchmark.nvs_metrics import evaluate_nvs  # noqa: PLC0415

    eval_out = evaluate_nvs(per_sample)
    aggregated     = eval_out.get("aggregated", {})
    by_sample_type = eval_out.get("by_sample_type", {})
    by_context     = eval_out.get("by_context_count", {})
    cross_delta    = eval_out.get("cross_phase_delta", {})

    return {
        "task":         "novel_view_synthesis",
        "model":        display_name,
        "model_key":    model_key,
        "split":        split,
        "num_samples":  len(per_sample),
        # Axis 1 — Task Performance
        "aggregated":   aggregated,
        # Axis 2 — Scene-change robustness
        "robustness":   {
            "by_sample_type":   by_sample_type,
            "by_context_count": by_context,
            "cross_phase_delta": cross_delta,
        },
        # Axis 3 — Compute cost
        "compute_cost": cost_block,
        # Timing — per-stage wall clocks
        "timing": {
            "model_run_ms_per_sample": {
                "median_ms": cost_block.get("latency_ms_per_sample"),
                "n_samples": len(latencies_ms),
            },
            "total_wall_seconds": wall_seconds,
        },
    }


def _write_summary_md(result: Dict[str, Any], path: Path) -> None:
    """Render a compact markdown summary of the three-axis result."""
    a = result.get("aggregated") or {}
    cc = result.get("compute_cost") or {}
    lines = [
        f"# {result['model']} — {result['split']}",
        "",
        f"- **Task:** `{result['task']}`",
        f"- **Samples:** {result['num_samples']}",
        "",
        "## Axis 1 — Task Performance",
        "",
    ]
    if "psnr" in a:           lines.append(f"- PSNR: **{a['psnr']:.3f} dB**")
    if "ssim" in a:           lines.append(f"- SSIM: **{a['ssim']:.4f}**")
    if "depth_absrel" in a:   lines.append(f"- Depth AbsRel: **{a['depth_absrel']:.4f}** (lower = better)")
    if "depth_rmse" in a:     lines.append(f"- Depth RMSE:   **{a['depth_rmse']:.4f} m**")
    if "depth_delta1" in a:   lines.append(f"- Depth δ<1.25: **{a['depth_delta1']:.4f}**")
    rb = result.get("robustness") or {}
    if rb.get("cross_phase_delta"):
        lines += ["", "## Axis 2 — Scene-change robustness", "",
                  "Cross-phase Δ (cross-phase − intra-phase):"]
        for k, v in rb["cross_phase_delta"].items():
            lines.append(f"- {k}: **{v:+.4f}**")
    lines += ["", "## Axis 3 — Compute cost", ""]
    if cc.get("params_m") is not None:
        lines.append(f"- Params: **{cc['params_m']:.2f} M**")
    if cc.get("flops_g") is not None:
        lines.append(f"- FLOPs: **{cc['flops_g']:.2f} G**")
    if cc.get("latency_ms_per_sample") is not None:
        lat = cc["latency_ms_per_sample"]
        # Sub-millisecond adapters (identity, lookup, etc.) need higher
        # precision; otherwise .1f rounds them to "0.0 ms".
        lat_str = f"{lat:.3f}" if lat < 1.0 else f"{lat:.1f}"
        lines.append(
            f"- Latency†: **{lat_str} ms / sample** "
            "(†hardware-dependent, supplementary)"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run an NVS model end-to-end against an RPX split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="registered NVS adapter key (see `python -c 'from nvs_models import list_models; print(list_models())'`)")
    ap.add_argument("--split", default="easy", choices=("easy", "medium", "hard"))
    ap.add_argument("--device", default="cuda",
                    help="device for the adapter (cpu / cuda / cuda:0 / mps)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap the number of NVS samples (smoke runs)")
    ap.add_argument("--extracted-root", type=Path, default=None,
                    help="override the auto-resolved HF snapshot's extracted/ root")
    ap.add_argument("--parquet-path", type=Path, default=None,
                    help="override the auto-resolved frames_v1.parquet path")
    ap.add_argument("--repo-id", default="IRVLUTD/RPX",
                    help="HuggingFace dataset repo id (used for cache resolution)")
    ap.add_argument("--results-root", type=Path, default=Path("./rpx_results"),
                    help="root dir for per-run output trees")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save rendered RGB + depth per sample under predictions/")
    args = ap.parse_args()

    from rpx_benchmark import cli_ux  # noqa: PLC0415

    cli_ux.banner(
        "run_nvs — Novel View Synthesis benchmark",
        "on-the-fly stratified samples, three-axis report, no composite",
    )
    cli_ux.config(vars(args))

    # ── Build pair generator
    cli_ux.section("Pair generator")
    with cli_ux.working("constructing NVSPairGenerator + loading parquet metadata"):
        from rpx_benchmark.nvs_pairs import NVSPairGenerator  # noqa: PLC0415

        # Resolve cache paths from local_manifest helpers if not given.
        if args.extracted_root is None or args.parquet_path is None:
            from local_manifest import _hf_snapshot_root  # noqa: PLC0415

            snap = _hf_snapshot_root(args.repo_id)
            extracted_root = args.extracted_root or (snap / "extracted")
            parquet_path = args.parquet_path or (snap / "manifest" / "frames_v1.parquet")
        else:
            extracted_root = args.extracted_root
            parquet_path = args.parquet_path

        gen = NVSPairGenerator(
            extracted_root=extracted_root,
            parquet_path=parquet_path,
            split=args.split,
        )
    cli_ux.note(gen.summary())

    # ── Resolve model
    cli_ux.section("Model")
    with cli_ux.working(f"building adapter '{args.model}'"):
        adapter = _build_model(args.model, args.device)
    from nvs_models import MODEL_DISPLAY_NAMES  # noqa: PLC0415

    display = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    cli_ux.kv("display name",     display)
    cli_ux.kv("native precision", getattr(adapter, "native_precision", "fp32"))
    cli_ux.kv("has torch module", getattr(adapter, "torch_module", None) is not None)

    # ── Run
    cli_ux.section("Render + evaluate")
    per_sample: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    saved_predictions: List[Path] = []
    pred_root = args.results_root / display / args.split / "predictions"

    t_wall_start = time.perf_counter()

    samples_iter = gen.iter_samples()
    total = args.max_samples  # may be None if unbounded
    with cli_ux.progress("samples", total=total) as (p, task):
        for i, sample in enumerate(samples_iter):
            if args.max_samples is not None and i >= args.max_samples:
                break

            # Load context arrays
            context_rgbs   = [_load_rgb(extracted_root / r)   for r in sample.context_rgb_paths]
            context_depths = [_load_depth(extracted_root / d) for d in sample.context_depth_paths]
            context_poses  = [_load_pose(extracted_root / pth) for pth in sample.context_pose_paths]
            target_pose    = _load_pose(extracted_root / sample.target_pose_path)
            gt_rgb         = _load_rgb(extracted_root / sample.target_rgb_path)
            gt_depth       = _load_depth(extracted_root / sample.target_depth_path)

            # Adapter call (timed)
            t0 = time.perf_counter()
            rendered = adapter(context_rgbs, context_depths, context_poses, target_pose)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            row = _evaluate_sample(sample, rendered, gt_rgb, gt_depth)
            per_sample.append(row)

            if args.save_predictions:
                import numpy as np  # noqa: PLC0415

                out_path = pred_root / sample.scene_id / str(sample.phase) / f"{sample.id}.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload: Dict[str, Any] = {"rgb": rendered.get("rgb")}
                if rendered.get("depth") is not None:
                    payload["depth"] = rendered["depth"]
                np.savez_compressed(out_path, **payload)
                saved_predictions.append(out_path)

            p.update(task, advance=1)  # type: ignore[attr-defined]

    wall_seconds = time.perf_counter() - t_wall_start

    if not per_sample:
        cli_ux.warn("No samples evaluated — check split, extracted_root, and parquet_path.")
        return

    # ── Build compute-cost block + assemble result
    cli_ux.section("Aggregate")
    cost_block = _compute_cost_block(adapter, latencies_ms)
    result = _assemble_result(
        model_key=args.model,
        display_name=display,
        split=args.split,
        per_sample=per_sample,
        cost_block=cost_block,
        latencies_ms=latencies_ms,
        wall_seconds=wall_seconds,
    )

    # ── Write artefacts
    out_dir = args.results_root / display / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary_md(result, md_path)
    cli_ux.step(f"wrote {json_path}")
    cli_ux.step(f"wrote {md_path}")
    if saved_predictions:
        cli_ux.step(f"saved {len(saved_predictions)} predictions under {pred_root}")

    # ── Summary panel
    a = result["aggregated"]
    rows: Dict[str, Any] = {
        "samples":  result["num_samples"],
        "wall":     cli_ux.fmt_duration(wall_seconds),
        "rate":     cli_ux.fmt_rate(result["num_samples"], wall_seconds),
    }
    if "psnr" in a:  rows["PSNR"] = f"{a['psnr']:.3f} dB"
    if "ssim" in a:  rows["SSIM"] = f"{a['ssim']:.4f}"
    if "depth_absrel" in a:  rows["depth AbsRel"] = f"{a['depth_absrel']:.4f}"
    if cost_block.get("latency_ms_per_sample") is not None:
        lat = cost_block["latency_ms_per_sample"]
        lat_str = f"{lat:.3f}" if lat < 1.0 else f"{lat:.1f}"
        rows["latency"] = f"{lat_str} ms / sample"
    cli_ux.summary(rows, title=f"{display} · {args.split}")


if __name__ == "__main__":
    main()

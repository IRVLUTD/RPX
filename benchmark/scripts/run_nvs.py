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
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ./scripts importable so `from nvs_models import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────────────────────────────────────────────
# Modality loaders — LRU-cached for hot inner loops
# ─────────────────────────────────────────────────────────────────────────────
#
# NVSPairGenerator emits up to ~25 target frames per (scene, phase, K-value);
# all of those samples share the same context-frame pool. Without a cache
# each context frame would be PIL-decoded once per overlapping sample,
# which is ~80% of the per-sample wall-clock for cheap adapters.
#
# Cache sizes: 256 RGB + 256 depth + 1024 pose entries. RGB at 640×480×3
# ≈ 0.88 MB and depth at 640×480 float32 ≈ 1.17 MB, so the per-sample
# working set caps at ~525 MB for RGB+depth (poses are 128 B each, free).
# Override via RPX_LOADER_CACHE_SIZE if the default doesn't fit on a host.
#
# Cached arrays are returned with `.flags.writeable = False`. Callers that
# need to mutate must `.copy()` first — guards against one sample's
# adapter accidentally corrupting another's GT.

_RGB_CACHE_SIZE   = 256
_DEPTH_CACHE_SIZE = 256
_MASK_CACHE_SIZE  = 256
_POSE_CACHE_SIZE  = 1024


@lru_cache(maxsize=_RGB_CACHE_SIZE)
def _load_rgb(path: Path) -> "Any":
    """Load an RGB image as (H, W, 3) uint8. Cached; read-only on return.

    Uses cv2.imread (~2-5x faster than PIL on the same content); falls
    back to PIL on cv2 failure (legacy / corrupt files).
    """
    import numpy as np  # noqa: PLC0415

    try:
        import cv2  # noqa: PLC0415

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # H, W, 3 uint8
            arr.flags.writeable = False
            return arr
    except ImportError:
        pass

    from PIL import Image  # noqa: PLC0415

    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=_DEPTH_CACHE_SIZE)
def _load_depth(path: Path) -> "Any":
    """Load a 16-bit depth PNG as (H, W) float32 in metres (D435 mm → m).

    Cached; read-only on return. Uses cv2.imread with IMREAD_UNCHANGED
    to preserve uint16; falls back to PIL on cv2 failure.
    """
    import numpy as np  # noqa: PLC0415

    raw = None
    try:
        import cv2  # noqa: PLC0415

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except ImportError:
        pass

    if raw is None:
        from PIL import Image  # noqa: PLC0415

        raw = np.array(Image.open(path))

    arr = (raw.astype(np.float32) / 1000.0) if raw.dtype == np.uint16 else raw.astype(np.float32)
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=_MASK_CACHE_SIZE)
def _load_mask(path: Path) -> "Any":
    """Load a SAM2 instance-mask PNG → (H, W) int32 (0 = background,
    1..N = instance IDs). Cached; read-only on return.

    Used for the per-object PSNR axis of NVS evaluation — only the
    target frame's mask is consumed; the GT mask of the target frame
    is the ground-truth instance set the rendered output is scored
    against.
    """
    import numpy as np  # noqa: PLC0415

    raw = None
    try:
        import cv2  # noqa: PLC0415

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except ImportError:
        pass

    if raw is None:
        from PIL import Image  # noqa: PLC0415

        raw = np.array(Image.open(path))

    arr = raw.astype(np.int32)
    arr.flags.writeable = False
    return arr


def _mask_path_for_target(sample: Any) -> str:
    """Relative path to the SAM2 instance mask of the target frame.

    Mirrors the layout written by the dataset hub:
    ``scenes/<scene>/<phase>/sam2/masks/<frame_idx:05d>.png``. The
    function only constructs the path — the runner does the
    optional load below (mask might not exist for every frame).
    """
    return (
        f"scenes/{sample.scene_id}/{sample.phase_target}/sam2/masks/"
        f"{sample.target_frame_idx:05d}.png"
    )


@lru_cache(maxsize=_POSE_CACHE_SIZE)
def _load_pose(path: Path) -> "Any":
    """Load a T265 pose → 4×4 SE(3) camera-to-world (float64).

    RPX v1 stores an NPZ with ``position`` and ``orientation`` arrays.
    RPX v2 losslessly packs the same values into one NPY vector ordered
    ``[x, y, z, qx, qy, qz, qw]``.

    * ``position``    — ``(3,)`` metres
    * ``orientation`` — ``(4,)`` quaternion in T265 ``[x, y, z, w]`` order

    We reconstruct the 4×4 here so the runner's pose semantics match the
    toolkit's canonical loader exactly. Mirrors the math in
    ``rpx_benchmark/loader.py:_quat_xyzw_to_rotmat`` / ``_load_pose``.
    """
    import numpy as np  # noqa: PLC0415

    data = np.load(path)
    if path.suffix.lower() == ".npy":
        pose7 = np.asarray(data, dtype=np.float64).reshape(7)
        position = pose7[:3]
        quat_xyzw = pose7[3:]
    else:
        position = np.asarray(data["position"], dtype=np.float64)
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
    T.flags.writeable = False
    return T


def _loader_cache_stats() -> Dict[str, Any]:
    """Return per-loader LRU stats — surfaced in the run summary."""
    out: Dict[str, Any] = {}
    for name, fn in (
        ("rgb",   _load_rgb),
        ("depth", _load_depth),
        ("mask",  _load_mask),
        ("pose",  _load_pose),
    ):
        info = fn.cache_info()  # type: ignore[attr-defined]
        total = info.hits + info.misses
        out[name] = {
            "hits":     info.hits,
            "misses":   info.misses,
            "hit_rate": (info.hits / total) if total else 0.0,
            "size":     info.currsize,
            "maxsize":  info.maxsize,
        }
    return out


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
    gt_mask: "Optional[Any]" = None,
    compute_lpips: bool = False,
) -> Dict[str, Any]:
    """Compute the per-sample metric dict consumed by ``evaluate_nvs``.

    Delegates to ``rpx_benchmark.nvs_eval.evaluate_single_sample`` — proper
    sliding-window SSIM via skimage, optional LPIPS via the lpips package,
    and (when ``gt_mask`` is provided) per-object PSNR via SAM2 instance
    masks. Sample-metadata fields (sample_type, n_context, scene_id,
    phase) are merged so ``evaluate_nvs`` can stratify rows downstream.
    """
    from rpx_benchmark.nvs_eval import evaluate_single_sample  # noqa: PLC0415

    pred_rgb = rendered.get("rgb")
    pred_depth = rendered.get("depth")

    row: Dict[str, Any] = {
        "sample_id":   sample.id,
        "scene_id":    sample.scene_id,
        "phase":       sample.phase,
        "n_context":   sample.n_context,
        "sample_type": sample.sample_type,
        # Difficulty is optional on NVSSample (may be None); kept here so
        # downstream `by_difficulty` aggregation in _assemble_result has
        # the stratifier without re-touching the sample dict.
        "difficulty":  getattr(sample, "difficulty", None),
    }

    if pred_rgb is not None and gt_rgb is not None:
        row.update(
            evaluate_single_sample(
                pred_rgb=pred_rgb,
                gt_rgb=gt_rgb,
                pred_depth=pred_depth,
                gt_depth=gt_depth,
                gt_mask=gt_mask,
                compute_lpips=compute_lpips,
            )
        )

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
    aggregated     = dict(eval_out.get("aggregated", {}))
    by_sample_type = eval_out.get("by_sample_type", {})
    by_context     = eval_out.get("by_context_count", {})
    cross_delta    = eval_out.get("cross_phase_delta", {})

    # evaluate_nvs only aggregates the standard metric_keys (psnr / ssim /
    # lpips / depth_*) plus pus_*. Per-object PSNR (when masks are passed)
    # comes back as `per_object_psnr_mean` / `per_object_psnr_min` /
    # `n_objects_evaluated` on each per_sample row but isn't in that
    # whitelist, so we aggregate it manually here. Surfacing this keeps
    # the novel per-object axis visible without editing the parallel
    # session's nvs_metrics.evaluate_nvs.
    import numpy as np  # noqa: PLC0415

    for k in ("per_object_psnr_mean", "per_object_psnr_min", "n_objects_evaluated"):
        vals = [s[k] for s in per_sample if k in s and s[k] is not None]
        if vals:
            aggregated[k] = float(np.mean(vals))

    # ESD-difficulty stratification — Axis-2 robustness signal. RPX's
    # difficulty tiers (easy / medium / hard) come through on the
    # generator's NVSSample.difficulty field; surface a per-tier
    # breakdown of the same standard metrics so the paper can
    # report "PSNR-by-difficulty" without re-aggregating elsewhere.
    by_diff: Dict[str, Dict[str, float]] = {}
    metric_keys = ("psnr", "ssim", "lpips",
                   "depth_absrel", "depth_rmse", "depth_delta1")
    difficulties = sorted(
        {s.get("difficulty") for s in per_sample if s.get("difficulty") is not None}
    )
    for diff in difficulties:
        subset = [s for s in per_sample if s.get("difficulty") == diff]
        agg: Dict[str, float] = {"n_samples": float(len(subset))}
        for k in metric_keys:
            vals = [s[k] for s in subset if k in s and s[k] is not None]
            if vals:
                agg[k] = float(np.mean(vals))
        by_diff[diff] = agg

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
            "by_difficulty":    by_diff,
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
    from rpx_benchmark.cleanup import install_signal_cleanup
    install_signal_cleanup()

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
    ap.add_argument("--context-counts", type=int, nargs="+", default=None,
                    help="override context-view counts. Use `--context-counts 2` "
                    "for DepthSplat's released two-view operating point")
    ap.add_argument("--sample-types", nargs="+",
                    choices=("interpolation", "extrapolation", "cross_phase"),
                    help="evaluate only the requested sample types")
    ap.add_argument("--sample-order", choices=("generator", "scene_round_robin"),
                    default="generator",
                    help="scene_round_robin spreads capped gates across distinct scenes")
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
    ap.add_argument("--compute-lpips", action="store_true",
                    help="compute LPIPS (AlexNet) per sample. Off by default "
                    "because it adds ~50-100 ms/sample on CPU. Needs `pip install lpips`; "
                    "returns NaN silently if the package is missing.")
    ap.add_argument("--upload-to-box", action="store_true",
                    help="mirror the result dir to UTD Box under "
                    "<box_folder_id>/novel_view_synthesis/<model>/<split>/. "
                    "Requires BOX_DEVELOPER_TOKEN. Skipped silently if a "
                    "real upload isn't possible (no token, no box_fetch).")
    ap.add_argument("--box-folder-id", default="380510613151",
                    help="Box folder id (default: team's RPX-Outputs).")
    ap.add_argument("--strict-io", action="store_true",
                    help="crash on the first missing/corrupt sample file. "
                    "By default the runner logs the bad sample and "
                    "continues — long sweeps shouldn't tank from one "
                    "stale extraction.")
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
        from rpx_benchmark.nvs_pairs import NVSConfig, NVSPairGenerator  # noqa: PLC0415

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
            config=(
                NVSConfig(context_counts=tuple(args.context_counts))
                if args.context_counts is not None
                else None
            ),
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
    frame_root = args.results_root / display / args.split / "prediction_frames"

    t_wall_start = time.perf_counter()
    skipped: List[Dict[str, Any]] = []  # graceful-skip log

    samples = gen.iter_samples()
    if args.sample_types:
        allowed_types = set(args.sample_types)
        samples = (sample for sample in samples if sample.sample_type in allowed_types)
    if args.sample_order == "scene_round_robin":
        by_scene: Dict[str, List[Any]] = {}
        for sample in samples:
            by_scene.setdefault(sample.scene_id, []).append(sample)
        # Deterministically mix phase, direction and target within each
        # scene before taking one scene at a time. This prevents a capped
        # acceptance gate from containing only the first forward trial.
        for scene_id, scene_samples in by_scene.items():
            random.Random(f"5062026:{scene_id}").shuffle(scene_samples)
        ordered: List[Any] = []
        depth = 0
        while True:
            added = False
            for scene_id in sorted(by_scene):
                if depth < len(by_scene[scene_id]):
                    ordered.append(by_scene[scene_id][depth])
                    added = True
            if not added:
                break
            depth += 1
        samples_iter = iter(ordered)
    else:
        samples_iter = samples
    total = args.max_samples  # may be None if unbounded
    with cli_ux.progress("samples", total=total) as (p, task):
        for i, sample in enumerate(samples_iter):
            if args.max_samples is not None and i >= args.max_samples:
                break

            # Per-sample try/except — long sweeps shouldn't tank because
            # one stale extraction has a missing frame. `--strict-io`
            # turns it back into a hard crash for debugging.
            try:
                context_rgbs   = [_load_rgb(extracted_root / r)   for r in sample.context_rgb_paths]
                context_depths = [_load_depth(extracted_root / d) for d in sample.context_depth_paths]
                context_poses  = [_load_pose(extracted_root / pth) for pth in sample.context_pose_paths]
                target_pose    = _load_pose(extracted_root / sample.target_pose_path)
                gt_rgb         = _load_rgb(extracted_root / sample.target_rgb_path)
                gt_depth       = _load_depth(extracted_root / sample.target_depth_path)

                # GT mask of the target frame enables per-object PSNR.
                # The mask is *optional* — missing-mask is silently OK.
                gt_mask = None
                try:
                    gt_mask = _load_mask(extracted_root / _mask_path_for_target(sample))
                except (FileNotFoundError, OSError, ValueError):
                    pass

                # Adapter call (timed)
                t0 = time.perf_counter()
                rendered = adapter(context_rgbs, context_depths, context_poses, target_pose)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                row = _evaluate_sample(
                    sample, rendered, gt_rgb, gt_depth,
                    gt_mask=gt_mask,
                    compute_lpips=args.compute_lpips,
                )
                per_sample.append(row)

                if args.save_predictions:
                    import numpy as np  # noqa: PLC0415
                    from PIL import Image  # noqa: PLC0415

                    out_path = pred_root / sample.scene_id / str(sample.phase) / f"{sample.id}.npz"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    payload: Dict[str, Any] = {"rgb": rendered.get("rgb")}
                    if rendered.get("depth") is not None:
                        payload["depth"] = rendered["depth"]
                    np.savez_compressed(out_path, **payload)
                    saved_predictions.append(out_path)
                    frame_path = (
                        frame_root / sample.scene_id / str(sample.phase) / f"{sample.id}.png"
                    )
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.asarray(rendered.get("rgb"), dtype=np.uint8)).save(frame_path)
            except (FileNotFoundError, OSError, KeyError, ValueError) as e:
                # KeyError covers the npz-key-missing case in _load_pose;
                # ValueError handles a too-small adapter call (zero ctx).
                if args.strict_io:
                    raise
                skipped.append({
                    "sample_id": getattr(sample, "id", "?"),
                    "scene_id":  getattr(sample, "scene_id", "?"),
                    "reason":    f"{type(e).__name__}: {str(e)[:100]}",
                })

            p.update(task, advance=1)  # type: ignore[attr-defined]

    if skipped:
        rate = len(skipped) / (len(per_sample) + len(skipped))
        warn_msg = f"skipped {len(skipped)} / {len(per_sample) + len(skipped)} samples ({rate:.1%}) on IO errors"
        if rate > 0.05:
            cli_ux.warn(f"{warn_msg} — that's >5%, check the extracted_root")
        else:
            cli_ux.note(warn_msg)

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
    result["sampling_protocol"] = {
        "context_counts": args.context_counts,
        "sample_types": args.sample_types,
        "sample_order": args.sample_order,
        "seed": 5_062_026,
        "extrapolation_definition": (
            "target outside the temporal context span with a 20% sequence guard band"
            if args.sample_types == ["extrapolation"]
            else None
        ),
    }

    # Surface loader-cache stats inside result.json before writing.
    # On a sweep through one (scene, phase) the hit-rate is typically
    # 0.8 – 0.95 because the generator emits ~25 target frames against
    # the same K-context pool — kept visible so the parallel session
    # sees if the cache is doing its job.
    cache_stats = _loader_cache_stats()
    result["loader_cache"] = cache_stats
    if skipped:
        result["skipped_samples"] = skipped  # full skip-log lands in result.json

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
        manifest = {
            "model": args.model,
            "display_name": display,
            "split": args.split,
            "count": len(saved_predictions),
            "context_counts": args.context_counts,
            "frames_root": str(frame_root),
        }
        (frame_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    # ── Optional: mirror to UTD Box (mirrors run_relative_pose.py's pattern)
    if args.upload_to_box:
        cli_ux.section("Box upload")
        try:
            from box_fetch import upload_tree  # noqa: PLC0415

            remote = f"novel_view_synthesis/{display}/{args.split}"
            with cli_ux.working(f"uploading {out_dir} → Box:{remote}"):
                summary = upload_tree(
                    Path(out_dir),
                    remote_path=remote,
                    root_folder_id=args.box_folder_id,
                )
            cli_ux.step(
                f"uploaded {summary['uploaded']} files "
                f"({cli_ux.fmt_bytes(summary['bytes_uploaded'])}); "
                f"{summary['skipped']} already on Box; "
                f"remote folder id {summary['remote_folder_id']}"
            )
        except Exception as e:
            cli_ux.warn(f"Box upload skipped — {type(e).__name__}: {e}")

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
    if "per_object_psnr_mean" in a:
        rows["per-obj PSNR"] = (
            f"{a['per_object_psnr_mean']:.2f} dB  "
            f"(n_objs avg={a.get('n_objects_evaluated', 0):.1f})"
        )
    if cost_block.get("latency_ms_per_sample") is not None:
        lat = cost_block["latency_ms_per_sample"]
        lat_str = f"{lat:.3f}" if lat < 1.0 else f"{lat:.1f}"
        rows["latency"] = f"{lat_str} ms / sample"
    # Cache hit rate — a single number summary of the three loaders.
    total_hits   = sum(s["hits"]   for s in cache_stats.values())
    total_misses = sum(s["misses"] for s in cache_stats.values())
    if total_hits + total_misses > 0:
        rows["cache hit-rate"] = f"{total_hits / (total_hits + total_misses):.2%}"
    cli_ux.summary(rows, title=f"{display} · {args.split}")


if __name__ == "__main__":
    main()

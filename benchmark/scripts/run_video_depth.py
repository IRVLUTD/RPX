"""Run a Video Depth model end-to-end against an RPX split.

Mirrors ``scripts/run_depth.py`` (Image Depth) but iterates per-clip
(``(scene, phase)`` tuple) instead of per-frame, and routes the
prediction through per-clip ``(s, t)`` alignment for relative-depth
models. Outputs:

* ``./rpx_results/<model>/<split>/result.json`` — aggregated metrics
* ``./rpx_results/<model>/<split>/cells.parquet`` — per-(scene, phase) cells
* ``./rpx_results/<model>/<split>/summary.md`` — short markdown summary

Usage
-----
    PYTHONPATH=. python scripts/run_video_depth.py \\
        --model depth-crafter --split easy

    # Temporal-resolution ablation (paper §5.2):
    PYTHONPATH=. python scripts/run_video_depth.py \\
        --model depth-crafter --split easy \\
        --frame-budget 75 --sampling fps_se3

    # With Box upload (team workflow):
    PYTHONPATH=. python scripts/run_video_depth.py \\
        --model depth-crafter --split easy --upload-to-box

The ``--model`` value must resolve to a callable that:

* takes a ``(T, H, W, 3)`` uint8 RGB sequence (or a list of such
  sequences for batched calls)
* returns a ``(T, H, W) float32`` depth-in-metres tensor (or list)
* declares ``depth_output_kind = "metric"`` or ``"relative"`` so the
  runner knows whether to apply per-clip alignment.

For now, model adapter registration is left to the contributor: the
20 canonical models live in ``rpx_benchmark.adapters.video_depth``
(skeletons) and ``scripts/depth_models/`` (Image Depth implementations the
video models will eventually port to). See
``benchmark/README.md`` for the team handoff workflow.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

DEFAULT_DATASET_REPO = "anonymous/RPX"
PINNED_DATASET_REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"

# Make ./scripts importable so `from depth_models.* import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_model(name: str, device: str, *, acknowledge_unverified: bool = False):
    """Resolve a model adapter by name.

    Lookup order:

    1. ``rpx_benchmark.adapters.video_depth.<name>`` — a contributor's
       drop-in module that exposes ``build(device)`` returning a
       ``BenchmarkModel`` whose ``task = TaskType.VIDEO_DEPTH``.
    2. ``video_depth_models.<name>`` — sibling of the existing
       ``depth_models`` directory, for in-development adapters.

    A new model is added by creating one of those modules and
    implementing ``build(device) -> BenchmarkModel``. See the team
    guide for a worked DA3 example.
    """
    candidate_modules = [
        f"rpx_benchmark.adapters.video_depth.{name.replace('-', '_')}",
        f"video_depth_models.{name.replace('-', '_')}",
    ]
    last_err: Exception | None = None
    for module_name in candidate_modules:
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            last_err = e
            continue
        if not hasattr(mod, "build"):
            raise SystemExit(
                f"{module_name} found but exposes no `build(device)` function. "
                "Add `def build(device: str) -> BenchmarkModel:` to the module."
            )
        # Only legacy adapters with a safety rail accept
        # acknowledge_unverified — try with it, fall back without.
        kwargs = {"device": device}
        if acknowledge_unverified:
            kwargs["acknowledge_unverified"] = True
        try:
            return mod.build(**kwargs)
        except TypeError as e:
            if "acknowledge_unverified" in str(e):
                kwargs.pop("acknowledge_unverified", None)
                return mod.build(**kwargs)
            raise
    raise SystemExit(
        f"No model adapter resolved for --model {name!r}. "
        f"Tried: {candidate_modules}. "
        f"Last error: {last_err}. "
        f"See benchmark/README.md for how to add an adapter."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[0] if __doc__ else "",
    )
    ap.add_argument("--model", required=True, help="adapter name (kebab-case)")
    ap.add_argument("--split", default="easy", help="easy | medium | hard")
    ap.add_argument(
        "--repo", default=DEFAULT_DATASET_REPO, help="HuggingFace dataset repo"
    )
    ap.add_argument(
        "--revision",
        default=PINNED_DATASET_REVISION,
        help="immutable Hugging Face dataset commit",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow an explicit CPU diagnostic; depth runs are GPU-only by default",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="default: ./rpx_results/<model>/<split>/",
    )
    ap.add_argument(
        "--manifest-path",
        default=None,
        help="Local manifest JSON (skip HF download). Use for runs against "
        "a staged lossless v2-webp tree before HF upload. Path is typically "
        "<staging>/manifests/video_depth/<split>.json from "
        "`python -m rpx_benchmark.dataset_hub.cli manifest`.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of clips to run (smoke testing)",
    )
    ap.add_argument(
        "--frame-budget",
        type=int,
        default=None,
        help="Subsample to N frames per clip (paper §5.2 ablation). "
        "Default: use every frame on disk.",
    )
    ap.add_argument(
        "--sampling",
        default="all",
        choices=["all", "stride", "fps_se3"],
        help="Subsampling strategy when --frame-budget is set",
    )
    ap.add_argument(
        "--budget-sweep",
        default=None,
        help="Comma-separated frame budgets to sweep (e.g. 50,100,150,250). "
        "Loops the pipeline once per budget, writing each budget's cells.parquet "
        "into <output_dir>/budget_<N>/ so the raw per-budget metrics are kept "
        "separate for downstream degradation analysis. Model weights are loaded "
        "once and reused across budgets. Mutually exclusive with --frame-budget. "
        "Sampling defaults to 'stride' when the sweep is on (override with --sampling).",
    )
    ap.add_argument(
        "--upload-to-box",
        action="store_true",
        help="After the run, ship results to UTD Box. Requires BOX_DEVELOPER_TOKEN.",
    )
    ap.add_argument(
        "--save-predictions",
        action="store_true",
        help="Atomically save one validated float32 NPZ per scene-phase clip.",
    )
    ap.add_argument(
        "--resume-predictions",
        action="store_true",
        help="Reuse validated saved clip predictions and infer only missing or "
        "corrupt clips. Requires --save-predictions.",
    )
    ap.add_argument(
        "--compute-fscore",
        action="store_true",
        help="Compute diagnostic F@5cm inline. Disabled by default because it "
        "is not in the D1-V K=6 vector and is CPU-expensive.",
    )
    ap.add_argument(
        "--acknowledge-unverified",
        action="store_true",
        help="Reserved for legacy adapters whose upstream weights have not "
        "been verified against the paper authors' release.",
    )
    args = ap.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        ap.error("--max-samples must be >= 1")
    if args.budget_sweep is not None and args.frame_budget is not None:
        ap.error("--budget-sweep and --frame-budget are mutually exclusive")
    if args.resume_predictions and not args.save_predictions:
        ap.error("--resume-predictions requires --save-predictions")

    budgets = None
    if args.budget_sweep is not None:
        try:
            budgets = [int(b) for b in args.budget_sweep.split(",")]
        except ValueError:
            ap.error(f"--budget-sweep must be comma-separated ints, got {args.budget_sweep!r}")
        if not budgets or any(b < 1 for b in budgets):
            ap.error("--budget-sweep values must all be >= 1")

    from rpx_benchmark.tasks._pipeline import resolve_device

    args.device = resolve_device(args.device, require_cuda=not args.allow_cpu)

    model = _load_model(
        args.model,
        args.device,
        acknowledge_unverified=args.acknowledge_unverified,
    )

    from rpx_benchmark.tasks.video_depth import VideoDepthRunConfig, run_video_depth

    if budgets is None:
        # Single-run path (existing behaviour).
        cfg = VideoDepthRunConfig(
            model=model,
            split=args.split,
            repo_id=args.repo,
            revision=args.revision,
            device=args.device,
            require_cuda=not args.allow_cpu,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            frame_budget=args.frame_budget,
            sampling=args.sampling,
            max_samples=args.max_samples,
            upload_to_box=args.upload_to_box,
            save_predictions=args.save_predictions,
            resume_predictions=args.resume_predictions,
            compute_fscore=args.compute_fscore,
        )
        _result, _dr, paths = run_video_depth(cfg)
        print("Video Depth run complete.")
        print(f"  result.json : {paths['json']}")
        print(f"  cells       : {paths['cells']}")
        print(f"  summary.md  : {paths['markdown']}")
        print(f"  metadata    : {paths['run_metadata']}")
        if "predictions_dir" in paths:
            print(f"  predictions : {paths['predictions_dir']}")
            print(f"  resume      : {paths['prediction_stats']}")
        if "box_remote" in paths:
            print(f"  box         : {paths['box_remote']}")
        return

    # Frame-budget sweep — one pipeline invocation per budget, model reused.
    # Sampling defaults to "stride" for sweeps unless the user picked
    # something specific; "all" doesn't make sense once a budget is set.
    sampling = args.sampling if args.sampling != "all" else "stride"
    base_output = args.output_dir or f"./rpx_results/{args.model}/{args.split}"
    print(f"[sweep] budgets={budgets}, sampling={sampling}, base={base_output}")

    per_budget_paths: dict[int, dict] = {}
    for budget in budgets:
        budget_dir = f"{base_output}/budget_{budget}"
        print(f"[sweep] budget={budget}: running → {budget_dir}")
        cfg = VideoDepthRunConfig(
            model=model,
            split=args.split,
            repo_id=args.repo,
            revision=args.revision,
            device=args.device,
            require_cuda=not args.allow_cpu,
            output_dir=budget_dir,
            manifest_path=args.manifest_path,
            frame_budget=budget,
            sampling=sampling,
            max_samples=args.max_samples,
            upload_to_box=args.upload_to_box,
            save_predictions=args.save_predictions,
            resume_predictions=args.resume_predictions,
            compute_fscore=args.compute_fscore,
        )
        _result, _dr, paths = run_video_depth(cfg)
        per_budget_paths[budget] = paths
        print(f"[sweep] budget={budget}: cells={paths['cells']}")

    print("\n[sweep] complete — one cells.parquet per budget:")
    for budget, paths in per_budget_paths.items():
        print(f"  budget={budget:>3d}: {paths['cells']}")
    print(
        "\nDegradation analysis (TCV, AUDC, critical budget, Wilcoxon) is a "
        "post-processing step over the per-budget cell logs — see "
        "rpx_benchmark.temporal_budget_sweep.degradation_analysis."
    )


if __name__ == "__main__":
    main()

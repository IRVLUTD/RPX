"""``rpx`` command-line entrypoint.

The CLI is intentionally thin: every subcommand delegates to the
matching registered :class:`TaskSpec` so adding a new task requires
zero CLI changes. The heavy lifting (download, load, evaluate,
report) lives in the task's own module.

Usage
-----

Dataset ops::

    rpx ls                                        list tasks + splits
    rpx models                                    list model adapters
    rpx info     --task <t> --split <d>           split stats
    rpx download --task <t> --split <d>           pre-fetch files
    rpx bench    <task>    [task-specific flags]  end-to-end benchmark

Global flags::

    --verbose / -v    DEBUG-level logging
    --quiet   / -q    WARNING-level logging only
    --plain           disable rich UI (for logs / CI)

Exit codes
----------

    0    success
    1    RPXError raised (config, dataset, model, metric, download)
    2    CLI argument error (handled by argparse)
    130  KeyboardInterrupt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import hub
from .api import Difficulty, TaskType
from .banner import show_banner
from .exceptions import RPXError
from .logging_utils import configure_logging, get_logger
from .models.registry import DEFERRED_MODELS, available_models
from .tasks.registry import iter_task_specs

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Task-independent subcommands
# --------------------------------------------------------------------------- #

def _add_common_task_split(p: argparse.ArgumentParser) -> None:
    """Attach the shared --task / --split / --repo / --cache-dir flags."""
    p.add_argument("--task", required=True, help="Task name, e.g. monocular_depth")
    p.add_argument(
        "--split", required=True,
        choices=[d.value for d in Difficulty],
        help="ESD difficulty split.",
    )
    p.add_argument("--repo", default=hub.DEFAULT_REPO_ID, help="HF dataset repo id.")
    p.add_argument("--cache-dir", default=None, help="HF cache directory.")
    p.add_argument("--revision", default=None, help="HF revision/branch/tag.")


def _cmd_download(args: argparse.Namespace) -> int:
    manifest_path = hub.download_split(
        task=args.task,
        split=args.split,
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest: {manifest_path}")
    print(f"Root:     {manifest.get('root')}")
    print(f"Samples:  {len(manifest.get('samples', []))}")
    return 0


def _cmd_ls(_args: argparse.Namespace) -> int:
    print("Tasks:")
    for t in TaskType:
        mods = hub.TASK_MODALITIES.get(t, [])
        print(f"  {t.value:<22} modalities: {', '.join(mods)}")
    print("\nSplits: " + ", ".join(d.value for d in Difficulty))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    manifest = hub.fetch_manifest(
        task=args.task,
        split=args.split,
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    pairs = hub._extract_scene_phase_pairs(manifest)
    print(f"Task:    {args.task}")
    print(f"Split:   {args.split}")
    print(f"Scenes:  {len({s for s, _ in pairs})}")
    print(f"(scene, phase) pairs: {len(pairs)}")
    print(f"Samples: {len(manifest.get('samples', []))}")
    return 0


def _cmd_models(_args: argparse.Namespace) -> int:
    runnable = available_models(include_deferred=False)
    deferred = sorted(DEFERRED_MODELS)
    print("Runnable models:")
    for n in runnable:
        print(f"  {n}")
    if deferred:
        print("\nDeferred (registered for visibility; raise on resolve):")
        for n in deferred:
            print(f"  {n}")
    return 0


# --------------------------------------------------------------------------- #
# Auto-generated `rpx bench <task>` subcommands
# --------------------------------------------------------------------------- #

def _make_bench_runner(spec):
    """Produce a closure that runs the given TaskSpec end-to-end."""
    def _runner(args: argparse.Namespace) -> int:
        from .ui import ConsoleUI
        ui = ConsoleUI.auto(force_plain=getattr(args, "plain", False))
        ui.header(
            spec.display_name,
            model=getattr(args, "model", None) or getattr(args, "hf_checkpoint", "<none>"),
            split=args.split,
            repo=getattr(args, "repo", hub.DEFAULT_REPO_ID),
            device=getattr(args, "device", "cuda"),
        )

        cfg = spec.build_config(args)
        cfg.progress = ui.progress_cb(total=None)
        try:
            result, dr_report, paths = spec.run(cfg)
        finally:
            close = getattr(ui._backend, "close_progress", None)
            if close:
                close()

        ui.result_table(result.aggregated, title="Aggregated metrics")
        if dr_report is not None:
            ui.phase_score_table(dr_report.weighted_phase_score)
            ui.efficiency_table(dr_report)
        ui.footer(
            samples=result.num_samples,
            json=paths.get("json"),
            markdown=paths.get("markdown"),
        )
        return 0
    return _runner


def _attach_bench_subcommands(bench_parser: argparse._SubParsersAction) -> None:
    """Populate the ``rpx bench`` subparser from the task registry."""
    specs = iter_task_specs()
    if not specs:
        log.debug("no tasks registered; `rpx bench` will have no subcommands")
        return
    for spec in specs:
        sub = bench_parser.add_parser(
            spec.task.value,
            help=f"{spec.display_name} benchmark",
            description=spec.description or spec.display_name,
        )
        spec.add_cli_arguments(sub)
        sub.add_argument("--plain", action="store_true",
                         help="Disable rich UI; use plain-text output.")
        sub.set_defaults(func=_make_bench_runner(spec))


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argparse parser.

    Importing ``rpx_benchmark.tasks`` (triggered by the line above the
    task-registry import) populates the task registry as a side
    effect, so the auto-generated bench subcommands are available at
    parser-construction time.
    """
    # Trigger task self-registration.
    from . import tasks  # noqa: F401

    parser = argparse.ArgumentParser(
        prog="rpx",
        description=(
            "RPX — choose and rank perception models for robot learning. "
            "Bring your model; we bring the dataset, splits, metrics, and "
            "deployment-readiness scoring."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress INFO logging; show WARNING and above only.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download files for a (task, split)")
    _add_common_task_split(p_dl)
    p_dl.set_defaults(func=_cmd_download)

    p_ls = sub.add_parser("ls", help="List supported tasks and splits")
    p_ls.set_defaults(func=_cmd_ls)

    p_info = sub.add_parser("info", help="Show split stats without downloading frames")
    _add_common_task_split(p_info)
    p_info.set_defaults(func=_cmd_info)

    p_bench = sub.add_parser("bench", help="Run a benchmark end-to-end")
    bench_sub = p_bench.add_subparsers(dest="bench_task", required=True)
    _attach_bench_subcommands(bench_sub)

    p_models = sub.add_parser("models", help="List registered model adapters")
    p_models.set_defaults(func=_cmd_models)

    return parser


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def _log_level_from_flags(args: argparse.Namespace) -> str:
    if getattr(args, "verbose", False):
        return "DEBUG"
    if getattr(args, "quiet", False):
        return "WARNING"
    return "INFO"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entrypoint. Installs logging and maps exceptions to exit codes.

    Shows the RPX banner once per invocation unless ``--quiet`` is
    passed or the ``RPX_NO_BANNER`` environment variable is set. The
    banner writes to ``stderr`` so it does not interfere with
    ``stdout`` parsing pipelines.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(_log_level_from_flags(args))

    # Skip the banner for --quiet so scripting pipelines can opt out
    # with one flag. Everything else shows it.
    if not getattr(args, "quiet", False):
        show_banner(
            subtitle=f"rpx {args.cmd}"
            + (f" {args.bench_task}" if getattr(args, "bench_task", None) else ""),
        )

    try:
        return args.func(args)
    except RPXError as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())

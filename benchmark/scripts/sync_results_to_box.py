"""Post-hoc Box mirror for any local ``rpx_results/`` tree.

Use when:

* A sweep ran without ``--upload-to-box`` and you don't want to re-run.
* The Box token expired mid-sweep and some runs never made it.
* You want to re-sync after manually editing a ``result.json`` or
  ``summary.md`` (re-upload is size-matched idempotent — only changed
  files travel the wire).

The runner already uploads per-run on success. This script is the
safety net: it walks every ``rpx_results/<Model>/<split>/`` directory,
reads ``result.json`` to learn the task, and mirrors the whole tree to
``<box_folder_id>/<task>/<Model>/<split>/`` exactly as the runner would.

Usage
-----
    # Mirror every run found under ./rpx_results
    PYTHONPATH=. python scripts/sync_results_to_box.py

    # Different local root, different Box folder
    PYTHONPATH=. python scripts/sync_results_to_box.py \\
        --local-root /path/to/results \\
        --box-folder-id 380510613151

    # Filter to one task or one model
    PYTHONPATH=. python scripts/sync_results_to_box.py --task relative_pose
    PYTHONPATH=. python scripts/sync_results_to_box.py --model Reloc3r-512

Requires
--------
``BOX_DEVELOPER_TOKEN`` exported in the environment. Tokens expire
every 60 minutes — refresh at
https://app.box.com/developers/console before launching a long sync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make scripts/ importable for box_fetch.upload_tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _human_bytes(n: int | float) -> str:
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _read_task(result_json: Path) -> str | None:
    """Read the ``task`` field out of a per-run result.json. Returns None
    if the file is missing or unparseable — caller decides whether to
    skip or fall back."""
    try:
        with result_json.open("r", encoding="utf-8") as f:
            return json.load(f).get("task")
    except (OSError, json.JSONDecodeError):
        return None


def _discover_runs(local_root: Path) -> list[tuple[Path, str, str, str]]:
    """Walk ``<local_root>/<Model>/<split>/`` two-deep.

    Returns ``[(out_dir, task, model, split), ...]`` for every directory
    that contains a parseable ``result.json``. Directories without a
    result.json are skipped with a notice.
    """
    runs: list[tuple[Path, str, str, str]] = []
    if not local_root.is_dir():
        raise FileNotFoundError(f"local-root not found: {local_root}")

    for model_dir in sorted(p for p in local_root.iterdir() if p.is_dir()):
        # Skip the sweep aggregator's output dir.
        if model_dir.name == "_sweep":
            continue
        for split_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            rj = split_dir / "result.json"
            if not rj.is_file():
                print(f"[skip] {split_dir.relative_to(local_root)}: no result.json")
                continue
            task = _read_task(rj)
            if task is None:
                print(
                    f"[skip] {split_dir.relative_to(local_root)}: "
                    "result.json missing or unparseable 'task' field"
                )
                continue
            runs.append((split_dir, task, model_dir.name, split_dir.name))
    return runs


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--local-root",
        type=Path,
        default=Path("rpx_results"),
        help="local results root (default: ./rpx_results)",
    )
    ap.add_argument(
        "--box-folder-id",
        default="380510613151",
        help="Box folder id to root the upload under (default: team's RPX-Outputs)",
    )
    ap.add_argument(
        "--task",
        default=None,
        help="filter: only sync runs whose result.json task matches "
        "(e.g. relative_pose, monocular_depth)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="filter: only sync runs whose top-level directory matches this name",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="filter: only sync runs in this split (easy | medium | hard)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be uploaded without making any Box requests",
    )
    args = ap.parse_args()

    runs = _discover_runs(args.local_root.resolve())
    if args.task:
        runs = [r for r in runs if r[1] == args.task]
    if args.model:
        runs = [r for r in runs if r[2] == args.model]
    if args.split:
        runs = [r for r in runs if r[3] == args.split]

    if not runs:
        print(f"[box-sync] no runs to upload under {args.local_root}")
        return

    print(f"[box-sync] {len(runs)} run(s) to mirror:")
    for out_dir, task, model, split in runs:
        n_files = sum(1 for _ in out_dir.rglob("*") if _.is_file())
        total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        remote = f"{task}/{model}/{split}"
        print(
            f"  - {out_dir.relative_to(args.local_root.resolve())}  →  "
            f"box:{remote}  ({n_files} files, {_human_bytes(total_bytes)})"
        )

    if args.dry_run:
        print("\n[box-sync] dry run — exiting before any Box request.")
        return

    # Lazy import so --dry-run / --help work without BOX_DEVELOPER_TOKEN.
    from rpx_benchmark.box_upload import upload_run_dir

    totals = {"uploaded": 0, "skipped": 0, "bytes": 0, "runs_ok": 0, "runs_fail": 0}
    for out_dir, task, model, split in runs:
        remote = f"{task}/{model}/{split}"
        print(f"\n=== mirroring {out_dir} → box:{remote} ===")
        try:
            r = upload_run_dir(
                out_dir,
                task=task,
                model_name=model,
                split=split,
                root_folder_id=args.box_folder_id,
            )
        except Exception as e:  # noqa: BLE001
            # Don't let one bad run stop the rest. Common cause: 401 if
            # the token aged past 60 min mid-sync.
            print(f"  ! failed: {type(e).__name__}: {e}")
            totals["runs_fail"] += 1
            continue
        totals["uploaded"] += r["uploaded"]
        totals["skipped"] += r["skipped"]
        totals["bytes"] += r["bytes_uploaded"]
        totals["runs_ok"] += 1
        print(
            f"  ok: {r['uploaded']} uploaded ({_human_bytes(r['bytes_uploaded'])}), "
            f"{r['skipped']} skipped (already on Box)"
        )

    print("\n=== summary ===")
    print(f"  runs ok:       {totals['runs_ok']}")
    print(f"  runs failed:   {totals['runs_fail']}")
    print(f"  files uploaded:{totals['uploaded']}")
    print(f"  files skipped: {totals['skipped']}")
    print(f"  bytes:         {_human_bytes(totals['bytes'])}")


if __name__ == "__main__":
    _cli()

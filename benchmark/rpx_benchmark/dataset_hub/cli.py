"""``python -m rpx_benchmark.dataset_hub.cli`` — sub-commands for the dataset hub.

Available sub-commands::

    mock      --out PATH [--multi N] [--single M] [--frames F] [--size S]
    scan      PATH                                                 [--json]
    pack      --src PATH --staging PATH [--label-version V] [--overwrite]
    manifest  --src PATH --pack-staging PATH --staging PATH [--splits FILE]
    upload    --staging PATH --repo-id ORG/REPO [--dry-run] [--public]

The CLI uses argparse + sub-parsers so we can grow it without breaking
existing invocations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .croissant import CroissantPatch, stage_croissant
from .dataset_card import CardSpec, write_dataset_card
from .downloader import download_for_task
from .lossless_convert import ConvertSpec, convert_capture_tree
from .manifest import build_frame_manifest
from .mock import MockSpec, generate_mock, measure_tree
from .packer import PackPlan, pack_capture_tree, pack_objects_meta
from .recipes import DEFAULT_REPO_ID, SceneType
from .scanner import scan_capture_root
from .staging import load_scene_splits, stage_splits
from .uploader import UploadPlan, upload_staging

# --------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------- #


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:6.1f} {unit}" if unit != "B" else f"{n:6d} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------- #
# `mock`
# --------------------------------------------------------------------- #


def _cmd_mock(args: argparse.Namespace) -> int:
    spec = MockSpec(
        multi_object_scenes=args.multi,
        single_object_scenes=args.single,
        frames_per_phase=args.frames,
        image_size=args.size,
    )
    out = generate_mock(Path(args.out), spec)
    files, total = measure_tree(out)
    print(f"[mock] wrote {files} files ({_human_bytes(total)}) under {out}")
    return 0


# --------------------------------------------------------------------- #
# `scan`
# --------------------------------------------------------------------- #


def _cmd_scan(args: argparse.Namespace) -> int:
    res = scan_capture_root(Path(args.root))

    if args.json:
        payload = {
            "root": str(res.root),
            "totals": {
                "scenes": len(res.scenes),
                "files": res.file_count,
                "bytes": res.total_bytes,
            },
            "by_type": {
                scene_type.value: {
                    "scenes": len(res.by_type(scene_type)),
                    "bytes": sum(s.total_bytes for s in res.by_type(scene_type)),
                }
                for scene_type in SceneType
            },
            "modalities": {
                name: {"files": inv.file_count, "bytes": inv.total_bytes}
                for name, inv in res.modality_totals().items()
            },
            "skipped": [str(p) for p in res.skipped],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"\n📂  {res.root}")
    print(
        f"    {len(res.scenes)} scenes, {res.file_count:,} files, "
        f"{_human_bytes(res.total_bytes)} total\n"
    )

    for scene_type in SceneType:
        subset = res.by_type(scene_type)
        if not subset:
            continue
        bytes_ = sum(s.total_bytes for s in subset)
        files_ = sum(s.file_count for s in subset)
        print(
            f"  · {scene_type.value:14s}  {len(subset):4d} scenes  "
            f"{files_:>9,} files  {_human_bytes(bytes_)}"
        )

    print("\n  per-modality totals (across all scenes & phases):")
    for name, inv in res.modality_totals().items():
        print(f"    {name:24s}  {inv.file_count:>9,} files  {_human_bytes(inv.total_bytes)}")

    if res.skipped:
        print(f"\n  ⚠ skipped {len(res.skipped)} non-matching entries:")
        for p in res.skipped[:5]:
            print(f"    - {p.name}")
        if len(res.skipped) > 5:
            print(f"    ... and {len(res.skipped) - 5} more")
    print()
    return 0


# --------------------------------------------------------------------- #
# `pack`
# --------------------------------------------------------------------- #


def _cmd_pack(args: argparse.Namespace) -> int:
    src = Path(args.src)
    staging = Path(args.staging)
    scan = scan_capture_root(src)
    plan = PackPlan(
        src_root=src,
        staging_root=staging,
        label_version=args.label_version,
        overwrite=args.overwrite,
    )
    res = pack_capture_tree(plan, scan)
    print(
        f"[pack] wrote {len(res.shards)} tar shards "
        f"({_human_bytes(res.total_bytes)} total) under {staging}"
    )
    if res.skipped_keys:
        print(
            f"[pack] skipped {len(res.skipped_keys)} unknown modality keys "
            f"(first 3: {res.skipped_keys[:3]})"
        )
    artefacts = pack_objects_meta(plan, scan)
    print(f"[pack] wrote {len(artefacts)} per-object questionnaires under {staging}/objects_meta/")
    return 0


# --------------------------------------------------------------------- #
# `manifest`
# --------------------------------------------------------------------- #


def _cmd_manifest(args: argparse.Namespace) -> int:
    src = Path(args.src)
    staging = Path(args.staging)
    scan = scan_capture_root(src)
    # Re-pack-walk: rebuild PackResult by mirroring what the staging dir
    # currently has on disk. We re-derive shard metadata from the tars.
    pack = _rehydrate_pack_result(scan, staging)

    splits = {}
    if args.splits:
        splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        # Accept both a flat {scene_id: tier} dict and the
        # scene_splits.json shape {"splits": {"easy": [...], ...}}.
        if "splits" in splits and isinstance(splits["splits"], dict):
            splits = {sid: tier for tier, ids in splits["splits"].items() for sid in ids}

    paths = build_frame_manifest(scan, pack, staging, splits=splits)
    print(f"[manifest] wrote {paths.parquet_path}")
    print(f"[manifest] wrote {paths.current_json_path}")

    # Per-task per-split JSONs the toolkit's reader (`download_split`)
    # expects at ``manifests/<task>/<split>.json``. Without these, a
    # freshly published HF dataset 404s on every benchmark call.
    from .split_manifests import write_split_manifests  # local import to avoid cycles

    written = write_split_manifests(staging)
    if not written:
        # Loud failure: the published dataset would be unusable.
        # Most common cause: ``--splits`` was not passed, so every row
        # in the parquet has ``split=None`` and no per-task JSONs are
        # produced. The hub README's canonical sequence requires
        # ``--splits benchmark/data/splits/scene_splits.json``.
        from ..exceptions import DatasetError

        raise DatasetError(
            "no per-task per-split manifests written — published HF tree "
            "would be unusable for `download_split`. Most likely cause: "
            "the --splits flag was not passed. Re-run with: "
            "`--splits benchmark/data/splits/scene_splits.json` (or "
            "another scene-tier mapping).",
            hint="If you intentionally want to skip split-manifests for "
            "a partial scene set, pass --splits with at least one "
            "scene mapped to a tier.",
        )
    print(
        f"[manifest] wrote {len(written)} per-task per-split manifests under {staging}/manifests/"
    )
    return 0


def _rehydrate_pack_result(scan, staging):
    """Best-effort reconstruction of PackResult from the staging dir.

    The packer writes deterministic tars; we just re-read tar member
    counts and sizes here so the manifest builder can populate its
    has_<modality> / shard_<modality> columns. Hashes are not
    recomputed — the manifest does not store them, and re-hashing 800
    GB of tars would be prohibitively slow at manifest time.
    """
    import tarfile

    from .packer import PackedShard, PackResult  # local import to avoid cycles

    shards = []
    for scene in scan.scenes:
        for phase in scene.phases:
            scene_root = "scenes" if scene.scene_type.value == "multi_object" else "objects"
            base = Path(staging) / scene_root / scene.scene_id / str(phase.phase_index)
            for tar_path in sorted(base.rglob("*.tar")) if base.is_dir() else []:
                try:
                    with tarfile.open(tar_path, "r") as tf:
                        members = tf.getmembers()
                        fc = sum(1 for m in members if m.isfile())
                except tarfile.TarError:
                    fc = 0
                rel = tar_path.relative_to(staging).as_posix()
                # Modality name: rgb.tar -> "rgb"; labels/masks/v1.tar -> "masks"
                if "/labels/" in rel:
                    modality = rel.split("/labels/", 1)[1].split("/", 1)[0]
                    is_label = True
                else:
                    modality = tar_path.stem
                    is_label = False
                shards.append(
                    PackedShard(
                        repo_path=rel,
                        scene_id=scene.scene_id,
                        phase=phase.phase_index,
                        modality=modality,
                        is_label=is_label,
                        file_count=fc,
                        total_bytes=tar_path.stat().st_size,
                        sha256=None,
                    )
                )
    return PackResult(shards=shards)


# --------------------------------------------------------------------- #
# `upload`
# --------------------------------------------------------------------- #


def _cmd_upload(args: argparse.Namespace) -> int:
    plan = UploadPlan(
        staging_root=Path(args.staging),
        repo_id=args.repo_id,
        revision=args.revision,
        private=not args.public,
        dry_run=args.dry_run,
        commit_message=args.message,
    )
    res = upload_staging(plan)
    if res.dry_run if hasattr(res, "dry_run") else args.dry_run:
        print(
            f"[upload-dry-run] would push {res.files_planned} files "
            f"({_human_bytes(res.bytes_planned)}) to {res.repo_id}"
        )
    else:
        print(
            f"[upload] pushed {res.files_planned} files "
            f"({_human_bytes(res.bytes_planned)}) to {res.repo_id}"
        )
        if res.commit_url:
            print(f"[upload] commit: {res.commit_url}")
    return 0


# --------------------------------------------------------------------- #
# `stage-splits`
# --------------------------------------------------------------------- #


def _cmd_stage_splits(args: argparse.Namespace) -> int:
    src = Path(args.splits_src) if args.splits_src else None
    staged = stage_splits(
        staging_root=Path(args.staging),
        splits_src=src,
        overwrite=args.overwrite,
        require_all=not args.allow_missing,
    )
    print(f"[stage-splits] copied {len(staged)} files into {args.staging}/splits/")
    for s in staged:
        print(f"               {s.repo_path}  ({_human_bytes(s.bytes_)})")
    return 0


# --------------------------------------------------------------------- #
# `dataset-card`
# --------------------------------------------------------------------- #


def _cmd_dataset_card(args: argparse.Namespace) -> int:
    src = Path(args.src)
    staging = Path(args.staging)
    scan = scan_capture_root(src)

    splits = None
    if (staging / "splits" / "scene_splits.json").is_file():
        try:
            splits = load_scene_splits(staging)
        except Exception as e:
            print(f"[dataset-card] WARN: could not load scene splits: {e}", file=sys.stderr)

    label_versions = None
    cur_path = staging / "manifest" / "current.json"
    if cur_path.is_file():
        label_versions = json.loads(cur_path.read_text("utf-8")).get(
            "label_versions",
        )

    spec = CardSpec(repo_id=args.repo_id)
    out = write_dataset_card(
        staging,
        scan,
        spec=spec,
        splits=splits,
        label_versions=label_versions,
        overwrite=args.overwrite,
    )
    print(f"[dataset-card] wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


# --------------------------------------------------------------------- #
# `lossless-convert`
# --------------------------------------------------------------------- #


def _cmd_lossless_convert(args: argparse.Namespace) -> int:
    spec = ConvertSpec(
        src_root=Path(args.src),
        out_root=Path(args.out),
        workers=args.workers,
        dry_run=args.dry_run,
        verify=not args.no_verify,
        overwrite_out=args.overwrite_out,
        webp_method=args.webp_method,
        png_compress_level=args.png_compress_level,
        skip_rgb_webp=args.skip_rgb_webp,
        skip_png_recompress=args.skip_png_recompress,
        skip_cam_pose=args.skip_cam_pose,
    )
    res = convert_capture_tree(spec)
    if spec.dry_run:
        print(
            f"[lossless-convert] DRY-RUN: would convert {res.files_converted:,} files, "
            f"link {res.files_linked:,} files (total {res.files_seen:,})"
        )
        for action, stats in sorted(res.by_action.items()):
            print(f"  {action:18s} {stats.files:>7,} files")
        return 0
    # Per-action savings breakdown
    for action, stats in sorted(res.by_action.items()):
        if stats.files == 0:
            continue
        print(
            f"[lossless-convert] {action:18s} "
            f"{stats.files:>7,} files  "
            f"{_human_bytes(stats.bytes_before)} → {_human_bytes(stats.bytes_after)}  "
            f"saved {_human_bytes(stats.saved_bytes)} ({stats.saved_pct:.1f}%)"
        )
    print(
        f"[lossless-convert] TOTAL: {res.files_seen:,} files  "
        f"{_human_bytes(res.bytes_before)} → {_human_bytes(res.bytes_after)}  "
        f"saved {_human_bytes(res.saved_bytes)} ({res.saved_pct:.1f}%)"
    )
    return 0


# --------------------------------------------------------------------- #
# `stage-croissant`
# --------------------------------------------------------------------- #


def _cmd_stage_croissant(args: argparse.Namespace) -> int:
    src = Path(args.src) if args.src else None
    patch = CroissantPatch(repo_id=args.repo_id, version=args.version, cite_as=args.cite_as)
    out = stage_croissant(
        staging_root=Path(args.staging), src=src, patch=patch, overwrite=args.overwrite
    )
    print(f"[stage-croissant] wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


# --------------------------------------------------------------------- #
# `download`
# --------------------------------------------------------------------- #


def _cmd_download(args: argparse.Namespace) -> int:
    extras = tuple(args.modalities.split(",")) if args.modalities else ()
    label_versions = {}
    for kv in args.label_version or ():
        k, _, v = kv.partition("=")
        if not k or not v:
            print(f"[download] ignoring malformed --label-version entry: {kv}", file=sys.stderr)
            continue
        label_versions[k] = v

    res = download_for_task(
        task=args.task,
        split=args.split,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        extra_modalities=extras,
        label_versions=label_versions,
    )
    print(f"[download] task={res.task} split={res.split} scene_type={res.scene_type.value}")
    print(
        f"[download] matched {len(res.matched_scenes)} scenes, {len(res.allow_patterns)} patterns"
    )
    print(f"[download] local_dir={res.local_dir}")
    print(f"[download] {res.files_fetched} files on disk ({_human_bytes(res.bytes_fetched)})")
    return 0


# --------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rpx_benchmark.dataset_hub.cli",
        description="RPX dataset-hub utilities (mock / scan).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mock = sub.add_parser("mock", help="Generate a synthetic capture tree.")
    p_mock.add_argument("--out", required=True, help="Output root (will be created).")
    p_mock.add_argument(
        "--multi", type=int, default=3, help="Number of multi-object scenes (default: 3)."
    )
    p_mock.add_argument(
        "--single", type=int, default=5, help="Number of single-object scenes (default: 5)."
    )
    p_mock.add_argument("--frames", type=int, default=4, help="Frames per phase (default: 4).")
    p_mock.add_argument(
        "--size", type=int, default=32, help="Image edge length in px (default: 32)."
    )
    p_mock.set_defaults(func=_cmd_mock)

    p_scan = sub.add_parser("scan", help="Inventory an on-disk capture tree.")
    p_scan.add_argument("root", help="Capture-tree root (test_dataset_aggregated/).")
    p_scan.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table."
    )
    p_scan.set_defaults(func=_cmd_scan)

    p_pack = sub.add_parser("pack", help="Pack a capture tree into HF tar shards.")
    p_pack.add_argument("--src", required=True, help="Capture-tree root.")
    p_pack.add_argument("--staging", required=True, help="Staging output dir.")
    p_pack.add_argument("--label-version", default="v1", help="Label version tag (default: v1).")
    p_pack.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing shards in the staging dir."
    )
    p_pack.set_defaults(func=_cmd_pack)

    p_man = sub.add_parser("manifest", help="Build the per-frame Parquet manifest.")
    p_man.add_argument("--src", required=True, help="Capture-tree root (for frame inventory).")
    p_man.add_argument(
        "--staging", required=True, help="Packed staging dir (must contain *.tar shards)."
    )
    p_man.add_argument(
        "--splits", default=None, help="Optional scene_splits.json path for split labels."
    )
    p_man.set_defaults(func=_cmd_manifest)

    p_up = sub.add_parser("upload", help="Push a staging dir to a HF dataset repo.")
    p_up.add_argument("--staging", required=True, help="Packed staging dir.")
    p_up.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    p_up.add_argument("--revision", default="main", help="HF revision/branch (default: main).")
    p_up.add_argument("--public", action="store_true", help="Create as public; default is private.")
    p_up.add_argument(
        "--dry-run", action="store_true", help="Walk the tree and report; no network call."
    )
    p_up.add_argument("--message", default="RPX dataset hub upload", help="Commit message.")
    p_up.set_defaults(func=_cmd_upload)

    p_dl = sub.add_parser("download", help="Pull just the tar shards a (task, split) needs.")
    p_dl.add_argument(
        "--task", required=True, help="Recipe key (e.g. segmentation, relative_pose)."
    )
    p_dl.add_argument("--split", default=None, help="easy | medium | hard (multi-object only).")
    p_dl.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    p_dl.add_argument("--revision", default=None, help="HF revision/branch/tag (default: main).")
    p_dl.add_argument(
        "--cache-dir", default=None, help="HF cache root (default: ~/.cache/huggingface)."
    )
    p_dl.add_argument(
        "--modalities", default="", help="Comma-separated extras beyond the recipe defaults."
    )
    p_dl.add_argument(
        "--label-version",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a label modality version (repeatable).",
    )
    p_dl.set_defaults(func=_cmd_download)

    p_st = sub.add_parser("stage-splits", help="Copy splits/* into a staging dir.")
    p_st.add_argument("--staging", required=True, help="Staging dir (the upload target).")
    p_st.add_argument(
        "--splits-src",
        default=None,
        help="Source dir holding scene_splits.json + tier txts (default: benchmark/data/splits/).",
    )
    p_st.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in <staging>/splits/."
    )
    p_st.add_argument(
        "--allow-missing",
        action="store_true",
        help="Don't raise if some expected splits files are absent in the source dir.",
    )
    p_st.set_defaults(func=_cmd_stage_splits)

    p_dc = sub.add_parser("dataset-card", help="Generate <staging>/README.md (HF dataset card).")
    p_dc.add_argument("--src", required=True, help="Capture-tree root (used for scan totals).")
    p_dc.add_argument(
        "--staging", required=True, help="Staging dir; the README lands at <staging>/README.md."
    )
    p_dc.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    p_dc.add_argument("--overwrite", action="store_true", help="Overwrite an existing README.md.")
    p_dc.set_defaults(func=_cmd_dataset_card)

    p_lc = sub.add_parser(
        "lossless-convert",
        help="Re-encode rgb/ and fisheye/ PNGs to lossless WebP (~25% smaller, bit-identical pixels).",
    )
    p_lc.add_argument("--src", required=True, help="Source capture-tree root.")
    p_lc.add_argument(
        "--out",
        required=True,
        help="Output capture-tree root; must not overlap --src.",
    )
    p_lc.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel encode workers (default: CPU count - 1).",
    )
    p_lc.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the tree and report planned counts without writing anything.",
    )
    p_lc.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the per-frame round-trip np.array_equal check (faster, not recommended).",
    )
    p_lc.add_argument(
        "--overwrite-out",
        action="store_true",
        help="Allow writing into a non-empty --out directory.",
    )
    p_lc.add_argument(
        "--webp-method",
        type=int,
        default=4,
        choices=range(0, 7),
        metavar="{0..6}",
        help=(
            "libwebp lossless effort level: 0=fastest/largest, 6=slowest/smallest. "
            "Default 4: empirically the sweet spot on photo content (method=6 takes "
            "10x longer for ~0%% extra savings on natural images)."
        ),
    )
    p_lc.add_argument(
        "--png-compress-level",
        type=int,
        default=9,
        choices=range(0, 10),
        metavar="{0..9}",
        help=(
            "PNG re-encode compression level (0-9). Default 9: densest, still "
            "lossless. Applied to depth/* and sam2/masks/* PNGs."
        ),
    )
    p_lc.add_argument(
        "--skip-rgb-webp",
        action="store_true",
        help="Skip the rgb/fisheye/ego PNG → WebP-lossless pass.",
    )
    p_lc.add_argument(
        "--skip-png-recompress",
        action="store_true",
        help="Skip the depth/masks PNG → PNG-level=9 re-encode pass.",
    )
    p_lc.add_argument(
        "--skip-cam-pose",
        action="store_true",
        help="Skip the cam_pose .npz → per-frame .npy consolidation pass.",
    )
    p_lc.set_defaults(func=_cmd_lossless_convert)

    p_cr = sub.add_parser("stage-croissant", help="Copy + patch the Croissant JSON.")
    p_cr.add_argument(
        "--staging", required=True, help="Staging dir; lands at <staging>/rpx_croissant.json."
    )
    p_cr.add_argument(
        "--src",
        default=None,
        help="Source JSON path (default: <package>/dataset_hub/croissant_template.json).",
    )
    p_cr.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id for url field (default: {DEFAULT_REPO_ID}).",
    )
    p_cr.add_argument("--version", default="1.0.0", help="Dataset version string (default: 1.0.0).")
    p_cr.add_argument(
        "--cite-as", default=None, help="Replace the citeAs field (e.g. a bibtex string)."
    )
    p_cr.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing rpx_croissant.json."
    )
    p_cr.set_defaults(func=_cmd_stage_croissant)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from rpx_benchmark.cleanup import install_signal_cleanup

    install_signal_cleanup()

    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

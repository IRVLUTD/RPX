#----------------------------------------------------------------------------------------------------
# Batch ego (GoPro) frame extraction across the full RPX 100-scene dataset.
#
#   python -m maskgen_pipeline.batch_extract_ego --scenes_root /path/to/scenes
#
# For each scene directory under --scenes_root, locate the ego mp4 (in the
# scene dir or under --ego_root by scene id), uniformly extract exactly
# EGO_SAMPLE_COUNT (250) frames, and save them as <scene>/ego/rgb/NNNNN.png.
#
# The ego camera and the D435 rig run on independent clocks; we do NOT attempt
# any temporal sync. See ego_sample.py and the paper appendix for the rationale.
#
# Per-scene failures do not stop the batch — the script keeps going, prints a
# per-scene status line, and prints an aggregated summary at the end.
#
# Work done while being at the Intelligent Robotics and Vision Lab at UT Dallas.
#----------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .annotate_ego import _resolve_ego_video
from .ego_sample import (
    EGO_IMAGE_FORMAT,
    EGO_JPEG_QUALITY,
    EGO_SAMPLE_COUNT,
    sample_ego_frames,
)


# --------------------------------------------------------------------------- #
# Per-scene result + the work function
# --------------------------------------------------------------------------- #

@dataclass
class SceneResult:
    scene_id: str
    scene_dir: Path
    ego_video: Optional[Path]
    status: str               # "ok" | "skip" | "no_ego" | "short" | "error"
    n_written: int = 0
    requested: int = 0
    elapsed_s: float = 0.0
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "short", "skip")


def _already_done(scene_dir: Path, requested: int) -> bool:
    """A scene is considered done if its ego/rgb/ holds ≥1 image AND the frame
    map exists. We don't require the count to be exactly ``requested`` because
    a shorter source mp4 legitimately yields fewer frames."""
    rgb = scene_dir / "ego" / "rgb"
    fmap = scene_dir / "ego" / "ego_frame_map.json"
    if not rgb.is_dir() or not fmap.is_file():
        return False
    return any(rgb.glob("*.png")) or any(rgb.glob("*.jpg"))


def _ego_for_scene(scene_dir: Path, ego_root: Optional[Path]) -> Optional[Path]:
    """Locate the ego mp4 for one scene. Prefer a file inside the scene dir;
    fall back to ``<ego_root>/<scene_id>.{mp4,MP4}`` if --ego_root is given."""
    in_scene = _resolve_ego_video(scene_dir)
    if in_scene is not None:
        return in_scene
    if ego_root is not None:
        for ext in (".mp4", ".MP4"):
            cand = ego_root / f"{scene_dir.name}{ext}"
            if cand.is_file():
                return cand
        # Case-insensitive last-ditch glob in case the case doesn't match.
        for cand in ego_root.glob(f"{scene_dir.name}.[mM][pP]4"):
            if cand.is_file():
                return cand
    return None


def _process_one(
    scene_dir_str: str,
    ego_root_str: Optional[str],
    n_samples: int,
    force: bool,
    image_format: str,
    jpeg_quality: int,
) -> SceneResult:
    """Pure work function: safe to call from a worker process."""
    scene_dir = Path(scene_dir_str)
    ego_root = Path(ego_root_str) if ego_root_str else None
    t0 = time.perf_counter()
    res = SceneResult(scene_id=scene_dir.name, scene_dir=scene_dir,
                      ego_video=None, status="error", requested=n_samples)
    try:
        if not force and _already_done(scene_dir, n_samples):
            res.status = "skip"
            res.message = "ego/rgb/ already populated; pass --force to re-extract"
            rgb = scene_dir / "ego" / "rgb"
            res.n_written = len(list(rgb.glob("*.png"))) + len(list(rgb.glob("*.jpg")))
            res.elapsed_s = time.perf_counter() - t0
            return res

        ego = _ego_for_scene(scene_dir, ego_root)
        if ego is None:
            res.status = "no_ego"
            res.message = "no ego mp4 found in scene dir or under --ego_root"
            res.elapsed_s = time.perf_counter() - t0
            return res
        res.ego_video = ego

        written = sample_ego_frames(
            scene_dir, ego, n_samples=n_samples,
            image_format=image_format, jpeg_quality=jpeg_quality,
        )
        res.n_written = len(written)
        res.status = "ok" if res.n_written == n_samples else "short"
        if res.status == "short":
            res.message = (
                f"requested {n_samples} but source mp4 yielded only {res.n_written}; "
                "frames are unique, no duplicates"
            )
    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.message = f"{type(e).__name__}: {e}"
        if "RPX_BATCH_TRACEBACK" in __import__("os").environ:
            res.message += "\n" + traceback.format_exc()
    res.elapsed_s = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------- #
# Discovery + summary
# --------------------------------------------------------------------------- #

def _discover_scene_dirs(scenes_root: Path, depth: int = 1) -> List[Path]:
    """Discover scene directories under ``scenes_root``.

    ``depth=1`` (default): every immediate subdirectory is a scene
        — matches ``<root>/<scene_id>/`` layouts.
    ``depth=2``: every grand-child subdirectory is a scene
        — matches batch-grouped layouts like ``<root>/<batch>/<scene_id>/``
        (e.g. ``test_gopro_organized-selected/1-5/1/``).
    ``depth=0``: auto-detect — pick the depth at which the most directories
        contain an mp4 file directly.
    """
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"--scenes_root not found: {scenes_root}")

    def _children_at(d: int):
        if d == 1:
            return [p for p in scenes_root.iterdir()
                    if p.is_dir() and not p.name.startswith(".")]
        if d == 2:
            return [p for p in scenes_root.glob("*/*")
                    if p.is_dir() and not p.name.startswith(".")]
        raise ValueError(f"depth must be 1 or 2, got {d}")

    if depth in (1, 2):
        return sorted(_children_at(depth), key=lambda p: (p.parent.name, p.name))

    # depth == 0: pick whichever level has more mp4-bearing children.
    def _mp4_count(dirs):
        return sum(1 for p in dirs if any(p.glob("*.[mM][pP]4")))
    d1, d2 = _children_at(1), _children_at(2)
    chosen = d2 if _mp4_count(d2) > _mp4_count(d1) else d1
    return sorted(chosen, key=lambda p: (p.parent.name, p.name))


def _print_line(res: SceneResult) -> None:
    sym = {"ok": "✓", "short": "△", "skip": "·", "no_ego": "✗", "error": "✗"}[res.status]
    ego = res.ego_video.name if res.ego_video else "—"
    msg = f"  {res.message}" if res.message else ""
    print(
        f"  [{sym}] {res.scene_id:<28s} "
        f"{res.status:<6s} {res.n_written:>4d}/{res.requested:<4d} "
        f"{res.elapsed_s:6.1f}s  ego={ego}{msg}",
        flush=True,
    )


def _print_summary(results: Sequence[SceneResult], n_samples: int) -> None:
    by = {k: [r for r in results if r.status == k] for k in ("ok", "short", "skip", "no_ego", "error")}
    total_frames = sum(r.n_written for r in results)
    total_time = sum(r.elapsed_s for r in results)
    print()
    print("=" * 72)
    print("  BATCH EGO EXTRACTION — SUMMARY")
    print("=" * 72)
    print(f"  scenes total       : {len(results)}")
    print(f"  ok (exact {n_samples:>3d})    : {len(by['ok'])}")
    print(f"  short window (<{n_samples}) : {len(by['short'])}")
    print(f"  skipped (already done): {len(by['skip'])}")
    print(f"  no ego mp4         : {len(by['no_ego'])}")
    print(f"  errors             : {len(by['error'])}")
    print(f"  total frames written: {total_frames}")
    print(f"  total wall time    : {total_time:.1f}s")
    if by["no_ego"]:
        print()
        print("  Scenes missing an ego mp4:")
        for r in by["no_ego"]:
            print(f"    - {r.scene_id}")
    if by["error"]:
        print()
        print("  Scenes that errored:")
        for r in by["error"]:
            print(f"    - {r.scene_id}: {r.message.splitlines()[0]}")
    if by["short"]:
        print()
        print(f"  Scenes whose source mp4 was shorter than {n_samples} frames:")
        for r in by["short"]:
            print(f"    - {r.scene_id}: got {r.n_written}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch-extract a fixed number of evenly-spaced frames from "
                    "each scene's ego (GoPro) mp4 across the full RPX dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes_root", type=Path, default=None,
                     help="Directory whose subdirectories (1 or 2 levels deep, see --depth) are scenes.")
    src.add_argument("--scene_dirs", type=Path, nargs="+", default=None,
                     help="Explicit list of scene directories to process.")
    ap.add_argument("--depth", type=int, default=0, choices=(0, 1, 2),
                    help="Discovery depth under --scenes_root: 1 = immediate subdirs (default scene layout); "
                         "2 = grand-child subdirs (batch-grouped layouts like 1-5/1/, 1-5/2/, ...); "
                         "0 = auto-detect (default).")
    ap.add_argument("--ego_root", type=Path, default=None,
                    help="Fallback directory of ego mp4s named <scene_id>.mp4 "
                         "for scenes where the mp4 doesn't live inside the scene dir.")
    ap.add_argument("--n_samples", type=int, default=EGO_SAMPLE_COUNT,
                    help="Frames to extract per ego mp4.")
    ap.add_argument("--format", dest="image_format", default=EGO_IMAGE_FORMAT,
                    choices=("jpg", "png"),
                    help="Output image format. 'jpg' (default, q=95, ~5x smaller than PNG on 1080p; "
                         "matches interactive_gsam2's *.jpg glob) or 'png' (lossless).")
    ap.add_argument("--jpeg_quality", type=int, default=EGO_JPEG_QUALITY,
                    help="JPEG quality (1-100). Ignored when --format=png.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Process pool size. 1 = sequential.")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if ego/rgb/ already contains frames.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.scenes_root is not None:
        try:
            scene_dirs = _discover_scene_dirs(args.scenes_root, depth=args.depth)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        scene_dirs = [Path(p) for p in args.scene_dirs]
        bad = [p for p in scene_dirs if not p.is_dir()]
        if bad:
            for p in bad:
                print(f"error: scene dir not found: {p}", file=sys.stderr)
            return 2

    if not scene_dirs:
        print("error: no scenes discovered", file=sys.stderr)
        return 2

    n = args.n_samples
    print(f"[batch-ego] {len(scene_dirs)} scenes  ·  {n} frames per scene  ·  "
          f"workers={args.workers}  ·  force={args.force}")
    if args.ego_root is not None:
        print(f"[batch-ego] ego_root fallback: {args.ego_root}")
    print()

    ego_root_str = str(args.ego_root) if args.ego_root else None
    results: List[SceneResult] = []

    if args.workers <= 1:
        for sd in scene_dirs:
            res = _process_one(str(sd), ego_root_str,
                               n, args.force, args.image_format, args.jpeg_quality)
            _print_line(res)
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process_one, str(sd), ego_root_str,
                                n, args.force, args.image_format, args.jpeg_quality): sd
                    for sd in scene_dirs}
            for fut in as_completed(futs):
                res = fut.result()
                _print_line(res)
                results.append(res)
        # Preserve discovery order in the final summary.
        order = {str(sd): i for i, sd in enumerate(scene_dirs)}
        results.sort(key=lambda r: order.get(str(r.scene_dir), 1 << 30))

    _print_summary(results, n)

    n_fail = sum(1 for r in results if not r.ok)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

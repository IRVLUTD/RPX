#----------------------------------------------------------------------------------------------------
# Extract ego (GoPro) frames from the Box-downloaded folder structure
#
#   <source>/<batch>/<scene_id>/GX010NNN.MP4
#       e.g. test_gopro_organized-selected/1-5/2/GX010761.MP4
#
# and lay them out in the canonical RPX scene structure that mirrors the
# rig-data convention (\\scene{N}/{0,1,2,ego}/...), so the ego/ folder slots
# next to the three D435 phase dirs without any rewiring downstream:
#
#   <output>/scene<NNN>/ego/<NNNNN>.png        (NNN = 001..100, NNNNN = 00000..00249)
#   <output>/scene_manifest.csv                 (full source → output mapping + status)
#
# Conventions
#   - Output frames are lossless PNG (chosen for team distribution; switch with
#     --format jpg if you need the labeller-friendly JPEG q=95 instead).
#   - Subdir names in the source tree are the scene numbers ("1", "2", ..., "100").
#   - Multi-take scenes use decimal suffixes ("25.1", "25.2", ...). By default the
#     FIRST take (.1) becomes the canonical scene<NNN>; the extra takes are still
#     recorded in the manifest but not used as the primary output. Override with
#     --multi_take_policy {first,longest}.
#
#----------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ego_sample import EGO_SAMPLE_COUNT, uniform_sample_indices


# --------------------------------------------------------------------------- #
# Per-scene record
# --------------------------------------------------------------------------- #

@dataclass
class SceneRecord:
    canonical: str                  # "scene001"
    scene_number: int               # 1..100
    take: Optional[str]             # None for single-take; "1", "2", "3" for multi-take
    source_dir: str                 # absolute path
    source_mp4: str                 # absolute path
    used_as_primary: bool           # True for the take that became <output>/scene<NNN>/ego/
    n_frames_written: int = 0
    n_frames_in_source: int = 0
    status: str = ""                # "ok" | "short" | "skip" | "error" | "duplicate"
    message: str = ""


# --------------------------------------------------------------------------- #
# Source discovery + scene-number parsing
# --------------------------------------------------------------------------- #

_SCENE_DIR_RE = re.compile(r"^(\d+)(?:\.(\d+))?$")  # "1", "25", "25.1", "25.2"


def _parse_scene_dir_name(name: str) -> Optional[Tuple[int, Optional[str]]]:
    """``"1" -> (1, None)``, ``"25.1" -> (25, "1")``, junk -> ``None``."""
    m = _SCENE_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def discover_sources(source_root: Path) -> List[Tuple[Path, int, Optional[str]]]:
    """Return ``(scene_dir, scene_number, take)`` for every depth-2 child of
    ``source_root`` whose name parses as a scene index and that contains an mp4.
    """
    out: List[Tuple[Path, int, Optional[str]]] = []
    for batch in sorted(source_root.iterdir()):
        if not batch.is_dir() or batch.name.startswith("."):
            continue
        for sd in sorted(batch.iterdir()):
            if not sd.is_dir() or sd.name.startswith("."):
                continue
            parsed = _parse_scene_dir_name(sd.name)
            if parsed is None:
                continue
            if not any(sd.glob("*.[mM][pP]4")):
                continue
            num, take = parsed
            out.append((sd, num, take))
    return out


def group_takes(found: List[Tuple[Path, int, Optional[str]]]) -> Dict[int, List[Tuple[Path, Optional[str]]]]:
    """``{ scene_number: [(dir, take), ...] }`` sorted by take."""
    by_scene: Dict[int, List[Tuple[Path, Optional[str]]]] = {}
    for sd, num, take in found:
        by_scene.setdefault(num, []).append((sd, take))
    for num in by_scene:
        # Sort: single-take (None) first, otherwise numeric ascending.
        by_scene[num].sort(key=lambda t: (1 if t[1] is None else 0, int(t[1] or 0)))
    return by_scene


def _pick_primary(takes: List[Tuple[Path, Optional[str]]], policy: str) -> int:
    """Index into ``takes`` of the primary take under ``policy``."""
    if len(takes) == 1:
        return 0
    if policy == "first":
        return 0
    if policy == "longest":
        import cv2  # noqa: PLC0415
        best_i, best_n = 0, -1
        for i, (sd, _) in enumerate(takes):
            mp4 = sorted(sd.glob("*.[mM][pP]4"))[0]
            cap = cv2.VideoCapture(str(mp4))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
            cap.release()
            if n > best_n:
                best_i, best_n = i, n
        return best_i
    raise ValueError(f"unknown multi_take_policy: {policy!r}")


# --------------------------------------------------------------------------- #
# Extraction (one mp4)
# --------------------------------------------------------------------------- #

def _extract_one(
    mp4: Path,
    out_dir: Path,
    n_samples: int,
    image_format: str,
    jpeg_quality: int,
) -> Tuple[int, int, str]:
    """Returns ``(n_written, n_in_source, message)``."""
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return 0, 0, f"could not open {mp4}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0, 0, f"{mp4.name} reports {total} frames"

    idxs = uniform_sample_indices(0, total - 1, n_samples)
    want = set(idxs)
    out_dir.mkdir(parents=True, exist_ok=True)

    if image_format == "jpg":
        ext, encode_params = "jpg", [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    else:
        ext, encode_params = "png", [cv2.IMWRITE_PNG_COMPRESSION, 3]

    written, src_frames = [], []
    local = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if local in want:
            out_path = out_dir / f"{len(written):05d}.{ext}"
            cv2.imwrite(str(out_path), frame, encode_params)
            written.append(out_path.name)
            src_frames.append(local)
        local += 1
    cap.release()

    # Provenance sidecar inside the ego/ folder.
    (out_dir / "ego_frame_map.json").write_text(json.dumps({
        "source_mp4": str(mp4),
        "ego_total_frames": total,
        "n_samples_requested": n_samples,
        "image_format": ext,
        "jpeg_quality": jpeg_quality if ext == "jpg" else None,
        "frames": {name: src_frames[i] for i, name in enumerate(written)},
    }, indent=2))

    msg = ""
    if len(written) < n_samples:
        msg = (f"source mp4 had only {total} frames; took all {len(written)} "
               f"(requested {n_samples}, no duplicates)")
    return len(written), total, msg


# --------------------------------------------------------------------------- #
# Worker function (process-pool friendly)
# --------------------------------------------------------------------------- #

def _process(record_dict: dict, n_samples: int, fmt: str, q: int, force: bool) -> dict:
    """Pure function executed by a worker; takes/returns dicts so it pickles."""
    t0 = time.perf_counter()
    out_dir = Path(record_dict.pop("out_dir"))
    rec = SceneRecord(**record_dict)
    rec.status = "error"
    try:
        # Honour --force semantics by re-extracting; otherwise skip if non-empty.
        existing = sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.jpg"))
        if existing and not force:
            rec.status = "skip"
            rec.n_frames_written = len(existing)
            rec.message = "out dir already populated; pass --force to re-extract"
            return {**asdict(rec), "elapsed_s": time.perf_counter() - t0}

        mp4 = Path(rec.source_mp4)
        n_written, n_src, msg = _extract_one(mp4, out_dir, n_samples, fmt, q)
        rec.n_frames_written = n_written
        rec.n_frames_in_source = n_src
        rec.message = msg
        rec.status = "ok" if n_written == n_samples else "short"
    except Exception as e:  # noqa: BLE001
        rec.status = "error"
        rec.message = f"{type(e).__name__}: {e}"
    return {**asdict(rec), "elapsed_s": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def write_manifest(records: List[SceneRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "canonical", "scene_number", "take", "used_as_primary",
            "source_dir", "source_mp4",
            "n_frames_written", "n_frames_in_source",
            "status", "message",
        ])
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in w.fieldnames})


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _print_line(r: SceneRecord, elapsed: float) -> None:
    sym = {"ok": "✓", "short": "△", "skip": "·",
           "duplicate": "d", "error": "✗"}.get(r.status, "?")
    take = f"take={r.take}" if r.take else "single"
    primary = "" if r.used_as_primary else "  (extra take, not primary)"
    msg = f"  {r.message}" if r.message else ""
    print(f"  [{sym}] {r.canonical}  {take:<7s}  "
          f"{r.n_frames_written:>4d}/{r.n_frames_in_source:<5d}  "
          f"{elapsed:6.1f}s  src={Path(r.source_mp4).name}{primary}{msg}",
          flush=True)


def _print_summary(records: List[SceneRecord], n_samples: int, out_root: Path) -> None:
    primary = [r for r in records if r.used_as_primary]
    by = {k: [r for r in primary if r.status == k]
          for k in ("ok", "short", "skip", "duplicate", "error")}
    extras = [r for r in records if not r.used_as_primary]
    total_frames = sum(r.n_frames_written for r in records)
    print()
    print("=" * 74)
    print("  EGO → RPX-LAYOUT EXTRACTION — SUMMARY")
    print("=" * 74)
    print(f"  output root          : {out_root}")
    print(f"  primary scenes total : {len(primary)}")
    print(f"  ok (exact {n_samples:>3d})       : {len(by['ok'])}")
    print(f"  short (<{n_samples})          : {len(by['short'])}")
    print(f"  skipped (already done): {len(by['skip'])}")
    print(f"  errors               : {len(by['error'])}")
    print(f"  extra takes recorded : {len(extras)}  (not written as primary)")
    print(f"  total frames written : {total_frames}")
    if by["short"]:
        print()
        print(f"  Scenes with mp4 shorter than {n_samples} frames:")
        for r in by["short"]:
            print(f"    - {r.canonical}: got {r.n_frames_written} from {r.source_mp4}")
    if by["error"]:
        print()
        print("  Scenes that errored:")
        for r in by["error"]:
            print(f"    - {r.canonical}: {r.message.splitlines()[0]}")
    if extras:
        print()
        print(f"  Multi-take scenes ({len({r.scene_number for r in extras})} unique scene numbers, extras not used as primary):")
        for r in extras:
            print(f"    - scene{r.scene_number:03d} take={r.take}  ({r.source_mp4})")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract 250 ego (GoPro) frames per scene from the Box-downloaded "
                    "test_gopro_organized-selected/ tree into the canonical RPX scene "
                    "layout (scene<NNN>/ego/<NNNNN>.png).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source_root", type=Path, required=True,
                    help="The Box-downloaded folder (depth-2 scene dirs under batch dirs).")
    ap.add_argument("--output_root", type=Path, required=True,
                    help="Where to write scene<NNN>/ego/ frames.")
    ap.add_argument("--n_samples", type=int, default=EGO_SAMPLE_COUNT,
                    help="Frames to extract per ego mp4.")
    ap.add_argument("--format", dest="image_format", default="png",
                    choices=("png", "jpg"),
                    help="Output image format. 'png' (default, lossless) or 'jpg' (q=95, ~5x smaller).")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                    help="JPEG quality (1-100). Ignored when --format=png.")
    ap.add_argument("--multi_take_policy", choices=("first", "longest"), default="first",
                    help="When a scene has multiple takes (25.1, 25.2, ...), which one "
                         "becomes the primary scene<NNN>/ego/?")
    ap.add_argument("--workers", type=int, default=1,
                    help="Process pool size. 1 = sequential.")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if output ego/ already contains frames.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.source_root.is_dir():
        print(f"error: --source_root not found: {args.source_root}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)

    found = discover_sources(args.source_root)
    if not found:
        print("error: no scene dirs (depth-2) with mp4s found", file=sys.stderr)
        return 2

    grouped = group_takes(found)
    print(f"[ego→rpx] {len(found)} mp4s in {len(grouped)} unique scenes "
          f"({sum(1 for v in grouped.values() if len(v) > 1)} multi-take scenes)  ·  "
          f"format={args.image_format}  ·  workers={args.workers}")

    # Build the work list: one primary per scene + every extra take is recorded
    # in the manifest (but not written as the primary).
    records: List[Tuple[SceneRecord, Path, bool]] = []  # (rec, out_dir, do_extract)
    for num in sorted(grouped):
        takes = grouped[num]
        primary_idx = _pick_primary(takes, args.multi_take_policy)
        for i, (sd, take) in enumerate(takes):
            mp4 = sorted(sd.glob("*.[mM][pP]4"))[0]
            is_primary = (i == primary_idx)
            canonical = f"scene{num:03d}"
            out_dir = args.output_root / canonical / "ego"
            rec = SceneRecord(
                canonical=canonical, scene_number=num, take=take,
                source_dir=str(sd), source_mp4=str(mp4),
                used_as_primary=is_primary,
                status="duplicate" if not is_primary else "pending",
            )
            records.append((rec, out_dir, is_primary))

    # Print discovery info up front so the user can sanity-check the mapping.
    print(f"[ego→rpx] writing {sum(1 for _, _, ex in records if ex)} primary scenes to {args.output_root}")
    print()

    fmt, q, force = args.image_format, args.jpeg_quality, args.force
    n = args.n_samples
    out: List[SceneRecord] = []

    work = [(rec, out_dir) for rec, out_dir, ex in records if ex]
    if args.workers <= 1:
        for rec, out_dir in work:
            d = {**asdict(rec), "out_dir": str(out_dir)}
            r_dict = _process(d, n, fmt, q, force)
            elapsed = r_dict.pop("elapsed_s")
            r = SceneRecord(**r_dict)
            _print_line(r, elapsed)
            out.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for rec, out_dir in work:
                d = {**asdict(rec), "out_dir": str(out_dir)}
                futs[pool.submit(_process, d, n, fmt, q, force)] = rec.canonical
            for fut in as_completed(futs):
                r_dict = fut.result()
                elapsed = r_dict.pop("elapsed_s")
                r = SceneRecord(**r_dict)
                _print_line(r, elapsed)
                out.append(r)
        out.sort(key=lambda r: (r.scene_number, str(r.take or "")))

    # Add the extra-take records to the output (they don't get processed but
    # the manifest needs to reflect them).
    extras = [rec for rec, _, ex in records if not ex]
    full = out + extras
    full.sort(key=lambda r: (r.scene_number, 0 if r.used_as_primary else 1, str(r.take or "")))

    manifest = args.output_root / "scene_manifest.csv"
    write_manifest(full, manifest)
    print()
    print(f"[ego→rpx] manifest written: {manifest}")
    _print_summary(full, n, args.output_root)

    n_fail = sum(1 for r in full if r.used_as_primary and r.status not in ("ok", "short", "skip"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

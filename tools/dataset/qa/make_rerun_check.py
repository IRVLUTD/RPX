#!/usr/bin/env python3
"""Build a Rerun visual QA recording for RPX RGB/depth/mask samples.

The default run checks one representative middle frame for every MOS
scene-phase listed in preview/data_studio_preview.csv: 100 scenes x 3 phases.
It writes:

- out/rpx_visual_check.rrd: Rerun recording for interactive inspection.
- out/rpx_visual_contact_sheet.jpg: overview grid using the preview composites.
- out/visual_check_report.json: structural and per-sample QA summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


DEPTH_MM_TO_M = 0.001
DEPTH_SATURATED_MM = np.iinfo(np.uint16).max


@dataclass
class SampleResult:
    index: int
    scene_phase: str
    scene_id: str
    phase_index: int
    difficulty: str
    rgb_path: str
    depth_path: str
    mask_path: str
    rgb_frames: int
    depth_frames: int
    mask_frames: int
    frame_index: int
    rgb_size: tuple[int, int]
    depth_size: tuple[int, int]
    mask_size: tuple[int, int]
    depth_min: int
    depth_max: int
    depth_nonzero_fraction: float
    depth_zero_fraction: float
    depth_saturated_fraction: float
    depth_valid_fraction: float
    depth_valid_min_m: float | None
    depth_valid_max_m: float | None
    depth_valid_mean_m: float | None
    depth_valid_p50_m: float | None
    mask_instance_count: int
    mask_nonzero_fraction: float
    status: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="RPX dataset repository root or downloaded local_dir.",
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=Path("preview/data_studio_preview.csv"),
        help="CSV containing scene_phase and source shard paths.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("rerun_check/out"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--one-per-scene",
        action="store_true",
        help="Sample only the first listed phase for each scene instead of all 300 scene-phases.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to inspect. Default: middle frame of each RGB tar.",
    )
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--contact-thumb-width", type=int, default=220)
    parser.add_argument("--skip-rerun", action="store_true")
    return parser.parse_args()


def repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_rows(csv_path: Path, *, one_per_scene: bool, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_scenes: set[str] = set()
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if one_per_scene and row["scene_id"] in seen_scenes:
                continue
            seen_scenes.add(row["scene_id"])
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def tar_png_names(path: Path) -> list[str]:
    with tarfile.open(path) as tar:
        return sorted(
            member.name
            for member in tar.getmembers()
            if member.isfile() and member.name.lower().endswith(".png")
        )


def read_png_from_tar(path: Path, index: int) -> tuple[Image.Image, int, str]:
    with tarfile.open(path) as tar:
        names = sorted(
            member.name
            for member in tar.getmembers()
            if member.isfile() and member.name.lower().endswith(".png")
        )
        if not names:
            raise ValueError(f"no PNG frames in {path}")
        safe_index = min(max(index, 0), len(names) - 1)
        extracted = tar.extractfile(names[safe_index])
        if extracted is None:
            raise ValueError(f"could not read {names[safe_index]} from {path}")
        image = Image.open(extracted).copy()
    return image, len(names), names[safe_index]


def resize_width(image: Image.Image, width: int, *, nearest: bool = False) -> Image.Image:
    if image.width <= width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize((width, height), method)


def depth_valid_mask(arr: np.ndarray) -> np.ndarray:
    return (arr > 0) & (arr < DEPTH_SATURATED_MM)


def depth_to_uint8(depth: Image.Image) -> Image.Image:
    arr_mm = np.asarray(depth).astype(np.float32)
    valid = depth_valid_mask(arr_mm)
    if not valid.any():
        return Image.fromarray(np.zeros(arr_mm.shape, dtype=np.uint8), mode="L")
    arr_m = arr_mm * DEPTH_MM_TO_M
    lo, hi = np.percentile(arr_m[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((arr_m - lo) / (hi - lo), 0, 1)
    gray = np.zeros(arr_mm.shape, dtype=np.uint8)
    gray[valid] = (32 + norm[valid] * 223).astype(np.uint8)
    return Image.fromarray(gray, mode="L")


def depth_to_meters(depth: Image.Image) -> np.ndarray:
    arr_mm = np.asarray(depth).astype(np.float32)
    valid = depth_valid_mask(arr_mm)
    arr_m = arr_mm * DEPTH_MM_TO_M
    arr_m[~valid] = 0.0
    return arr_m


def valid_depth_metric_stats(depth_arr: np.ndarray) -> dict[str, float | None]:
    valid = depth_valid_mask(depth_arr)
    if not valid.any():
        return {
            "depth_valid_min_m": None,
            "depth_valid_max_m": None,
            "depth_valid_mean_m": None,
            "depth_valid_p50_m": None,
        }
    depth_m = depth_arr[valid].astype(np.float32) * DEPTH_MM_TO_M
    return {
        "depth_valid_min_m": float(np.min(depth_m)),
        "depth_valid_max_m": float(np.max(depth_m)),
        "depth_valid_mean_m": float(np.mean(depth_m)),
        "depth_valid_p50_m": float(np.percentile(depth_m, 50)),
    }


def colorize_mask(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    palette = np.asarray(
        [
            [244, 103, 83],
            [79, 210, 154],
            [255, 184, 77],
            [180, 101, 255],
            [64, 182, 255],
            [255, 99, 179],
            [141, 224, 83],
            [255, 226, 102],
            [102, 232, 232],
            [231, 91, 116],
            [161, 135, 255],
            [76, 217, 100],
        ],
        dtype=np.uint8,
    )
    for value in np.unique(arr):
        if value == 0:
            continue
        rgb[arr == value] = palette[(int(value) - 1) % len(palette)]
    return Image.fromarray(rgb, mode="RGB")


def sample_status(
    row: dict[str, str],
    rgb: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    rgb_frames: int,
    depth_frames: int,
    mask_frames: int,
    frame_index: int,
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
) -> SampleResult:
    notes: list[str] = []
    if not (rgb.size == depth.size == mask.size):
        notes.append(f"size mismatch: rgb={rgb.size} depth={depth.size} mask={mask.size}")
    if not (rgb_frames == depth_frames == mask_frames):
        notes.append(
            f"frame count mismatch: rgb={rgb_frames} depth={depth_frames} mask={mask_frames}"
        )

    depth_arr = np.asarray(depth)
    mask_arr = np.asarray(mask)
    depth_nonzero_fraction = float(np.count_nonzero(depth_arr) / depth_arr.size)
    depth_zero_fraction = float(np.count_nonzero(depth_arr == 0) / depth_arr.size)
    depth_saturated_fraction = float(np.count_nonzero(depth_arr == DEPTH_SATURATED_MM) / depth_arr.size)
    depth_valid_fraction = float(np.count_nonzero(depth_valid_mask(depth_arr)) / depth_arr.size)
    mask_nonzero_fraction = float(np.count_nonzero(mask_arr) / mask_arr.size)
    mask_values = np.unique(mask_arr)
    mask_instance_count = int(np.count_nonzero(mask_values))
    if depth_valid_fraction <= 0:
        notes.append("depth has no valid metric values")
    if mask_instance_count == 0:
        notes.append("mask has no foreground instances")
    depth_metric_stats = valid_depth_metric_stats(depth_arr)

    return SampleResult(
        index=-1,
        scene_phase=row["scene_phase"],
        scene_id=row["scene_id"],
        phase_index=int(row["phase_index"]),
        difficulty=row["difficulty"],
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        mask_path=str(mask_path),
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        mask_frames=mask_frames,
        frame_index=frame_index,
        rgb_size=rgb.size,
        depth_size=depth.size,
        mask_size=mask.size,
        depth_min=int(depth_arr.min()),
        depth_max=int(depth_arr.max()),
        depth_nonzero_fraction=depth_nonzero_fraction,
        depth_zero_fraction=depth_zero_fraction,
        depth_saturated_fraction=depth_saturated_fraction,
        depth_valid_fraction=depth_valid_fraction,
        **depth_metric_stats,
        mask_instance_count=mask_instance_count,
        mask_nonzero_fraction=mask_nonzero_fraction,
        status="ok" if not notes else "warning",
        notes=notes,
    )


def make_contact_sheet(
    repo_root: Path,
    rows: Iterable[dict[str, str]],
    out_path: Path,
    thumb_width: int,
) -> None:
    rows = list(rows)
    cols = 10
    label_h = 34
    thumbs: list[tuple[dict[str, str], Image.Image]] = []
    for row in rows:
        preview_name = Path(row["image_preview_url"]).name
        preview_path = repo_root / "preview" / "image_examples" / "preview" / "images" / preview_name
        image = Image.open(preview_path).convert("RGB")
        thumb_h = round(image.height * thumb_width / image.width)
        thumb = image.resize((thumb_width, thumb_h), Image.Resampling.LANCZOS)
        thumbs.append((row, thumb))

    if not thumbs:
        raise ValueError("no rows to render")
    thumb_h = thumbs[0][1].height
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for i, (row, thumb) in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        label = f"{i:03d} {row['scene_phase']} {row['difficulty']}"
        draw.rectangle((x, y + thumb_h, x + thumb_width, y + thumb_h + label_h), fill=(245, 245, 245))
        draw.text((x + 4, y + thumb_h + 4), label, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def open_rerun(rrd_path: Path):
    import rerun as rr

    rr.init("rpx_visual_check", spawn=False)
    rr.save(str(rrd_path))
    return rr


def log_to_rerun(rr, index: int, row: dict[str, str], rgb: Image.Image, depth: Image.Image, mask: Image.Image) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("sample", index)
    else:
        rr.set_time("sample", sequence=index)
    rgb_small = resize_width(rgb.convert("RGB"), 480)
    depth_m = depth_to_meters(resize_width(depth, 480, nearest=True))
    mask_arr = np.asarray(resize_width(mask, 480, nearest=True))
    rr.log("rpx/rgb", rr.Image(np.asarray(rgb_small)))
    try:
        rr.log("rpx/depth", rr.DepthImage(depth_m, meter=1.0))
    except TypeError:
        rr.log("rpx/depth", rr.DepthImage(depth_m))
    rr.log("rpx/masks", rr.SegmentationImage(mask_arr))
    markdown = (
        f"## {index:03d} {row['scene_phase']}\n\n"
        f"- scene: `{row['scene_id']}`\n"
        f"- phase: `{row['phase_index']}` / `{row['phase_name']}`\n"
        f"- difficulty: `{row['difficulty']}`\n"
        f"- rpx_ds: `{row['rpx_ds']}`\n"
    )
    if hasattr(rr, "TextDocument"):
        rr.log("rpx/metadata", rr.TextDocument(markdown, media_type="text/markdown"))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    rows_csv = repo_path(repo_root, args.rows_csv)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = out_dir / "rpx_visual_check.rrd"
    report_path = out_dir / "visual_check_report.json"
    contact_sheet_path = out_dir / "rpx_visual_contact_sheet.jpg"

    rows = load_rows(rows_csv, one_per_scene=args.one_per_scene, limit=args.limit)
    make_contact_sheet(repo_root, rows, contact_sheet_path, args.contact_thumb_width)

    rr = None if args.skip_rerun else open_rerun(rrd_path)
    results: list[SampleResult] = []
    for index, row in enumerate(tqdm(rows, desc="checking scene-phases")):
        rgb_path = repo_path(repo_root, row["source_rgb_shard"])
        depth_path = repo_path(repo_root, row["source_depth_shard"])
        mask_path = repo_path(repo_root, row["source_mask_shard"])
        rgb_names = tar_png_names(rgb_path)
        frame_index = args.frame_index if args.frame_index is not None else len(rgb_names) // 2
        rgb, rgb_frames, _ = read_png_from_tar(rgb_path, frame_index)
        depth, depth_frames, _ = read_png_from_tar(depth_path, frame_index)
        mask, mask_frames, _ = read_png_from_tar(mask_path, frame_index)
        result = sample_status(
            row,
            rgb,
            depth,
            mask,
            rgb_frames,
            depth_frames,
            mask_frames,
            frame_index,
            rgb_path,
            depth_path,
            mask_path,
        )
        result.index = index
        results.append(result)
        if rr is not None:
            log_to_rerun(rr, index, row, rgb, depth, mask)

    warnings = [result for result in results if result.status != "ok"]
    scene_ids = sorted({result.scene_id for result in results})
    summary = {
        "repo_root": str(repo_root),
        "rows_csv": str(rows_csv),
        "samples_checked": len(results),
        "unique_scenes": len(scene_ids),
        "scene_phase_mode": "one_per_scene" if args.one_per_scene else "all_scene_phases",
        "rrd_path": str(rrd_path) if rr is not None else None,
        "contact_sheet_path": str(contact_sheet_path),
        "warnings": len(warnings),
        "warning_scene_phases": [result.scene_phase for result in warnings],
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {contact_sheet_path}")
    if rr is not None:
        print(f"wrote {rrd_path}")
    if warnings:
        print(f"warnings: {len(warnings)}")
    else:
        print("warnings: 0")


if __name__ == "__main__":
    main()

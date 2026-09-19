#!/usr/bin/env python3
"""Full RPX release QA across modalities, labels, manifests, and previews.

This is the broader companion to make_rerun_check.py. It validates all MOS
scene-phases and all selected SOS objects, not only RGB-D segmentation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


MOS_EXPECTED_FRAMES = 250
SOS_EXPECTED_FRAMES = 500
RGBD_SIZE = (640, 480)
FISHEYE_SIZE = (848, 800)
LABEL_VERSION = "v1"
DEPTH_MM_TO_M = 0.001
DEPTH_SATURATED_MM = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class Unit:
    kind: str
    unit_id: str
    root: Path
    expected_frames: int
    view: str | None = None
    scene_id: str | None = None
    phase_index: int | None = None
    phase_name: str | None = None
    difficulty: str | None = None
    object_id: str | None = None
    global_object_id: int | None = None
    object_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="RPX dataset repository root or downloaded local_dir.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("rerun_check/out_all"))
    parser.add_argument(
        "--sample-policy",
        choices=["middle", "endpoints", "all"],
        default="endpoints",
        help=(
            "Frames to decode inside each stream. 'endpoints' means first/middle/last. "
            "'all' is exhaustive but can take a long time."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--kinds",
        default="mos,sos",
        help="Comma-separated unit kinds to check: mos, sos, or both.",
    )
    parser.add_argument("--image-width", type=int, default=360)
    parser.add_argument("--contact-thumb-width", type=int, default=140)
    parser.add_argument("--contact-cols", type=int, default=3)
    parser.add_argument("--skip-contact-sheet", action="store_true")
    parser.add_argument("--skip-rerun", action="store_true")
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser.parse_args()


def repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def tar_file_names(path: Path, errors: list[str]) -> list[str]:
    if not path.exists():
        errors.append(f"missing shard: {path}")
        return []
    try:
        with tarfile.open(path) as tar:
            return sorted(member.name for member in tar.getmembers() if member.isfile())
    except (tarfile.TarError, OSError) as exc:
        errors.append(f"unreadable tar {path}: {exc}")
        return []


def names_with_prefix(names: Iterable[str], prefix: str, suffix: str) -> list[str]:
    needle = prefix.rstrip("/") + "/"
    suffix_lower = suffix.lower()
    return sorted(
        name
        for name in names
        if name.startswith(needle) and name.lower().endswith(suffix_lower)
    )


def frame_indices_for_count(count: int, policy: str) -> list[int]:
    if count <= 0:
        return []
    if policy == "all":
        return list(range(count))
    if policy == "middle":
        return [count // 2]
    return sorted({0, count // 2, count - 1})


def pick_names(names: list[str], policy: str) -> list[str]:
    return [names[index] for index in frame_indices_for_count(len(names), policy)]


def count_tar_pngs(path: Path, prefix: str) -> int:
    if not path.exists():
        return 0
    try:
        with tarfile.open(path) as tar:
            return len(
                [
                    member
                    for member in tar.getmembers()
                    if member.isfile()
                    and member.name.startswith(prefix.rstrip("/") + "/")
                    and member.name.lower().endswith(".png")
                ]
            )
    except (tarfile.TarError, OSError):
        return 0


def numbered_sequence_summary(
    names: list[str],
    prefix: str,
    suffix: str,
    expected_count: int,
    errors: list[str],
    label: str,
) -> dict[str, Any]:
    indices: list[int] = []
    malformed: list[str] = []
    prefix = prefix.rstrip("/") + "/"
    for name in names:
        rel = name[len(prefix) :] if name.startswith(prefix) else name
        if "/" in rel:
            malformed.append(name)
            continue
        if not rel.lower().endswith(suffix.lower()):
            malformed.append(name)
            continue
        stem = rel[: -len(suffix)]
        if not stem.isdigit():
            malformed.append(name)
            continue
        indices.append(int(stem))

    count = len(indices)
    if count != expected_count:
        errors.append(f"{label} count {count}, expected {expected_count}")
    expected = set(range(expected_count))
    actual = set(indices)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicate_count = count - len(actual)
    if malformed:
        errors.append(f"{label} has malformed names, first examples: {malformed[:5]}")
    if missing:
        errors.append(f"{label} missing frame ids, first examples: {missing[:10]}")
    if extra:
        errors.append(f"{label} has unexpected frame ids, first examples: {extra[:10]}")
    if duplicate_count:
        errors.append(f"{label} has {duplicate_count} duplicate frame ids")

    return {
        "count": count,
        "first": min(indices) if indices else None,
        "last": max(indices) if indices else None,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "duplicate_count": duplicate_count,
        "malformed_count": len(malformed),
    }


def image_to_stats(image: Image.Image) -> dict[str, Any]:
    arr = np.asarray(image)
    stats: dict[str, Any] = {
        "mode": image.mode,
        "size": list(image.size),
        "dtype": str(arr.dtype),
    }
    if arr.size:
        stats.update(
            {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
            }
        )
    return stats


def depth_valid_mask(arr: np.ndarray) -> np.ndarray:
    return (arr > 0) & (arr < DEPTH_SATURATED_MM)


def depth_to_stats(image: Image.Image) -> dict[str, Any]:
    stats = image_to_stats(image)
    arr = np.asarray(image)
    valid = depth_valid_mask(arr)
    invalid = ~valid
    stats.update(
        {
            "unit": "millimeter",
            "meter_scale": DEPTH_MM_TO_M,
            "zero_fraction": float(np.count_nonzero(arr == 0) / arr.size) if arr.size else 0.0,
            "saturated_fraction": float(np.count_nonzero(arr == DEPTH_SATURATED_MM) / arr.size)
            if arr.size
            else 0.0,
            "invalid_fraction": float(np.count_nonzero(invalid) / arr.size) if arr.size else 0.0,
            "valid_fraction": float(np.count_nonzero(valid) / arr.size) if arr.size else 0.0,
        }
    )
    if valid.any():
        valid_mm = arr[valid].astype(np.float32)
        valid_m = valid_mm * DEPTH_MM_TO_M
        stats.update(
            {
                "valid_min_mm": float(np.min(valid_mm)),
                "valid_max_mm": float(np.max(valid_mm)),
                "valid_mean_mm": float(np.mean(valid_mm)),
                "valid_std_mm": float(np.std(valid_mm)),
                "valid_min_m": float(np.min(valid_m)),
                "valid_max_m": float(np.max(valid_m)),
                "valid_mean_m": float(np.mean(valid_m)),
                "valid_std_m": float(np.std(valid_m)),
                "valid_p01_m": float(np.percentile(valid_m, 1)),
                "valid_p50_m": float(np.percentile(valid_m, 50)),
                "valid_p99_m": float(np.percentile(valid_m, 99)),
            }
        )
    return stats


def decode_images(
    path: Path,
    sample_names: list[str],
    errors: list[str],
    warnings: list[str],
    label: str,
    expected_size: tuple[int, int] | None = None,
    require_nonzero: bool = False,
    stats_fn: Callable[[Image.Image], dict[str, Any]] = image_to_stats,
) -> tuple[list[dict[str, Any]], dict[str, Image.Image]]:
    stats: list[dict[str, Any]] = []
    images: dict[str, Image.Image] = {}
    if not sample_names:
        return stats, images
    try:
        with tarfile.open(path) as tar:
            for name in sample_names:
                try:
                    extracted = tar.extractfile(name)
                    if extracted is None:
                        errors.append(f"{label}: could not extract {name}")
                        continue
                    image = Image.open(extracted).copy()
                except Exception as exc:  # PIL raises several exception types.
                    errors.append(f"{label}: could not decode {name}: {exc}")
                    continue

                item = {"name": name, **stats_fn(image)}
                if expected_size is not None and image.size != expected_size:
                    errors.append(
                        f"{label}: {name} size {image.size}, expected {expected_size}"
                    )
                valid_fraction = item.get("valid_fraction", item.get("nonzero_fraction", 0.0))
                if require_nonzero and valid_fraction <= 0:
                    warnings.append(f"{label}: {name} has no valid pixels")
                if item.get("std", 0.0) <= 0:
                    warnings.append(f"{label}: {name} has zero image variance")
                stats.append(item)
                images[name] = image
    except (tarfile.TarError, OSError) as exc:
        errors.append(f"{label}: could not reopen {path}: {exc}")
    return stats, images


def decode_npz_samples(
    path: Path,
    sample_names: list[str],
    errors: list[str],
    warnings: list[str],
    label: str,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    if not sample_names:
        return stats
    try:
        with tarfile.open(path) as tar:
            for name in sample_names:
                try:
                    extracted = tar.extractfile(name)
                    if extracted is None:
                        errors.append(f"{label}: could not extract {name}")
                        continue
                    arrays = np.load(io.BytesIO(extracted.read()))
                except Exception as exc:
                    errors.append(f"{label}: could not load {name}: {exc}")
                    continue

                item: dict[str, Any] = {"name": name, "fields": sorted(arrays.files)}
                if sorted(arrays.files) != ["orientation", "position"]:
                    errors.append(f"{label}: {name} fields {arrays.files}")
                for field, shape in [("position", (3,)), ("orientation", (4,))]:
                    if field not in arrays:
                        continue
                    arr = np.asarray(arrays[field])
                    item[field] = {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "finite": bool(np.isfinite(arr).all()),
                        "norm": float(np.linalg.norm(arr)),
                    }
                    if arr.shape != shape:
                        errors.append(f"{label}: {name} {field} shape {arr.shape}")
                    if not np.isfinite(arr).all():
                        errors.append(f"{label}: {name} {field} contains non-finite values")
                if "orientation" in item:
                    norm = item["orientation"]["norm"]
                    if not math.isclose(norm, 1.0, rel_tol=0.02, abs_tol=0.02):
                        warnings.append(f"{label}: {name} quaternion norm is {norm:.4f}")
                stats.append(item)
    except (tarfile.TarError, OSError) as exc:
        errors.append(f"{label}: could not reopen {path}: {exc}")
    return stats


def read_tar_text(tar: tarfile.TarFile, name: str) -> str:
    extracted = tar.extractfile(name)
    if extracted is None:
        raise ValueError(f"could not extract {name}")
    return extracted.read().decode("utf-8", errors="replace")


def read_tar_json(tar: tarfile.TarFile, name: str) -> Any:
    return json.loads(read_tar_text(tar, name))


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


def colorize_mask(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        return mask.convert("RGB")
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


def fit_image(image: Image.Image, box: tuple[int, int], *, nearest: bool = False) -> Image.Image:
    image = image.convert("RGB")
    max_w, max_h = box
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    resized = image.resize(size, method)
    canvas = Image.new("RGB", box, (16, 18, 22))
    canvas.paste(resized, ((max_w - size[0]) // 2, (max_h - size[1]) // 2))
    return canvas


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Times New Roman.ttf", "Times_New_Roman.ttf", "LiberationSerif-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_unit_tile(
    unit: Unit,
    result: dict[str, Any],
    viz: dict[str, Image.Image],
    thumb_width: int,
) -> Image.Image:
    panel_h = max(90, round(thumb_width * 0.72))
    label_h = 38
    panel_names = ["rgb", "depth", "mask", "fisheye_left", "fisheye_right"]
    tile = Image.new("RGB", (thumb_width * len(panel_names), panel_h + label_h), (13, 17, 23))
    draw = ImageDraw.Draw(tile)
    font = load_font(13)
    small_font = load_font(11)

    for i, name in enumerate(panel_names):
        image = viz.get(name)
        x = i * thumb_width
        if image is None:
            panel = Image.new("RGB", (thumb_width, panel_h), (38, 42, 49))
            ImageDraw.Draw(panel).text((8, 8), "missing", fill=(230, 230, 230), font=small_font)
        else:
            nearest = name in {"depth", "mask"}
            panel = fit_image(image, (thumb_width, panel_h), nearest=nearest)
        tile.paste(panel, (x, 0))
        draw.text((x + 5, panel_h - 16), name.replace("_", " "), fill=(235, 235, 235), font=small_font)

    status = result["status"].upper()
    label = unit.unit_id
    if unit.kind == "mos":
        label += f" {unit.difficulty or ''}"
    elif unit.kind == "ego":
        label += " ego"
    else:
        label += f" gid={unit.global_object_id}"
    status_color = (83, 210, 154) if result["status"] == "ok" else (255, 184, 77)
    if result["status"] == "error":
        status_color = (244, 103, 83)
    draw.rectangle((0, panel_h, tile.width, tile.height), fill=(245, 245, 242))
    draw.text((7, panel_h + 5), label[:62], fill=(20, 24, 30), font=font)
    draw.text((7, panel_h + 21), status, fill=status_color, font=small_font)
    if result["errors"] or result["warnings"]:
        msg = (result["errors"] or result["warnings"])[0]
        draw.text((70, panel_h + 21), msg[:82], fill=(60, 64, 72), font=small_font)
    return tile


def make_contact_sheet(tiles: list[Image.Image], out_path: Path, cols: int) -> None:
    if not tiles:
        return
    cols = max(1, cols)
    tile_w, tile_h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (245, 245, 242))
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        sheet.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)


def open_rerun(rrd_path: Path):
    import rerun as rr

    rr.init("rpx_all_modalities_visual_check", spawn=False)
    rr.save(str(rrd_path))
    return rr


def set_rerun_time(rr, index: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("unit", index)
    else:
        rr.set_time("unit", sequence=index)


def resize_width(image: Image.Image, width: int, *, nearest: bool = False) -> Image.Image:
    if image.width <= width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize((width, height), method)


def log_to_rerun(
    rr,
    index: int,
    unit: Unit,
    result: dict[str, Any],
    viz: dict[str, Image.Image],
    image_width: int,
) -> None:
    set_rerun_time(rr, index)
    if "rgb" in viz:
        rr.log("rpx/rgb", rr.Image(np.asarray(resize_width(viz["rgb"].convert("RGB"), image_width))))
    if "depth" in viz:
        depth = resize_width(viz["depth"], image_width, nearest=True)
        depth_m = depth_to_meters(depth)
        try:
            rr.log("rpx/depth", rr.DepthImage(depth_m, meter=1.0))
        except TypeError:
            rr.log("rpx/depth", rr.DepthImage(depth_m))
    if "mask" in viz:
        mask = resize_width(viz["mask"], image_width, nearest=True)
        rr.log("rpx/masks", rr.SegmentationImage(np.asarray(mask)))
    if "fisheye_left" in viz:
        left = resize_width(viz["fisheye_left"].convert("RGB"), image_width)
        rr.log("rpx/fisheye/left", rr.Image(np.asarray(left)))
    if "fisheye_right" in viz:
        right = resize_width(viz["fisheye_right"].convert("RGB"), image_width)
        rr.log("rpx/fisheye/right", rr.Image(np.asarray(right)))

    markdown = [
        f"## {index:03d} {unit.unit_id}",
        "",
        f"- kind: `{unit.kind}`",
        f"- view: `{unit.view}`",
        f"- status: `{result['status']}`",
        f"- expected frames: `{unit.expected_frames}`",
    ]
    if unit.kind == "mos":
        markdown.extend(
            [
                f"- scene: `{unit.scene_id}`",
                f"- phase: `{unit.phase_index}` / `{unit.phase_name}`",
                f"- difficulty: `{unit.difficulty}`",
            ]
        )
    elif unit.kind == "ego":
        markdown.extend(
            [
                f"- scene: `{unit.scene_id}`",
                "- ESD: `not applicable`",
            ]
        )
    else:
        markdown.extend(
            [
                f"- object: `{unit.object_id}`",
                f"- global object id: `{unit.global_object_id}`",
                f"- object name: `{unit.object_name}`",
            ]
        )
    for message in result["errors"][:5]:
        markdown.append(f"- error: {message}")
    for message in result["warnings"][:5]:
        markdown.append(f"- warning: {message}")
    if hasattr(rr, "TextDocument"):
        rr.log("rpx/metadata", rr.TextDocument("\n".join(markdown), media_type="text/markdown"))


def build_units(repo_root: Path, kinds: set[str]) -> tuple[list[Unit], dict[str, Any]]:
    context: dict[str, Any] = {}
    units: list[Unit] = []

    split_rows: list[dict[str, str]] = []
    for split in ("easy", "medium", "hard"):
        csv_path = repo_root / "splits" / f"{split}.csv"
        rows = read_csv_rows(csv_path)
        for row in rows:
            row = dict(row)
            row["split"] = split
            split_rows.append(row)
    context["split_rows"] = split_rows

    if "mos" in kinds:
        for row in sorted(split_rows, key=lambda r: (r["scene_id"], int(r["phase_index"]))):
            phase_index = int(row["phase_index"])
            units.append(
                Unit(
                    kind="mos",
                    unit_id=f"{row['scene_id']}.phase{phase_index}",
                    root=repo_root / "scenes" / row["scene_id"] / str(phase_index),
                    expected_frames=MOS_EXPECTED_FRAMES,
                    view="phase",
                    scene_id=row["scene_id"],
                    phase_index=phase_index,
                    phase_name=row["phase_name"],
                    difficulty=row["difficulty"],
                )
            )

    selected_rows = read_csv_rows(repo_root / "manifest" / "selected_sos_objects_v1.csv")
    context["selected_sos_rows"] = selected_rows
    if "sos" in kinds:
        for row in sorted(selected_rows, key=lambda r: int(r["global_object_id"])):
            units.append(
                Unit(
                    kind="sos",
                    unit_id=f"object:{row['object_id']}",
                    root=repo_root / "objects" / row["object_id"] / "0",
                    expected_frames=SOS_EXPECTED_FRAMES,
                    view="object",
                    object_id=row["object_id"],
                    global_object_id=int(row["global_object_id"]),
                    object_name=row["object_name"],
                )
            )

    ego_rows: list[dict[str, str]] = []
    ego_frames_csv = repo_root / "manifest" / "ego_frames_v1.csv"
    if ego_frames_csv.exists():
        ego_rows = read_csv_rows(ego_frames_csv)
    context["ego_frame_rows"] = ego_rows
    if "ego" in kinds:
        if ego_rows:
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in ego_rows:
                grouped[row["scene_id"]].append(row)
            for scene_id, rows in sorted(grouped.items()):
                units.append(
                    Unit(
                        kind="ego",
                        unit_id=f"{scene_id}.ego",
                        root=repo_root / "scenes" / scene_id / "ego",
                        expected_frames=len(rows),
                        view="ego",
                        scene_id=scene_id,
                        phase_name="ego",
                    )
                )
        else:
            for root in sorted((repo_root / "scenes").glob("scene*/ego")):
                frame_count = count_tar_pngs(root / "rgb.tar", "rgb")
                if frame_count <= 0:
                    continue
                units.append(
                    Unit(
                        kind="ego",
                        unit_id=f"{root.parent.name}.ego",
                        root=root,
                        expected_frames=frame_count,
                        view="ego",
                        scene_id=root.parent.name,
                        phase_name="ego",
                    )
                )

    mask_map_rows = read_csv_rows(repo_root / "manifest" / "mos_mask_object_map_v1.csv")
    mask_map_by_phase: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in mask_map_rows:
        mask_map_by_phase[(row["scene_id"], row["phase"])].append(row)
    context["mask_map_rows"] = mask_map_rows
    context["mask_map_by_phase"] = mask_map_by_phase
    ego_mask_map_path = repo_root / "manifest" / "mos_ego_mask_object_map_v1.csv"
    ego_mask_map_rows = read_csv_rows(ego_mask_map_path) if ego_mask_map_path.exists() else []
    ego_mask_map_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ego_mask_map_rows:
        ego_mask_map_by_scene[row["scene_id"]].append(row)
    context["ego_mask_map_rows"] = ego_mask_map_rows
    context["ego_mask_map_by_scene"] = ego_mask_map_by_scene
    return units, context


def check_global_manifests(repo_root: Path, units: list[Unit], context: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {"errors": errors, "warnings": warnings}

    current_path = repo_root / "manifest" / "current.json"
    try:
        current = load_json(current_path)
        summary["current_json"] = {
            "schema_version": current.get("schema_version"),
            "label_versions": current.get("label_versions"),
            "mos": current.get("mos"),
            "sos": current.get("sos"),
        }
        expected_versions = {
            "masks": LABEL_VERSION,
            "masks_aux": LABEL_VERSION,
            "sam2_meta": LABEL_VERSION,
            "cam_pose": LABEL_VERSION,
        }
        if current.get("label_versions") != expected_versions:
            errors.append(f"manifest/current.json label_versions != {expected_versions}")
    except Exception as exc:
        errors.append(f"could not read manifest/current.json: {exc}")

    split_rows = context["split_rows"]
    phase_ids = [row["scene_phase"] for row in split_rows]
    difficulty_counts = Counter(row["difficulty"] for row in split_rows)
    split_counts = Counter(row["split"] for row in split_rows)
    summary["phase_splits"] = {
        "rows": len(split_rows),
        "unique_phase_ids": len(set(phase_ids)),
        "split_counts": dict(split_counts),
        "difficulty_counts": dict(difficulty_counts),
    }
    if len(split_rows) != 300 or len(set(phase_ids)) != 300:
        errors.append("phase split CSVs should contain 300 unique scene phases")
    for split in ("easy", "medium", "hard"):
        if split_counts[split] != 100:
            errors.append(f"splits/{split}.csv has {split_counts[split]} rows, expected 100")
        mismatch = [row["scene_phase"] for row in split_rows if row["split"] == split and row["difficulty"] != split]
        if mismatch:
            errors.append(f"splits/{split}.csv difficulty mismatch examples: {mismatch[:5]}")

    scene_splits_path = repo_root / "splits" / "scene_splits.json"
    try:
        scene_splits = load_json(scene_splits_path)
        weights = scene_splits.get("provenance", {}).get("weights", {})
        feature_names = scene_splits.get("provenance", {}).get("feature_names", [])
        tiers = scene_splits.get("splits", {})
        tier_ids = [scene_id for ids in tiers.values() for scene_id in ids]
        summary["scene_splits_json"] = {
            "n_scenes": scene_splits.get("n_scenes"),
            "primary_method": scene_splits.get("primary_method"),
            "scoring_version": scene_splits.get("scoring_version"),
            "feature_count": len(feature_names),
            "weight_count": len(weights),
            "weight_sum": round(float(sum(weights.values())), 8) if weights else None,
            "tier_counts": {key: len(value) for key, value in tiers.items()},
        }
        if len(feature_names) != 27 or len(weights) != 27:
            errors.append("scene_splits.json should expose 27 ESD features and 27 weights")
        if set(feature_names) != set(weights):
            errors.append("scene_splits.json feature_names and weight keys differ")
        if weights and not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            errors.append(f"scene_splits.json weights sum to {sum(weights.values())}, expected 1.0")
        if len(tier_ids) != 100 or len(set(tier_ids)) != 100:
            errors.append("scene_splits.json should contain 100 unique scene-tier IDs")
    except Exception as exc:
        errors.append(f"could not read splits/scene_splits.json: {exc}")

    preview_csv = repo_root / "preview" / "mos_phase_preview.csv"
    if not preview_csv.exists():
        preview_csv = repo_root / "preview" / "data_studio_preview.csv"
    try:
        preview_rows = read_csv_rows(preview_csv)
        preview_errors = 0
        preview_sizes = Counter()
        for row in preview_rows:
            image_name = Path(row["image_preview_url"]).name
            image_path = repo_root / "preview" / "image_examples" / "preview" / "images" / image_name
            if not image_path.exists():
                preview_errors += 1
                continue
            try:
                with Image.open(image_path) as image:
                    preview_sizes[str(image.size)] += 1
            except Exception:
                preview_errors += 1
        summary["data_studio_preview"] = {
            "rows": len(preview_rows),
            "unique_scene_phases": len({row["scene_phase"] for row in preview_rows}),
            "image_decode_errors": preview_errors,
            "image_sizes": dict(preview_sizes),
        }
        if len(preview_rows) != 300:
            errors.append(f"preview/data_studio_preview.csv has {len(preview_rows)} rows, expected 300")
        if preview_errors:
            errors.append(f"Data Studio preview image decode/missing errors: {preview_errors}")
    except Exception as exc:
        errors.append(f"could not read preview/data_studio_preview.csv: {exc}")

    selected_rows = context["selected_sos_rows"]
    global_ids = sorted(int(row["global_object_id"]) for row in selected_rows)
    summary["selected_sos_objects"] = {
        "rows": len(selected_rows),
        "unique_object_ids": len({row["object_id"] for row in selected_rows}),
        "global_id_min": global_ids[0] if global_ids else None,
        "global_id_max": global_ids[-1] if global_ids else None,
    }
    if len(selected_rows) != 70 or global_ids != list(range(1, 71)):
        errors.append("selected_sos_objects_v1.csv should contain global IDs 1..70")

    try:
        object_catalog = load_json(repo_root / "manifest" / "object_catalog_v1.json")
        catalog_objects = object_catalog.get("objects", [])
        summary["object_catalog"] = {"objects": len(catalog_objects)}
        if len(catalog_objects) != 70:
            errors.append("object_catalog_v1.json should contain 70 selected objects")
    except Exception as exc:
        errors.append(f"could not read object_catalog_v1.json: {exc}")

    try:
        meta_index = load_json(repo_root / "objects_meta" / "_index.json")
        summary["objects_meta_index"] = {
            "selected_object_count": meta_index.get("selected_object_count"),
            "object_ids": len(meta_index.get("object_ids", [])),
        }
        if meta_index.get("selected_object_count") != 70:
            errors.append("objects_meta/_index.json selected_object_count should be 70")
    except Exception as exc:
        errors.append(f"could not read objects_meta/_index.json: {exc}")

    mask_rows = context["mask_map_rows"]
    summary["mos_mask_object_map"] = {
        "rows": len(mask_rows),
        "unique_scene_phases": len({(row["scene_id"], row["phase"]) for row in mask_rows}),
    }
    if len(mask_rows) != 2100:
        errors.append(f"mos_mask_object_map_v1.csv has {len(mask_rows)} rows, expected 2100")

    ego_frame_rows = context.get("ego_frame_rows", [])
    ego_mask_rows = context.get("ego_mask_map_rows", [])
    if ego_frame_rows:
        summary["ego_frames"] = {
            "rows": len(ego_frame_rows),
            "unique_scenes": len({row["scene_id"] for row in ego_frame_rows}),
        }
    if ego_mask_rows:
        summary["mos_ego_mask_object_map"] = {
            "rows": len(ego_mask_rows),
            "unique_scenes": len({row["scene_id"] for row in ego_mask_rows}),
        }

    ego_preview_csv = repo_root / "preview" / "ego_preview.csv"
    if ego_preview_csv.exists():
        try:
            ego_preview_rows = read_csv_rows(ego_preview_csv)
            summary["ego_preview"] = {
                "rows": len(ego_preview_rows),
                "unique_scenes": len({row["scene_id"] for row in ego_preview_rows}),
            }
        except Exception as exc:
            errors.append(f"could not read preview/ego_preview.csv: {exc}")

    for path in [repo_root / "assets" / "rpx_teaser.png", repo_root / "assets" / "rpx-jumbotron.webm"]:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty asset: {path}")
    try:
        with Image.open(repo_root / "assets" / "rpx_teaser.png") as image:
            summary["teaser_image"] = {"size": list(image.size), "mode": image.mode}
    except Exception as exc:
        errors.append(f"could not decode assets/rpx_teaser.png: {exc}")

    parquet_summary = check_parquet_manifests(repo_root, units, warnings, errors)
    summary["parquet"] = parquet_summary
    summary["status"] = "error" if errors else ("warning" if warnings else "ok")
    return summary


def check_parquet_manifests(
    repo_root: Path,
    units: list[Unit],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        warnings.append(f"pyarrow unavailable; parquet row-count checks skipped: {exc}")
        return {"skipped": True, "reason": str(exc)}

    expected_rows = {
        "manifest/frames_v1.parquet": 110000,
        "preview/mos_phase_preview.parquet": 300,
        "manifest/selected_sos_objects_v1.parquet": 70,
        "splits/easy.parquet": 100,
        "splits/medium.parquet": 100,
        "splits/hard.parquet": 100,
    }
    if not (repo_root / "preview" / "mos_phase_preview.parquet").exists():
        expected_rows["preview/media_preview.parquet"] = 300
        expected_rows.pop("preview/mos_phase_preview.parquet", None)
    ego_frames_csv = repo_root / "manifest" / "ego_frames_v1.csv"
    ego_frame_rows = read_csv_rows(ego_frames_csv) if ego_frames_csv.exists() else []
    if ego_frame_rows:
        expected_rows["manifest/ego_frames_v1.parquet"] = len(ego_frame_rows)
        expected_rows["manifest/frames_v2.parquet"] = 110000 + len(ego_frame_rows)
    ego_mask_csv = repo_root / "manifest" / "mos_ego_mask_object_map_v1.csv"
    if ego_mask_csv.exists():
        expected_rows["manifest/mos_ego_mask_object_map_v1.parquet"] = len(read_csv_rows(ego_mask_csv))
    ego_preview_csv = repo_root / "preview" / "ego_preview.csv"
    if ego_preview_csv.exists():
        expected_rows["preview/ego_preview.parquet"] = len(read_csv_rows(ego_preview_csv))
    for rel, expected in expected_rows.items():
        path = repo_root / rel
        if not path.exists():
            errors.append(f"missing parquet: {rel}")
            continue
        try:
            pf = pq.ParquetFile(path)
            row_count = pf.metadata.num_rows
            summary[rel] = {"rows": row_count, "columns": pf.schema_arrow.names}
            if row_count != expected:
                errors.append(f"{rel} row count {row_count}, expected {expected}")
        except Exception as exc:
            errors.append(f"could not read {rel}: {exc}")

    frames_path = repo_root / "manifest" / "frames_v1.parquet"
    if frames_path.exists():
        try:
            table = pq.read_table(
                frames_path,
                columns=[
                    "scene_id",
                    "scene_type",
                    "phase",
                    "has_rgb",
                    "has_depth",
                    "has_fisheye",
                    "has_cam_pose",
                    "has_masks",
                    "has_masks_aux",
                    "has_sam2_meta",
                ],
            )
            data = {name: table[name].to_pylist() for name in table.column_names}
            unit_counts: Counter[tuple[str, str, int]] = Counter()
            false_counts: Counter[str] = Counter()
            for index in range(table.num_rows):
                unit_counts[(data["scene_type"][index], data["scene_id"][index], int(data["phase"][index]))] += 1
                for column in (
                    "has_rgb",
                    "has_depth",
                    "has_fisheye",
                    "has_cam_pose",
                    "has_masks",
                    "has_masks_aux",
                    "has_sam2_meta",
                ):
                    if not data[column][index]:
                        false_counts[column] += 1

            frame_count_mismatches = []
            for unit in units:
                if unit.kind == "mos":
                    key = ("multi_object", unit.scene_id or "", unit.phase_index)
                elif unit.kind == "sos":
                    key = ("single_object", unit.object_id or "", 0)
                else:
                    continue
                count = unit_counts.get(key, 0)
                if count != unit.expected_frames:
                    frame_count_mismatches.append(
                        {"unit_id": unit.unit_id, "manifest_frames": count, "expected": unit.expected_frames}
                    )
            summary["manifest/frames_v1.parquet"]["unit_frame_count_mismatches"] = frame_count_mismatches[:20]
            summary["manifest/frames_v1.parquet"]["false_modality_flags"] = dict(false_counts)
            if frame_count_mismatches:
                errors.append(f"frames_v1.parquet unit frame mismatches: {len(frame_count_mismatches)}")
            if false_counts:
                errors.append(f"frames_v1.parquet has false modality flags: {dict(false_counts)}")
        except Exception as exc:
            errors.append(f"could not validate frames_v1.parquet unit rows: {exc}")
    return summary


def check_image_stream(
    shard_path: Path,
    names: list[str],
    prefix: str,
    expected_count: int,
    expected_size: tuple[int, int],
    policy: str,
    errors: list[str],
    warnings: list[str],
    label: str,
    require_nonzero: bool = False,
    stats_fn: Callable[[Image.Image], dict[str, Any]] = image_to_stats,
) -> tuple[dict[str, Any], dict[str, Image.Image]]:
    stream_names = names_with_prefix(names, prefix, ".png")
    sequence = numbered_sequence_summary(stream_names, prefix, ".png", expected_count, errors, label)
    sample_names = pick_names(stream_names, policy)
    sample_stats, sample_images = decode_images(
        shard_path,
        sample_names,
        errors,
        warnings,
        label,
        expected_size=expected_size,
        require_nonzero=require_nonzero,
        stats_fn=stats_fn,
    )
    return {
        "prefix": prefix,
        **sequence,
        "sampled": sample_stats,
    }, sample_images


def check_masks_aux(
    shard_path: Path,
    names: list[str],
    unit: Unit,
    policy: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    group_counts: Counter[str] = Counter()
    for name in names:
        parts = name.split("/")
        key = "/".join(parts[:2]) if len(parts) >= 2 else name
        group_counts[key] += 1

    summary: dict[str, Any] = {"file_count": len(names), "group_counts": dict(sorted(group_counts.items()))}
    expected = unit.expected_frames
    frame_aligned_groups = ["sam2/contour_gt_masks", "sam2/palette", "sam2/rgb_and_mask"]
    if unit.kind in {"mos", "ego"}:
        frame_aligned_groups.append("sam2/masks_contour_with_hidden")

    decoded_groups: dict[str, Any] = {}
    for group in frame_aligned_groups:
        group_names = names_with_prefix(names, group, ".png")
        sequence = numbered_sequence_summary(
            group_names,
            group,
            ".png",
            expected,
            errors,
            f"{unit.unit_id} masks_aux:{group}",
        )
        sample_stats, _ = decode_images(
            shard_path,
            pick_names(group_names, policy),
            errors,
            warnings,
            f"{unit.unit_id} masks_aux:{group}",
            expected_size=RGBD_SIZE,
        )
        decoded_groups[group] = {**sequence, "sampled": sample_stats}

    bbox_names = names_with_prefix(names, "sam2/bbox_overlay", ".png")
    decoded_bbox, _ = decode_images(
        shard_path,
        bbox_names[:3],
        errors,
        warnings,
        f"{unit.unit_id} masks_aux:sam2/bbox_overlay",
        expected_size=RGBD_SIZE,
    )
    decoded_groups["sam2/bbox_overlay"] = {"count": len(bbox_names), "sampled": decoded_bbox}

    dino_names = [name for name in names if name.startswith("sam2/dino_output/")]
    dino_png_names = [name for name in dino_names if name.lower().endswith(".png")]
    dino_json_names = [name for name in dino_names if name.lower().endswith(".json")]
    if unit.kind in {"mos", "ego"}:
        if not dino_names:
            warnings.append(f"{unit.unit_id} masks_aux:sam2/dino_output is missing")
        decoded_dino, _ = decode_images(
            shard_path,
            dino_png_names[:3],
            errors,
            warnings,
            f"{unit.unit_id} masks_aux:sam2/dino_output",
        )
        decoded_groups["sam2/dino_output"] = {
            "count": len(dino_names),
            "png_count": len(dino_png_names),
            "json_count": len(dino_json_names),
            "sampled": decoded_dino,
        }
    elif dino_names:
        decoded_dino, _ = decode_images(
            shard_path,
            dino_png_names[:3],
            errors,
            warnings,
            f"{unit.unit_id} masks_aux:sam2/dino_output",
        )
        decoded_groups["sam2/dino_output"] = {
            "count": len(dino_names),
            "png_count": len(dino_png_names),
            "json_count": len(dino_json_names),
            "sampled": decoded_dino,
        }

    summary["decoded_groups"] = decoded_groups
    return summary


def check_sam2_meta(
    shard_path: Path,
    names: list[str],
    unit: Unit,
    context: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"file_count": len(names), "files": names}
    required = (
        ["sam2/mask_to_object.json"]
        if unit.kind in {"mos", "ego"}
        else ["sam2/iter1_faulty.txt", "sam2/usr_bbox_prompts.npy", "sam2/verify_page.txt"]
    )
    missing = [name for name in required if name not in names]
    if missing:
        errors.append(f"{unit.unit_id} sam2_meta missing files: {missing}")

    try:
        with tarfile.open(shard_path) as tar:
            if unit.kind in {"mos", "ego"} and "sam2/mask_to_object.json" in names:
                mask_to_object = read_tar_json(tar, "sam2/mask_to_object.json")
                summary["mask_to_object_count"] = len(mask_to_object)
                if unit.kind == "mos":
                    expected_rows = context["mask_map_by_phase"].get((unit.scene_id, f"phase{unit.phase_index}"), [])
                else:
                    expected_rows = context.get("ego_mask_map_by_scene", {}).get(unit.scene_id, [])
                expected_ids = {int(row["local_mask_id"]) for row in expected_rows}
                actual_ids = {int(key) for key in mask_to_object.keys()}
                if len(expected_rows) != len(mask_to_object):
                    errors.append(
                        f"{unit.unit_id} mask map row count {len(expected_rows)}, mask_to_object count {len(mask_to_object)}"
                    )
                if expected_ids != actual_ids:
                    errors.append(f"{unit.unit_id} local mask IDs differ between manifest and sam2_meta")
            if unit.kind in {"mos", "ego"}:
                line_counts = {}
                for name in names:
                    if name.endswith(".txt") and name in names:
                        text = read_tar_text(tar, name)
                        line_counts[name] = len([line for line in text.splitlines() if line.strip()])
                summary["text_line_counts"] = line_counts
            else:
                if "sam2/usr_bbox_prompts.npy" in names:
                    extracted = tar.extractfile("sam2/usr_bbox_prompts.npy")
                    if extracted is None:
                        errors.append(f"{unit.unit_id} could not extract usr_bbox_prompts.npy")
                    else:
                        bbox = np.load(io.BytesIO(extracted.read()), allow_pickle=True)
                        summary["usr_bbox_prompts"] = {
                            "shape": list(bbox.shape),
                            "dtype": str(bbox.dtype),
                            "finite": bool(np.isfinite(bbox).all()) if np.issubdtype(bbox.dtype, np.number) else None,
                        }
                        if bbox.ndim != 2 or bbox.shape[-1] != 4:
                            errors.append(f"{unit.unit_id} usr_bbox_prompts.npy shape {bbox.shape}, expected Nx4")
                if "sam2/verify_page.txt" in names:
                    summary["verify_page"] = read_tar_text(tar, "sam2/verify_page.txt").strip()
    except Exception as exc:
        errors.append(f"{unit.unit_id} sam2_meta could not be parsed: {exc}")
    return summary


def check_questionnaire(repo_root: Path, unit: Unit, errors: list[str]) -> dict[str, Any]:
    if unit.kind != "sos" or unit.object_id is None:
        return {}
    meta_path = repo_root / "objects_meta" / unit.object_id / "metadata.json"
    questionnaire_path = repo_root / "objects_meta" / unit.object_id / "questionnaire.json"
    summary: dict[str, Any] = {}
    for label, path in [("metadata", meta_path), ("questionnaire", questionnaire_path)]:
        if not path.exists():
            errors.append(f"{unit.unit_id} missing {label}: {path}")
            continue
        try:
            data = load_json(path)
            summary[label] = {
                "path": as_rel(path, repo_root),
                "object_id": data.get("object_id"),
                "global_object_id": data.get("global_object_id"),
            }
            if data.get("object_id") != unit.object_id:
                errors.append(f"{unit.unit_id} {label} object_id mismatch")
            if data.get("global_object_id") != unit.global_object_id:
                errors.append(f"{unit.unit_id} {label} global_object_id mismatch")
            if label == "questionnaire":
                questions = data.get("questions", {})
                summary[label]["question_count"] = len(questions)
                if not questions:
                    errors.append(f"{unit.unit_id} questionnaire has no questions")
        except Exception as exc:
            errors.append(f"{unit.unit_id} could not parse {label}: {exc}")
    return summary


def check_unit(
    repo_root: Path,
    unit: Unit,
    context: dict[str, Any],
    policy: str,
) -> tuple[dict[str, Any], dict[str, Image.Image]]:
    errors: list[str] = []
    warnings: list[str] = []
    root = unit.root
    result: dict[str, Any] = {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "view": unit.view,
        "root": as_rel(root, repo_root),
        "expected_frames": unit.expected_frames,
        "errors": errors,
        "warnings": warnings,
        "shards": {},
    }
    viz: dict[str, Image.Image] = {}
    if not root.exists():
        errors.append(f"missing unit directory: {root}")

    shard_paths = {
        "rgb": root / "rgb.tar",
        "depth": root / "depth.tar",
        "fisheye": root / "fisheye.tar",
        "masks": root / "labels" / "masks" / f"{LABEL_VERSION}.tar",
        "masks_aux": root / "labels" / "masks_aux" / f"{LABEL_VERSION}.tar",
        "sam2_meta": root / "labels" / "sam2_meta" / f"{LABEL_VERSION}.tar",
        "cam_pose": root / "labels" / "cam_pose" / f"{LABEL_VERSION}.tar",
    }

    tar_names: dict[str, list[str]] = {}
    for key, path in shard_paths.items():
        names = tar_file_names(path, errors)
        tar_names[key] = names
        result["shards"][key] = {
            "path": as_rel(path, repo_root),
            "file_count": len(names),
        }

    rgb_summary, rgb_images = check_image_stream(
        shard_paths["rgb"],
        tar_names["rgb"],
        "rgb",
        unit.expected_frames,
        RGBD_SIZE,
        policy,
        errors,
        warnings,
        f"{unit.unit_id} rgb",
    )
    result["shards"]["rgb"].update(rgb_summary)
    if rgb_images:
        viz["rgb"] = rgb_images[pick_names(names_with_prefix(tar_names["rgb"], "rgb", ".png"), "middle")[0]]

    depth_summary, depth_images = check_image_stream(
        shard_paths["depth"],
        tar_names["depth"],
        "depth",
        unit.expected_frames,
        RGBD_SIZE,
        policy,
        errors,
        warnings,
        f"{unit.unit_id} depth",
        require_nonzero=True,
        stats_fn=depth_to_stats,
    )
    result["shards"]["depth"].update(depth_summary)
    if depth_images:
        viz["depth"] = depth_images[pick_names(names_with_prefix(tar_names["depth"], "depth", ".png"), "middle")[0]]

    mask_summary, mask_images = check_image_stream(
        shard_paths["masks"],
        tar_names["masks"],
        "sam2/masks",
        unit.expected_frames,
        RGBD_SIZE,
        policy,
        errors,
        warnings,
        f"{unit.unit_id} masks",
        require_nonzero=True,
    )
    result["shards"]["masks"].update(mask_summary)
    mask_sample_stats = result["shards"]["masks"].get("sampled", [])
    for item in mask_sample_stats:
        # Mask PNGs are integer maps; count nonzero IDs on sampled frames.
        name = item["name"]
        image = mask_images.get(name)
        if image is not None:
            values = np.unique(np.asarray(image))
            item["nonzero_instance_ids"] = int(np.count_nonzero(values))
    if mask_images:
        viz["mask"] = mask_images[pick_names(names_with_prefix(tar_names["masks"], "sam2/masks", ".png"), "middle")[0]]

    fisheye_left_summary, fisheye_left_images = check_image_stream(
        shard_paths["fisheye"],
        tar_names["fisheye"],
        "fisheye/left",
        unit.expected_frames,
        FISHEYE_SIZE,
        policy,
        errors,
        warnings,
        f"{unit.unit_id} fisheye_left",
    )
    fisheye_right_summary, fisheye_right_images = check_image_stream(
        shard_paths["fisheye"],
        tar_names["fisheye"],
        "fisheye/right",
        unit.expected_frames,
        FISHEYE_SIZE,
        policy,
        errors,
        warnings,
        f"{unit.unit_id} fisheye_right",
    )
    result["shards"]["fisheye"].update(
        {
            "left": fisheye_left_summary,
            "right": fisheye_right_summary,
        }
    )
    left_names = names_with_prefix(tar_names["fisheye"], "fisheye/left", ".png")
    right_names = names_with_prefix(tar_names["fisheye"], "fisheye/right", ".png")
    if fisheye_left_images and left_names:
        viz["fisheye_left"] = fisheye_left_images[pick_names(left_names, "middle")[0]]
    if fisheye_right_images and right_names:
        viz["fisheye_right"] = fisheye_right_images[pick_names(right_names, "middle")[0]]

    cam_pose_names = names_with_prefix(tar_names["cam_pose"], "cam_pose", ".npz")
    cam_sequence = numbered_sequence_summary(
        cam_pose_names,
        "cam_pose",
        ".npz",
        unit.expected_frames,
        errors,
        f"{unit.unit_id} cam_pose",
    )
    result["shards"]["cam_pose"].update(
        {
            **cam_sequence,
            "sampled": decode_npz_samples(
                shard_paths["cam_pose"],
                pick_names(cam_pose_names, policy),
                errors,
                warnings,
                f"{unit.unit_id} cam_pose",
            ),
        }
    )

    result["shards"]["masks_aux"].update(
        check_masks_aux(shard_paths["masks_aux"], tar_names["masks_aux"], unit, policy, errors, warnings)
    )
    result["shards"]["sam2_meta"].update(
        check_sam2_meta(shard_paths["sam2_meta"], tar_names["sam2_meta"], unit, context, errors, warnings)
    )
    if unit.kind == "sos":
        result["object_metadata"] = check_questionnaire(repo_root, unit, errors)

    frame_counts = {
        "rgb": result["shards"]["rgb"].get("count"),
        "depth": result["shards"]["depth"].get("count"),
        "masks": result["shards"]["masks"].get("count"),
        "cam_pose": result["shards"]["cam_pose"].get("count"),
        "fisheye_left": result["shards"]["fisheye"].get("left", {}).get("count"),
        "fisheye_right": result["shards"]["fisheye"].get("right", {}).get("count"),
    }
    if any(count != unit.expected_frames for count in frame_counts.values()):
        errors.append(f"{unit.unit_id} frame count alignment failed: {frame_counts}")
    result["frame_counts"] = frame_counts
    result["status"] = "error" if errors else ("warning" if warnings else "ok")
    return result, viz


def write_markdown_summary(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# RPX All-Modality QA Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Units checked: `{summary['units_checked']}`",
        f"- MOS units: `{summary['unit_counts'].get('mos', 0)}`",
        f"- Ego units: `{summary['unit_counts'].get('ego', 0)}`",
        f"- SOS units: `{summary['unit_counts'].get('sos', 0)}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        f"- Rerun recording: `{summary.get('rrd_path')}`",
        f"- Contact sheet: `{summary.get('contact_sheet_path')}`",
        "",
        "## Coverage",
        "",
        "- RGB, depth, masks, fisheye left/right, camera poses, masks_aux, and SAM2 metadata shards.",
        "- MOS split tables, scene ESD weights/tier JSON, Data Studio preview images, ego previews, and mask-object manifest joins.",
        "- SOS selected-object table, object catalog, object metadata, and questionnaires.",
        "- Parquet row counts and frame-manifest unit counts when pyarrow is available.",
        "",
    ]
    if summary["errors_by_unit"]:
        lines.extend(["## Error Examples", ""])
        for item in summary["errors_by_unit"][:20]:
            lines.append(f"- `{item['unit_id']}`: {item['errors'][0]}")
        lines.append("")
    if summary["warnings_by_unit"]:
        lines.extend(["## Warning Examples", ""])
        for item in summary["warnings_by_unit"][:20]:
            lines.append(f"- `{item['unit_id']}`: {item['warnings'][0]}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = {item.strip() for item in args.kinds.split(",") if item.strip()}
    invalid_kinds = kinds - {"mos", "sos", "ego"}
    if invalid_kinds:
        raise SystemExit(f"invalid --kinds values: {sorted(invalid_kinds)}")

    units, context = build_units(repo_root, kinds)
    if args.limit is not None:
        units = units[: args.limit]

    report_path = out_dir / "all_modalities_report.json"
    markdown_path = out_dir / "all_modalities_summary.md"
    contact_sheet_path = out_dir / "all_modalities_contact_sheet.jpg"
    rrd_path = out_dir / "rpx_all_modalities_visual_check.rrd"

    global_summary = check_global_manifests(repo_root, units, context)
    rr = None if args.skip_rerun else open_rerun(rrd_path)
    tiles: list[Image.Image] = []
    results: list[dict[str, Any]] = []
    for index, unit in enumerate(tqdm(units, desc="checking RPX units")):
        result, viz = check_unit(repo_root, unit, context, args.sample_policy)
        results.append(result)
        if not args.skip_contact_sheet:
            rendered_viz = dict(viz)
            if "depth" in rendered_viz:
                rendered_viz["depth"] = depth_to_uint8(rendered_viz["depth"])
            if "mask" in rendered_viz:
                rendered_viz["mask"] = colorize_mask(rendered_viz["mask"])
            tiles.append(make_unit_tile(unit, result, rendered_viz, args.contact_thumb_width))
        if rr is not None:
            log_to_rerun(rr, index, unit, result, viz, args.image_width)

    if tiles:
        make_contact_sheet(tiles, contact_sheet_path, args.contact_cols)

    unit_counts = Counter(result["kind"] for result in results)
    errors_by_unit = [
        {"unit_id": result["unit_id"], "errors": result["errors"]}
        for result in results
        if result["errors"]
    ]
    warnings_by_unit = [
        {"unit_id": result["unit_id"], "warnings": result["warnings"]}
        for result in results
        if result["warnings"]
    ]
    error_count = sum(len(result["errors"]) for result in results) + len(global_summary["errors"])
    warning_count = sum(len(result["warnings"]) for result in results) + len(global_summary["warnings"])
    status = "error" if error_count else ("warning" if warning_count else "ok")
    summary = {
        "repo_root": str(repo_root),
        "sample_policy": args.sample_policy,
        "status": status,
        "units_checked": len(results),
        "unit_counts": dict(unit_counts),
        "error_count": error_count,
        "warning_count": warning_count,
        "global": global_summary,
        "errors_by_unit": errors_by_unit,
        "warnings_by_unit": warnings_by_unit,
        "rrd_path": str(rrd_path) if rr is not None else None,
        "contact_sheet_path": str(contact_sheet_path) if tiles else None,
        "results": results,
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, markdown_path)

    print(f"status: {status}")
    print(f"units checked: {len(results)}")
    print(f"errors: {error_count}")
    print(f"warnings: {warning_count}")
    print(f"wrote {report_path}")
    print(f"wrote {markdown_path}")
    if tiles:
        print(f"wrote {contact_sheet_path}")
    if rr is not None:
        print(f"wrote {rrd_path}")
    if error_count or (warning_count and args.fail_on_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

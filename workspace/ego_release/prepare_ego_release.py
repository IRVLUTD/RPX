#!/usr/bin/env python3
"""Prepare egocentric MOS captures for the RPX Hugging Face dataset layout.

The source tree is expected to contain one or more capture directories with
siblings such as rgb/, depth/, fisheye/, cam_pose/, and sam2/. Each capture is
packed into scenes/<scene_id>/ego/ and the HF-facing manifest, preview, and
parquet files are regenerated.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_VERSION = "v1"
EGO_VIEW = "ego"
RGBD_SIZE = (640, 480)
FISHEYE_SIZE = (848, 800)
PHASE_NAMES = {0: "clutter", 1: "interaction", 2: "clean"}
DEPTH_MM_TO_M = 0.001
DEPTH_SATURATED_MM = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class Capture:
    source_root: Path
    scene_id: str
    target_root: Path
    frame_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="RPX dataset repository root.",
    )
    parser.add_argument(
        "--ego-source-root",
        type=Path,
        required=True,
        help="Root containing raw ego captures, e.g. ego_example.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write tars/manifests/previews. Without this, only prints the inferred captures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing scenes/<scene>/ego directory.",
    )
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=None,
        help="Frame index for preview images. Default: middle frame per capture.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root))


def scene_number_map(repo_root: Path) -> dict[str, str]:
    rows = read_csv_rows(repo_root / "manifest" / "scene_name_mapping_v1.csv")
    return {row["scene_number"]: row["new_scene_name"] for row in rows}


def infer_scene_id(capture_root: Path, source_root: Path, number_map: dict[str, str]) -> str:
    parts = capture_root.relative_to(source_root).parts
    for part in parts:
        if part.startswith("scene") and part[5:].isdigit():
            return f"scene{int(part[5:]):03d}"
        if part.isdigit():
            return number_map.get(part, f"scene{int(part):03d}")
    raise ValueError(f"could not infer scene id from {capture_root}")


def numeric_files(path: Path, suffix: str) -> list[Path]:
    return sorted(
        [item for item in path.glob(f"*{suffix}") if item.stem.isdigit()],
        key=lambda item: int(item.stem),
    )


def numeric_indices(path: Path, suffix: str) -> tuple[int, ...]:
    return tuple(int(item.stem) for item in numeric_files(path, suffix))


def find_capture_roots(source_root: Path, repo_root: Path) -> list[Capture]:
    number_map = scene_number_map(repo_root)
    captures: list[Capture] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_dir():
            continue
        if not (path / "rgb").is_dir():
            continue
        if not ((path / "depth").is_dir() or (path / "fisheye").is_dir() or (path / "cam_pose").is_dir()):
            continue
        frame_indices = numeric_indices(path / "rgb", ".png")
        if not frame_indices:
            continue
        scene_id = infer_scene_id(path, source_root, number_map)
        captures.append(
            Capture(
                source_root=path,
                scene_id=scene_id,
                target_root=repo_root / "scenes" / scene_id / EGO_VIEW,
                frame_indices=frame_indices,
            )
        )
    return captures


def add_file_to_tar(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(src), arcname)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with src.open("rb") as handle:
        tar.addfile(info, handle)


def write_tar(tar_path: Path, entries: list[tuple[Path, str]]) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for src, arcname in sorted(entries, key=lambda item: item[1]):
            add_file_to_tar(tar, src, arcname)


def png_entries(directory: Path, prefix: str) -> list[tuple[Path, str]]:
    if not directory.exists():
        return []
    return [(path, f"{prefix}/{path.name}") for path in numeric_files(directory, ".png")]


def npz_entries(directory: Path, prefix: str) -> list[tuple[Path, str]]:
    if not directory.exists():
        return []
    return [(path, f"{prefix}/{path.name}") for path in numeric_files(directory, ".npz")]


def all_file_entries(directory: Path, prefix: str) -> list[tuple[Path, str]]:
    if not directory.exists():
        return []
    entries: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            entries.append((path, str(Path(prefix) / path.relative_to(directory))))
    return entries


def pack_capture(capture: Capture, repo_root: Path, overwrite: bool) -> dict[str, Any]:
    if capture.target_root.exists():
        if not overwrite:
            raise FileExistsError(f"{capture.target_root} exists; pass --overwrite to replace it")
        shutil.rmtree(capture.target_root)
    capture.target_root.mkdir(parents=True, exist_ok=True)

    source = capture.source_root
    sam2 = source / "sam2"
    shard_counts: dict[str, int] = {}

    entries = png_entries(source / "rgb", "rgb")
    write_tar(capture.target_root / "rgb.tar", entries)
    shard_counts["rgb"] = len(entries)

    entries = png_entries(source / "depth", "depth")
    if entries:
        write_tar(capture.target_root / "depth.tar", entries)
    shard_counts["depth"] = len(entries)

    fisheye_entries = png_entries(source / "fisheye" / "left", "fisheye/left")
    fisheye_entries += png_entries(source / "fisheye" / "right", "fisheye/right")
    if fisheye_entries:
        write_tar(capture.target_root / "fisheye.tar", fisheye_entries)
    shard_counts["fisheye"] = len(fisheye_entries)

    entries = npz_entries(source / "cam_pose", "cam_pose")
    if entries:
        write_tar(capture.target_root / "labels" / "cam_pose" / f"{LABEL_VERSION}.tar", entries)
    shard_counts["cam_pose"] = len(entries)

    entries = png_entries(sam2 / "masks", "sam2/masks")
    if entries:
        write_tar(capture.target_root / "labels" / "masks" / f"{LABEL_VERSION}.tar", entries)
    shard_counts["masks"] = len(entries)

    aux_entries: list[tuple[Path, str]] = []
    if sam2.exists():
        for child in sorted(sam2.iterdir()):
            if child.name == "masks" or not child.is_dir():
                continue
            aux_entries.extend(all_file_entries(child, f"sam2/{child.name}"))
    if aux_entries:
        write_tar(capture.target_root / "labels" / "masks_aux" / f"{LABEL_VERSION}.tar", aux_entries)
    shard_counts["masks_aux"] = len(aux_entries)

    meta_entries = [(path, f"sam2/{path.name}") for path in sorted(sam2.iterdir()) if path.is_file()] if sam2.exists() else []
    if meta_entries:
        write_tar(capture.target_root / "labels" / "sam2_meta" / f"{LABEL_VERSION}.tar", meta_entries)
    shard_counts["sam2_meta"] = len(meta_entries)

    return {"scene_id": capture.scene_id, "target_root": rel(capture.target_root, repo_root), "shard_counts": shard_counts}


def read_split_lookup(repo_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for split in ("easy", "medium", "hard"):
        for row in read_csv_rows(repo_root / "splits" / f"{split}.csv"):
            lookup[(row["scene_id"], int(row["phase_index"]))] = row
    return lookup


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: Any | None = None, metadata: dict[bytes, bytes] | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    if metadata:
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def ego_frame_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("scene_id", pa.string()),
            ("scene_type", pa.string()),
            ("view", pa.string()),
            ("capture_id", pa.string()),
            ("frame_idx", pa.int64()),
            ("frame_filename", pa.string()),
            ("split", pa.string()),
            ("has_rgb", pa.bool_()),
            ("has_depth", pa.bool_()),
            ("has_fisheye", pa.bool_()),
            ("has_cam_pose", pa.bool_()),
            ("has_masks", pa.bool_()),
            ("has_masks_aux", pa.bool_()),
            ("has_sam2_meta", pa.bool_()),
            ("shard_rgb", pa.string()),
            ("shard_depth", pa.string()),
            ("shard_fisheye", pa.string()),
            ("shard_cam_pose", pa.string()),
            ("shard_masks", pa.string()),
            ("shard_masks_aux", pa.string()),
            ("shard_sam2_meta", pa.string()),
        ]
    )


def ego_mask_map_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("scene_id", pa.string()),
            ("view", pa.string()),
            ("capture_id", pa.string()),
            ("local_mask_id", pa.int64()),
            ("object_id", pa.string()),
            ("global_object_id", pa.int64()),
            ("source_catalog_id", pa.string()),
            ("object_name", pa.string()),
            ("class_name", pa.string()),
            ("questionnaire_path", pa.string()),
            ("sos_path", pa.string()),
            ("sos_data_path", pa.string()),
            ("mask_source", pa.string()),
        ]
    )


def build_ego_frame_rows(repo_root: Path, captures: list[Capture]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capture in captures:
        root = rel(capture.target_root, repo_root)
        has_depth = (capture.target_root / "depth.tar").exists()
        has_fisheye = (capture.target_root / "fisheye.tar").exists()
        has_cam_pose = (capture.target_root / "labels" / "cam_pose" / f"{LABEL_VERSION}.tar").exists()
        has_masks = (capture.target_root / "labels" / "masks" / f"{LABEL_VERSION}.tar").exists()
        has_masks_aux = (capture.target_root / "labels" / "masks_aux" / f"{LABEL_VERSION}.tar").exists()
        has_sam2_meta = (capture.target_root / "labels" / "sam2_meta" / f"{LABEL_VERSION}.tar").exists()
        for frame_idx in capture.frame_indices:
            filename = f"{frame_idx:05d}.png"
            rows.append(
                {
                    "scene_id": capture.scene_id,
                    "scene_type": "multi_object_ego",
                    "view": EGO_VIEW,
                    "capture_id": EGO_VIEW,
                    "frame_idx": frame_idx,
                    "frame_filename": filename,
                    "split": EGO_VIEW,
                    "has_rgb": True,
                    "has_depth": has_depth,
                    "has_fisheye": has_fisheye,
                    "has_cam_pose": has_cam_pose,
                    "has_masks": has_masks,
                    "has_masks_aux": has_masks_aux,
                    "has_sam2_meta": has_sam2_meta,
                    "shard_rgb": f"{root}/rgb.tar",
                    "shard_depth": f"{root}/depth.tar" if has_depth else "",
                    "shard_fisheye": f"{root}/fisheye.tar" if has_fisheye else "",
                    "shard_cam_pose": f"{root}/labels/cam_pose/{LABEL_VERSION}.tar" if has_cam_pose else "",
                    "shard_masks": f"{root}/labels/masks/{LABEL_VERSION}.tar" if has_masks else "",
                    "shard_masks_aux": f"{root}/labels/masks_aux/{LABEL_VERSION}.tar" if has_masks_aux else "",
                    "shard_sam2_meta": f"{root}/labels/sam2_meta/{LABEL_VERSION}.tar" if has_sam2_meta else "",
                }
            )
    return rows


def build_frames_v2_rows(repo_root: Path, ego_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    split_lookup = read_split_lookup(repo_root)
    old_rows = pq.read_table(repo_root / "manifest" / "frames_v1.parquet").to_pylist()
    rows: list[dict[str, Any]] = []
    for row in old_rows:
        scene_type = row["scene_type"]
        phase = row["phase"]
        phase_index = int(phase) if scene_type == "multi_object" else None
        phase_info = split_lookup.get((row["scene_id"], phase_index), {}) if phase_index is not None else {}
        capture_id = f"phase{phase_index}" if phase_index is not None else str(phase)
        rows.append(
            {
                **row,
                "view": "phase" if scene_type == "multi_object" else "object",
                "capture_id": capture_id,
                "phase_index": phase_index,
                "phase_name": PHASE_NAMES.get(phase_index) if phase_index is not None else None,
                "difficulty": phase_info.get("difficulty"),
                "rpx_ds": float(phase_info["rpx_ds"]) if phase_info.get("rpx_ds") else None,
            }
        )
    for row in ego_rows:
        rows.append(
            {
                **row,
                "phase": None,
                "phase_index": None,
                "phase_name": None,
                "difficulty": None,
                "rpx_ds": None,
            }
        )
    return rows


def object_catalog_lookup(repo_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    catalog = load_json(repo_root / "manifest" / "object_catalog_v1.json")
    by_source = {str(item["source_catalog_id"]): item for item in catalog.get("objects", [])}
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in catalog.get("objects", []):
        by_name[str(item["object_name"])].append(item)
    return by_source, by_name


def read_mask_to_object_from_tar(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with tarfile.open(path) as tar:
        extracted = tar.extractfile("sam2/mask_to_object.json")
        if extracted is None:
            return None
        return json.load(extracted)


def build_ego_mask_map_rows(repo_root: Path, captures: list[Capture]) -> list[dict[str, Any]]:
    by_source, by_name = object_catalog_lookup(repo_root)
    rows: list[dict[str, Any]] = []
    for capture in captures:
        meta_tar = capture.target_root / "labels" / "sam2_meta" / f"{LABEL_VERSION}.tar"
        mask_to_object = read_mask_to_object_from_tar(meta_tar)
        if not mask_to_object:
            continue
        for mask_id, item in sorted(mask_to_object.items(), key=lambda kv: int(kv[0])):
            source_catalog_id = str(item["object"]["id"])
            object_name = str(item["object"]["name"])
            catalog_item = by_source.get(source_catalog_id)
            if catalog_item is None:
                matches = by_name.get(object_name, [])
                catalog_item = matches[0] if len(matches) == 1 else None
            if catalog_item is None:
                raise KeyError(
                    f"could not join ego mask {capture.scene_id}:{mask_id} "
                    f"({source_catalog_id}, {object_name}) to object_catalog_v1.json"
                )
            rows.append(
                {
                    "scene_id": capture.scene_id,
                    "view": EGO_VIEW,
                    "capture_id": EGO_VIEW,
                    "local_mask_id": int(mask_id),
                    "object_id": catalog_item["object_id"],
                    "global_object_id": int(catalog_item["global_object_id"]),
                    "source_catalog_id": str(catalog_item["source_catalog_id"]),
                    "object_name": catalog_item["object_name"],
                    "class_name": catalog_item.get("class_name", catalog_item["object_name"]),
                    "questionnaire_path": catalog_item["questionnaire_path"],
                    "sos_path": catalog_item.get("sos_path", catalog_item["sos_data_path"]),
                    "sos_data_path": catalog_item["sos_data_path"],
                    "mask_source": f"{rel(meta_tar, repo_root)}:sam2/mask_to_object.json",
                }
            )
    return rows


def depth_to_uint8(depth: Image.Image) -> Image.Image:
    arr_mm = np.asarray(depth).astype(np.float32)
    valid = (arr_mm > 0) & (arr_mm < DEPTH_SATURATED_MM)
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
        ],
        dtype=np.uint8,
    )
    for value in np.unique(arr):
        if value == 0:
            continue
        rgb[arr == value] = palette[(int(value) - 1) % len(palette)]
    return Image.fromarray(rgb, mode="RGB")


def fit_image(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    resized = image.resize(new_size, method)
    canvas = Image.new("RGB", size, (14, 17, 22))
    canvas.paste(resized, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))
    return canvas


def read_png_from_tar(path: Path, name: str) -> Image.Image | None:
    if not path.exists():
        return None
    with tarfile.open(path) as tar:
        extracted = tar.extractfile(name)
        if extracted is None:
            return None
        return Image.open(extracted).copy()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Times New Roman.ttf", "Times_New_Roman.ttf", "LiberationSerif-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_ego_preview(capture: Capture, repo_root: Path, frame_idx: int, out_path: Path) -> None:
    frame = f"{frame_idx:05d}.png"
    rgb = read_png_from_tar(capture.target_root / "rgb.tar", f"rgb/{frame}")
    depth = read_png_from_tar(capture.target_root / "depth.tar", f"depth/{frame}")
    mask = read_png_from_tar(capture.target_root / "labels" / "masks" / f"{LABEL_VERSION}.tar", f"sam2/masks/{frame}")
    left = read_png_from_tar(capture.target_root / "fisheye.tar", f"fisheye/left/{frame}")
    right = read_png_from_tar(capture.target_root / "fisheye.tar", f"fisheye/right/{frame}")

    panels = [
        ("RGB", rgb),
        ("Depth", depth_to_uint8(depth) if depth is not None else None),
        ("Masks", colorize_mask(mask) if mask is not None else None),
        ("Fisheye L", left),
        ("Fisheye R", right),
    ]
    panel_size = (320, 240)
    label_h = 32
    footer_h = 46
    image = Image.new("RGB", (panel_size[0] * len(panels), panel_size[1] + label_h + footer_h), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    small = load_font(15)
    for index, (label, panel) in enumerate(panels):
        x = index * panel_size[0]
        if panel is None:
            tile = Image.new("RGB", panel_size, (40, 43, 50))
            ImageDraw.Draw(tile).text((12, 12), "missing", fill=(235, 235, 235), font=small)
        else:
            tile = fit_image(panel, panel_size, nearest=label in {"Depth", "Masks"})
        image.paste(tile, (x, label_h))
        draw.text((x + 10, 7), label, fill=(21, 25, 32), font=font)
    footer_y = label_h + panel_size[1]
    draw.rectangle((0, footer_y, image.width, image.height), fill=(18, 23, 30))
    draw.text((14, footer_y + 10), f"RPX EGO / {capture.scene_id} / frame {frame_idx:05d}", fill=(235, 238, 242), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=90)


def build_ego_preview_rows(repo_root: Path, captures: list[Capture], preview_frame: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capture in captures:
        frame_idx = preview_frame if preview_frame is not None else capture.frame_indices[len(capture.frame_indices) // 2]
        out_name = f"rpx_ego_{capture.scene_id}.jpg"
        preview_path = repo_root / "preview" / "image_examples" / "ego_preview" / "images" / out_name
        make_ego_preview(capture, repo_root, frame_idx, preview_path)
        root = rel(capture.target_root, repo_root)
        with preview_path.open("rb") as handle:
            image_bytes = handle.read()
        rows.append(
            {
                "image": {"bytes": image_bytes, "path": out_name},
                "scene_id": capture.scene_id,
                "view": EGO_VIEW,
                "capture_id": EGO_VIEW,
                "frame_count": len(capture.frame_indices),
                "preview_frame_index": frame_idx,
                "source_rgb_shard": f"{root}/rgb.tar",
                "source_depth_shard": f"{root}/depth.tar",
                "source_fisheye_shard": f"{root}/fisheye.tar",
                "source_cam_pose_shard": f"{root}/labels/cam_pose/{LABEL_VERSION}.tar",
                "source_mask_shard": f"{root}/labels/masks/{LABEL_VERSION}.tar",
                "source_mask_aux_shard": f"{root}/labels/masks_aux/{LABEL_VERSION}.tar",
                "source_sam2_meta_shard": f"{root}/labels/sam2_meta/{LABEL_VERSION}.tar",
                "caption": f"Egocentric RGB-D/mask preview for {capture.scene_id} ({len(capture.frame_indices)} frames).",
            }
        )
    return rows


def write_ego_preview_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa

    schema = pa.schema(
        [
            ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("scene_id", pa.string()),
            ("view", pa.string()),
            ("capture_id", pa.string()),
            ("frame_count", pa.int64()),
            ("preview_frame_index", pa.int64()),
            ("source_rgb_shard", pa.string()),
            ("source_depth_shard", pa.string()),
            ("source_fisheye_shard", pa.string()),
            ("source_cam_pose_shard", pa.string()),
            ("source_mask_shard", pa.string()),
            ("source_mask_aux_shard", pa.string()),
            ("source_sam2_meta_shard", pa.string()),
            ("caption", pa.string()),
        ]
    )
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "image": {"_type": "Image"},
                        "scene_id": {"dtype": "string", "_type": "Value"},
                        "view": {"dtype": "string", "_type": "Value"},
                        "capture_id": {"dtype": "string", "_type": "Value"},
                        "frame_count": {"dtype": "int64", "_type": "Value"},
                        "preview_frame_index": {"dtype": "int64", "_type": "Value"},
                        "caption": {"dtype": "string", "_type": "Value"},
                    }
                }
            }
        ).encode("utf-8")
    }
    write_parquet(path, rows, schema=schema, metadata=metadata)


def write_csv_preview(path: Path, rows: list[dict[str, Any]], repo_id: str = "IRVLUTD/RPX") -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        out = {key: value for key, value in row.items() if key != "image"}
        out["image_preview_url"] = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
            f"preview/image_examples/ego_preview/images/{row['image']['path']}"
        )
        csv_rows.append(out)
    fieldnames = [
        "scene_id",
        "view",
        "capture_id",
        "frame_count",
        "preview_frame_index",
        "image_preview_url",
        "source_rgb_shard",
        "source_depth_shard",
        "source_fisheye_shard",
        "source_cam_pose_shard",
        "source_mask_shard",
        "source_mask_aux_shard",
        "source_sam2_meta_shard",
        "caption",
    ]
    write_csv_rows(path, csv_rows, fieldnames)


def write_current_json(repo_root: Path, ego_rows: list[dict[str, Any]], ego_mask_rows: list[dict[str, Any]]) -> None:
    path = repo_root / "manifest" / "current.json"
    current = load_json(path)
    current["schema_version"] = "v2"
    current.setdefault("manifests", {}).update(
        {
            "frames_v2": "manifest/frames_v2.parquet",
            "ego_frames": "manifest/ego_frames_v1.parquet",
            "ego_frames_csv": "manifest/ego_frames_v1.csv",
            "mos_ego_mask_object_map": "manifest/mos_ego_mask_object_map_v1.parquet",
            "mos_ego_mask_object_map_csv": "manifest/mos_ego_mask_object_map_v1.csv",
        }
    )
    mos = current.setdefault("mos", {})
    mos["schema_version"] = "v2"
    mos["ego_directory_value"] = EGO_VIEW
    mos["ego_contributes_to_esd"] = False
    mos["ego_view_count"] = len({row["scene_id"] for row in ego_rows})
    mos["ego_frame_count"] = len(ego_rows)
    mos["ego_mask_object_row_count"] = len(ego_mask_rows)
    mos["ego_frames"] = "manifest/ego_frames_v1.parquet"
    mos["ego_mask_object_map_csv"] = "manifest/mos_ego_mask_object_map_v1.csv"
    mos["ego_mask_object_map_parquet"] = "manifest/mos_ego_mask_object_map_v1.parquet"
    mos["ego_raw_source"] = "scenes/<scene_id>/ego/labels/sam2_meta/v1.tar:sam2/mask_to_object.json"
    mos["phase_directory_values"] = ["0", "1", "2"]
    mos["view_directory_values"] = ["0", "1", "2", EGO_VIEW]
    current.setdefault("metadata_versions", {})["ego_frames"] = "v1"
    current["metadata_versions"]["mos_ego_mask_object_map"] = "v1"
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def copy_mos_phase_preview(repo_root: Path) -> None:
    src = repo_root / "preview" / "media_preview.parquet"
    dst = repo_root / "preview" / "mos_phase_preview.parquet"
    if src.exists():
        shutil.copy2(src, dst)
    csv_src = repo_root / "preview" / "data_studio_preview.csv"
    csv_dst = repo_root / "preview" / "mos_phase_preview.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, csv_dst)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_root = args.ego_source_root.resolve()
    captures = find_capture_roots(source_root, repo_root)
    if not captures:
        raise SystemExit(f"no ego captures found under {source_root}")

    print("inferred ego captures:")
    for capture in captures:
        print(f"  {capture.source_root} -> {capture.target_root} ({len(capture.frame_indices)} frames)")

    if not args.apply:
        print("dry run only; pass --apply to write dataset artifacts")
        return

    package_summary = [pack_capture(capture, repo_root, args.overwrite) for capture in captures]
    ego_frame_rows = build_ego_frame_rows(repo_root, captures)
    frames_v2_rows = build_frames_v2_rows(repo_root, ego_frame_rows)
    ego_mask_rows = build_ego_mask_map_rows(repo_root, captures)
    ego_preview_rows = build_ego_preview_rows(repo_root, captures, args.preview_frame)

    frame_fields = list(ego_frame_rows[0].keys())
    write_csv_rows(repo_root / "manifest" / "ego_frames_v1.csv", ego_frame_rows, frame_fields)
    write_parquet(repo_root / "manifest" / "ego_frames_v1.parquet", ego_frame_rows, schema=ego_frame_schema())
    write_parquet(repo_root / "manifest" / "frames_v2.parquet", frames_v2_rows)

    mask_fields = [
        "scene_id",
        "view",
        "capture_id",
        "local_mask_id",
        "object_id",
        "global_object_id",
        "source_catalog_id",
        "object_name",
        "class_name",
        "questionnaire_path",
        "sos_path",
        "sos_data_path",
        "mask_source",
    ]
    write_csv_rows(repo_root / "manifest" / "mos_ego_mask_object_map_v1.csv", ego_mask_rows, mask_fields)
    write_parquet(
        repo_root / "manifest" / "mos_ego_mask_object_map_v1.parquet",
        ego_mask_rows,
        schema=ego_mask_map_schema(),
    )

    write_ego_preview_parquet(repo_root / "preview" / "ego_preview.parquet", ego_preview_rows)
    write_csv_preview(repo_root / "preview" / "ego_preview.csv", ego_preview_rows)
    copy_mos_phase_preview(repo_root)
    write_current_json(repo_root, ego_frame_rows, ego_mask_rows)

    summary = {
        "captures": package_summary,
        "ego_frame_rows": len(ego_frame_rows),
        "frames_v2_rows": len(frames_v2_rows),
        "ego_mask_object_rows": len(ego_mask_rows),
        "ego_preview_rows": len(ego_preview_rows),
        "ego_contributes_to_esd": False,
    }
    out_path = repo_root / "ego_release" / "ego_release_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

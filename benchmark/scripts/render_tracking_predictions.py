#!/usr/bin/env python3
"""Render persisted RPX instance masks over their exact manifest RGB frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    return parser.parse_args()


def _rgb_index(cache_root: Path) -> dict[tuple[str, str, str], Path]:
    index: dict[tuple[str, str, str], Path] = {}
    manifests = sorted(cache_root.rglob("manifests/object_tracking/easy.json"))
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        snapshot_root = manifest_path.parents[2]
        for sample in payload.get("samples") or []:
            relative = Path(str(sample["rgb"]))
            rgb_path = relative if relative.is_absolute() else snapshot_root / relative
            if rgb_path.is_file():
                index[
                    (
                        str(sample["scene_id"]),
                        str(sample["phase"]),
                        rgb_path.stem,
                    )
                ] = rgb_path
    return index


def _colour(instance_id: int) -> np.ndarray:
    return np.asarray(
        [
            64 + (37 * instance_id) % 192,
            64 + (83 * instance_id) % 192,
            64 + (149 * instance_id) % 192,
        ],
        dtype=np.uint8,
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    prediction_root = output_dir / "predictions"
    preview_root = output_dir / "prediction_frames"
    predictions = sorted(prediction_root.glob("*/*/*.npz"))
    if not predictions:
        raise SystemExit(f"No predictions found under {prediction_root}")

    rgb_index = _rgb_index(Path(args.cache_dir))
    if not rgb_index:
        raise SystemExit("No usable RGB entries found in cached Easy tracking manifests.")

    rows: list[dict[str, str]] = []
    for prediction_path in predictions:
        scene, phase, filename = prediction_path.relative_to(prediction_root).parts
        frame = Path(filename).stem
        rgb_path = rgb_index.get((scene, phase, frame))
        if rgb_path is None:
            raise SystemExit(f"Manifest has no RGB mapping for {scene}/{phase}/{frame}")

        with np.load(prediction_path, allow_pickle=False) as archive:
            mask = archive["mask"].astype(np.int32)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != mask.shape:
            raise SystemExit(
                f"Shape mismatch for {scene}/{phase}/{frame}: "
                f"RGB={rgb.shape[:2]}, mask={mask.shape}"
            )

        overlay = rgb.copy()
        object_ids = [int(value) for value in np.unique(mask) if value > 0]
        for object_id in object_ids:
            selected = mask == object_id
            overlay[selected] = (0.55 * rgb[selected] + 0.45 * _colour(object_id)).astype(np.uint8)

        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        for object_id in object_ids:
            ys, xs = np.where(mask == object_id)
            if len(xs):
                draw.text(
                    (int(xs.mean()), int(ys.mean())),
                    str(object_id),
                    fill=(255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

        destination = preview_root / scene / phase / f"{frame}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, compress_level=4)
        rows.append(
            {
                "scene": scene,
                "phase": phase,
                "frame": frame,
                "rgb": str(rgb_path),
                "prediction": str(prediction_path),
                "preview": str(destination),
            }
        )

    (preview_root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(rows)} prediction frames into {preview_root}")


if __name__ == "__main__":
    main()

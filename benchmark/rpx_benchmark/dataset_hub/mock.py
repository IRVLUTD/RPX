"""Synthetic mock generator that mimics the on-disk capture layout.

Lets us validate the entire dataset-hub pipeline (scan → pack → upload →
download → load) without touching the real ~890 GB capture set on the
target system. The mock is deliberately tiny (small images, few frames)
so a full end-to-end test runs in seconds.

Layout produced (matches the ``mos/`` + ``sos/`` wireframe under
``benchmark/templates/rpx_capture_wireframe/``)::

    <root>/
    ├── mos/
    │   └── scene1/
    │       ├── 0/
    │       │   ├── cam_pose/      *.json
    │       │   ├── depth/         00000.png ... (16-bit grayscale)
    │       │   ├── fisheye/       00000.png ... (8-bit grayscale)
    │       │   ├── rgb/           00000.png ... (8-bit RGB)
    │       │   └── sam2/
    │       │       ├── masks/                       00000.png ...
    │       │       ├── bbox_overlay/                00000.png ...
    │       │       ├── contour_gt_masks/            00000.png ...
    │       │       ├── dino_output/                 00000.png ...
    │       │       ├── masks_contour_with_hidden/   00000.png ...
    │       │       ├── palette/                     00000.png ...
    │       │       ├── rgb_and_mask/                00000.png ...
    │       │       ├── mask_to_object.json
    │       │       ├── verified_masks.txt
    │       │       └── iter1_faulty.txt ... iter4_faulty.txt
    │       ├── 1/   (same structure)
    │       └── 2/
    └── sos/
        └── mock_obj_01/                   # bare object name, no "object" prefix
            └── 0/
                (same modality dirs, one phase only)
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------- #
# Tiny pure-Python PNG writer
#
# We avoid pulling in PIL/numpy for the mock — the goal is to generate
# many small files quickly with zero optional deps. Each function emits
# a valid 8-bit-grayscale, 8-bit-RGB, or 16-bit-grayscale PNG.
# --------------------------------------------------------------------- #


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _png(width: int, height: int, bit_depth: int, color_type: int, raw_rows: List[bytes]) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in raw_rows)  # filter byte = 0
    idat = zlib.compress(raw, 6)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _grayscale8_png(width: int, height: int, value: int) -> bytes:
    row = bytes([value]) * width
    return _png(width, height, 8, 0, [row] * height)


def _rgb8_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    row = bytes([r, g, b]) * width
    return _png(width, height, 8, 2, [row] * height)


def _grayscale16_png(width: int, height: int, value: int) -> bytes:
    hi, lo = (value >> 8) & 0xFF, value & 0xFF
    row = bytes([hi, lo]) * width
    return _png(width, height, 16, 0, [row] * height)


# --------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class MockSpec:
    """Knobs for the mock generator. Defaults give a ~1 MB tree."""

    multi_object_scenes: int = 3
    single_object_scenes: int = 5
    phases_per_multi: int = 3
    frames_per_phase: int = 4
    image_size: int = 32
    sam2_aux_subdirs: tuple[str, ...] = (
        "bbox_overlay",
        "contour_gt_masks",
        "dino_output",
        "masks_contour_with_hidden",
        "palette",
        "rgb_and_mask",
    )
    sam2_iter_files: int = 4


# --------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------- #


def generate_mock(out_root: Path, spec: MockSpec | None = None) -> Path:
    """Materialise a synthetic capture tree under ``out_root``.

    The tree is *deterministic*: re-running with the same spec produces
    identical bytes (so test assertions on file hashes are stable).
    """
    spec = spec or MockSpec()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mos").mkdir(exist_ok=True)
    (out_root / "sos").mkdir(exist_ok=True)

    # SOS first so MOS scenes can reference real SOS object names.
    sos_object_names: list[str] = []
    for i in range(1, spec.single_object_scenes + 1):
        # SOS scene dir name = the bare object name (no "object" prefix).
        name = f"mock_obj_{i:02d}"
        sos_object_names.append(name)
        obj_dir = out_root / "sos" / name
        _emit_phase(obj_dir / "0", spec, salt=10_000 + i, mask_to_object={"1": name})
        # FewSOL-style questionnaire at the SOS scene root.
        (obj_dir / "questionnaire.txt").write_text(
            _mock_questionnaire(name),
            encoding="utf-8",
        )

    # MOS scenes — reference SOS objects in mask_to_object.json so the
    # questionnaire-dedup layer has something real to look up.
    for i in range(1, spec.multi_object_scenes + 1):
        scene_dir = out_root / "mos" / f"scene{i}"
        # Pick up to 3 of the SOS objects to populate this scene with.
        if sos_object_names:
            picked = sos_object_names[
                (i - 1) % len(sos_object_names) : (i - 1) % len(sos_object_names) + 3
            ]
            if not picked:
                picked = sos_object_names[:3]
        else:
            picked = []
        m2o = {str(idx + 1): name for idx, name in enumerate(picked)}
        for phase in range(spec.phases_per_multi):
            _emit_phase(
                scene_dir / str(phase), spec, salt=i * 100 + phase, mask_to_object=m2o or None
            )

    return out_root


def _mock_questionnaire(object_name: str) -> str:
    """Tiny FewSOL-style template so packer/loader tests have real text."""
    return (
        f"1. What is the name of the object in these images?\n"
        f"{object_name}, mock_alias_a, mock_alias_b\n\n"
        f"2. What is the category of the object in these images?\n"
        f"mock_category\n\n"
        f"3. What is the object in these images made of?\n"
        f"mock_material\n\n"
        f"4. What can be the object in these images used for?\n"
        f"mock_use_a, mock_use_b\n\n"
        f"5. What is the color of the object in these images?\n"
        f"mock_color_a, mock_color_b\n"
    )


def _emit_phase(
    phase_dir: Path,
    spec: MockSpec,
    salt: int,
    mask_to_object: Optional[dict[str, str]] = None,
) -> None:
    """Write all modality subdirs for one (scene, phase).

    ``mask_to_object`` overrides the synthetic ``sam2/mask_to_object.json``
    with a caller-supplied mapping. Used by :func:`generate_mock` so MOS
    scenes reference real SOS object names (the dedup key for the
    objects_meta/ layer).
    """
    rgb_dir = phase_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = phase_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    # Fisheye is the T265 stereo pair: two subdirs with synced filenames.
    fisheye_root = phase_dir / "fisheye"
    fisheye_root.mkdir(exist_ok=True)
    fisheye_left = fisheye_root / "left"
    fisheye_left.mkdir(exist_ok=True)
    fisheye_right = fisheye_root / "right"
    fisheye_right.mkdir(exist_ok=True)
    cam_pose_dir = phase_dir / "cam_pose"
    cam_pose_dir.mkdir(exist_ok=True)
    sam2_dir = phase_dir / "sam2"
    sam2_dir.mkdir(exist_ok=True)
    masks_dir = sam2_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    aux_dirs = {name: (sam2_dir / name) for name in spec.sam2_aux_subdirs}
    for d in aux_dirs.values():
        d.mkdir(exist_ok=True)

    n, sz = spec.frames_per_phase, spec.image_size
    for k in range(n):
        stem = f"{k:05d}.png"
        v = (salt + k) & 0xFF
        (rgb_dir / stem).write_bytes(_rgb8_png(sz, sz, v, (v + 50) & 0xFF, (v + 100) & 0xFF))
        (depth_dir / stem).write_bytes(_grayscale16_png(sz, sz, ((salt + k) * 37) & 0xFFFF))
        # Synced filenames: same stem in both fisheye/left/ and fisheye/right/
        (fisheye_left / stem).write_bytes(_grayscale8_png(sz, sz, (255 - v) & 0xFF))
        (fisheye_right / stem).write_bytes(_grayscale8_png(sz, sz, v))
        (masks_dir / stem).write_bytes(_grayscale8_png(sz, sz, (k % 5) + 1))
        for d in aux_dirs.values():
            (d / stem).write_bytes(_grayscale8_png(sz, sz, v))
        (cam_pose_dir / f"{k:05d}.json").write_text(
            json.dumps(
                {
                    "pose": [[1, 0, 0, 0.01 * k], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "salt": salt + k,
                }
            ),
            encoding="utf-8",
        )

    # SAM2 metadata files.
    m2o = mask_to_object or {str(i): f"mock_object_{i}" for i in range(1, 6)}
    (sam2_dir / "mask_to_object.json").write_text(
        json.dumps(m2o, sort_keys=True),
        encoding="utf-8",
    )
    (sam2_dir / "verified_masks.txt").write_text(
        "\n".join(f"{k:05d}.png" for k in range(n)) + "\n",
        encoding="utf-8",
    )
    for it in range(1, spec.sam2_iter_files + 1):
        (sam2_dir / f"iter{it}_faulty.txt").write_text(
            "" if it > 1 else "00000.png\n",
            encoding="utf-8",
        )


# --------------------------------------------------------------------- #
# Convenience: total size + file count
# --------------------------------------------------------------------- #


def measure_tree(root: Path) -> tuple[int, int]:
    """Return ``(file_count, total_bytes)`` under ``root``. Pure stdlib."""
    files, total = 0, 0
    for p in Path(root).rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return files, total

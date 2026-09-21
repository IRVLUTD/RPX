#----------------------------------------------------------------------------------------------------
# Ego (GoPro) frame sampler.
#
# The ego camera (GoPro) and the D435 rig were recorded on independent clocks
# with no timecode crossover, so attempting to sync them frame-for-frame is
# brittle and would risk corrupting the Intel data. Instead we uniformly sample
# a fixed EGO_SAMPLE_COUNT (250) frames across the *entire* ego mp4 and write
# them as JPEG q=95 by default (matches the existing interactive_gsam2 labelling
# tool which globs ``*.jpg``; lossless PNG is also supported via ``image_format``).
#
# Output layout (one scene):
#     <scene>/ego/
#         rgb/00000.jpg 00001.jpg ... NNNNN.jpg   <- 250 (or fewer) frames
#         ego_frame_map.json                       <- {NNNNN.jpg: source_frame_idx}
#
#----------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

EGO_SAMPLE_COUNT = 250        # fixed number of frames extracted from every ego video
EGO_IMAGE_FORMAT = "jpg"      # default extension — JPEG q=95 matches interactive_gsam2's glob
EGO_JPEG_QUALITY = 95         # imperceptible on H.264-decoded GoPro frames; ~5x smaller than PNG
EGO_PNG_COMPRESSION = 3       # cv2 default 3 (range 0-9; higher = smaller but slower)


def _lazy_cv2():
    try:
        import cv2  # noqa: PLC0415
        return cv2
    except ImportError as e:
        raise RuntimeError(
            "OpenCV is required for ego sampling. "
            "pip install opencv-python"
        ) from e


def uniform_sample_indices(start: int, end: int, n: int) -> List[int]:
    """Return ``n`` frame indices spread uniformly across the inclusive
    ``[start, end]`` range. Endpoints are always included; indices are unique
    and sorted. If ``n`` exceeds the range size, every index is returned (no
    duplicates).
    """
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start}).")
    if n <= 1:
        return [start]
    span = end - start + 1
    if n >= span:
        return list(range(start, end + 1))
    step = (span - 1) / (n - 1)
    return sorted({start + round(i * step) for i in range(n)})


def sample_ego_frames(
    scene_dir: Path | str,
    ego_video: Path | str,
    *,
    n_samples: Optional[int] = None,
    image_format: str = EGO_IMAGE_FORMAT,
    jpeg_quality: int = EGO_JPEG_QUALITY,
) -> List[Path]:
    """Uniformly extract frames from ``ego_video`` and save them as
    ``<scene_dir>/ego/rgb/NNNNN.{jpg,png}``.

    Parameters
    ----------
    scene_dir : path
        Scene directory. Output lands at ``<scene_dir>/ego/rgb/``.
    ego_video : path
        Path to the GoPro mp4 for this scene.
    n_samples : int, optional
        Number of frames to extract (default ``EGO_SAMPLE_COUNT`` = 250). When
        the video has fewer frames than requested, every frame is taken (no
        duplicates) and a warning is printed.
    image_format : str
        Output extension: ``"jpg"`` (default; JPEG q=``jpeg_quality``) or
        ``"png"`` (lossless). JPEG q=95 is ~5x smaller than PNG on 1080p GoPro
        frames and visually indistinguishable from the H.264-decoded source;
        it also matches the ``*.jpg`` glob used by ``interactive_gsam2``.
    jpeg_quality : int
        cv2 IMWRITE_JPEG_QUALITY value (1-100). Ignored when ``image_format='png'``.

    Returns
    -------
    list[Path]
        The written frame paths, in order.
    """
    cv2 = _lazy_cv2()
    scene_dir = Path(scene_dir)
    ego_video = Path(ego_video)

    fmt = image_format.lower().lstrip(".")
    if fmt not in ("jpg", "jpeg", "png"):
        raise ValueError(f"image_format must be 'jpg' or 'png', got {image_format!r}")
    fmt = "jpg" if fmt in ("jpg", "jpeg") else "png"
    if fmt == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    else:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, int(EGO_PNG_COMPRESSION)]

    cap = cv2.VideoCapture(str(ego_video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open ego video {ego_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"ego video {ego_video} reports {total} frames.")

    if n_samples is None:
        n_samples = EGO_SAMPLE_COUNT

    idxs = uniform_sample_indices(0, total - 1, n_samples)
    if len(idxs) < n_samples:
        print(
            f"[ego-sample] WARNING: {ego_video.name} has only {total} frames; "
            f"requested {n_samples}. Taking all {len(idxs)} (no duplicates).",
            file=sys.stderr,
        )
    want = set(idxs)

    out_dir = scene_dir / "ego" / "rgb"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sequential read (robust against codec seek jitter) + keep selected frames.
    written: List[Path] = []
    src_frames: List[int] = []
    local = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if local in want:
            out_path = out_dir / f"{len(written):05d}.{fmt}"
            cv2.imwrite(str(out_path), frame, encode_params)
            written.append(out_path)
            src_frames.append(local)
        local += 1
    cap.release()

    if not written:
        raise RuntimeError(f"sampled 0 frames from {ego_video}.")

    # Provenance sidecar: source mp4 frame index for each sampled image.
    mapping = out_dir.parent / "ego_frame_map.json"
    mapping.write_text(
        json.dumps(
            {
                "ego_video": str(ego_video),
                "ego_total_frames": total,
                "n_samples_requested": n_samples,
                "image_format": fmt,
                "jpeg_quality": jpeg_quality if fmt == "jpg" else None,
                "frames": {
                    f"{i:05d}.{fmt}": src_frames[i] for i in range(len(written))
                },
            },
            indent=2,
        )
    )
    return written


__all__ = [
    "EGO_SAMPLE_COUNT", "EGO_IMAGE_FORMAT", "EGO_JPEG_QUALITY",
    "uniform_sample_indices", "sample_ego_frames",
]

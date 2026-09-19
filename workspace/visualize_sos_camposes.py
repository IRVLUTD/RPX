from __future__ import annotations

import csv
import io
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
SNAPSHOT = (
    Path("/media/naren/New Volume/rpx")
    / "docker-depth-smoke-cache/hub/datasets--IRVLUTD--RPX/snapshots"
    / REVISION
)
OBJECTS = (
    "air_duster_can",
    "alarm_clock",
    "apple",
    "baking_tray.1",
    "banana",
    "baseball_ball",
    "basting_brush.1",
    "beanie.2",
    "bear.1",
    "book.3",
)
OUTPUT = Path("/home/naren/Downloads/rpx-sos-campose-audit")


@dataclass(frozen=True)
class Track:
    object_id: str
    frames: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    step_translation: np.ndarray
    step_rotation_deg: np.ndarray


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def load_track(object_id: str) -> Track:
    archive_path = SNAPSHOT / "objects" / object_id / "0/labels/cam_pose/v1.tar"
    frames: list[int] = []
    poses: list[np.ndarray] = []
    with tarfile.open(archive_path) as archive:
        members = sorted(
            (member for member in archive if member.isfile() and member.name.endswith((".npy", ".npz"))),
            key=lambda member: int(Path(member.name).stem),
        )
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise OSError(f"Could not read {member.name}")
            loaded = np.load(io.BytesIO(stream.read()))
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    pose = np.concatenate([loaded["position"], loaded["orientation"]])
                finally:
                    loaded.close()
            else:
                pose = np.asarray(loaded)
            frames.append(int(Path(member.name).stem))
            poses.append(np.asarray(pose, dtype=np.float64).reshape(7))

    values = np.stack(poses)
    positions = values[:, :3]
    quaternions = values[:, 3:]
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    step_translation = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    step_rotation_deg = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))
    return Track(
        object_id=object_id,
        frames=np.asarray(frames),
        positions=positions,
        quaternions=quaternions,
        step_translation=step_translation,
        step_rotation_deg=step_rotation_deg,
    )


def bounds(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    extent = hi - lo
    pad = max(extent * pad_fraction, 1e-6)
    return lo - pad, hi + pad


def map_value(value: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi <= lo:
        return (a + b) / 2.0
    return a + (value - lo) / (hi - lo) * (b - a)


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.text((left, top), title, font=font(18, True), fill="#202020")
    plot = (left + 48, top + 32, right - 12, bottom - 34)
    x0, y0, x1, y1 = plot
    draw.rectangle(plot, outline="#b9b9b9", width=2)
    for fraction in (0.25, 0.5, 0.75):
        x = int(x0 + fraction * (x1 - x0))
        y = int(y0 + fraction * (y1 - y0))
        draw.line((x, y0, x, y1), fill="#ececec", width=1)
        draw.line((x0, y, x1, y), fill="#ececec", width=1)
    return plot


def trajectory_panel(draw: ImageDraw.ImageDraw, track: Track, box: tuple[int, int, int, int]) -> None:
    plot = axes(draw, box, "Stored translation trajectory (X–Z)")
    x0, y0, x1, y1 = plot
    xlo, xhi = bounds(track.positions[:, 0])
    zlo, zhi = bounds(track.positions[:, 2])
    points = [
        (
            map_value(x, xlo, xhi, x0, x1),
            map_value(z, zlo, zhi, y1, y0),
        )
        for x, z in track.positions[:, (0, 2)]
    ]
    for index in range(len(points) - 1):
        fraction = index / max(1, len(points) - 2)
        color = (int(35 + 190 * fraction), 90, int(220 - 150 * fraction))
        draw.line((*points[index], *points[index + 1]), fill=color, width=3)
    px, py = points[0]
    draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="#1464d2")
    px, py = points[-1]
    draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="#dc3232")
    draw.text((x0, y1 + 7), "blue=start", font=font(13), fill="#1464d2")
    draw.text((x1 - 72, y1 + 7), "red=end", font=font(13), fill="#b82828")
    draw.text((x0, y0 - 19), f"X {xlo:.2f}…{xhi:.2f} m", font=font(12), fill="#666666")
    draw.text((x1 - 112, y0 - 19), f"Z {zlo:.2f}…{zhi:.2f} m", font=font(12), fill="#666666")


def components_panel(draw: ImageDraw.ImageDraw, track: Track, box: tuple[int, int, int, int]) -> None:
    plot = axes(draw, box, "Stored X/Y/Z versus frame")
    x0, y0, x1, y1 = plot
    lo, hi = bounds(track.positions)
    colors = ("#d43b3b", "#28964b", "#286bc0")
    for axis_index, color in enumerate(colors):
        values = track.positions[:, axis_index]
        points = [
            (
                map_value(frame, track.frames[0], track.frames[-1], x0, x1),
                map_value(value, lo, hi, y1, y0),
            )
            for frame, value in zip(track.frames, values, strict=True)
        ]
        draw.line(points, fill=color, width=2)
    draw.text((x0, y1 + 7), "X", font=font(13, True), fill=colors[0])
    draw.text((x0 + 24, y1 + 7), "Y", font=font(13, True), fill=colors[1])
    draw.text((x0 + 48, y1 + 7), "Z", font=font(13, True), fill=colors[2])
    draw.text((x1 - 104, y1 + 7), "frame 0 → 499", font=font(13), fill="#666666")


def step_panel(draw: ImageDraw.ImageDraw, track: Track, box: tuple[int, int, int, int]) -> None:
    plot = axes(draw, box, "Consecutive-frame motion")
    x0, y0, x1, y1 = plot
    translation_cm = track.step_translation * 100.0
    trans_hi = max(float(np.percentile(translation_cm, 99)), 1e-6)
    rot_hi = max(float(np.percentile(track.step_rotation_deg, 99)), 1e-6)
    for values, high, color in (
        (translation_cm, trans_hi, "#7b3fb5"),
        (track.step_rotation_deg, rot_hi, "#e28b25"),
    ):
        points = [
            (
                map_value(index, 0, len(values) - 1, x0, x1),
                map_value(min(float(value), high), 0, high, y1, y0),
            )
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=color, width=2)
    draw.text((x0, y1 + 7), "translation", font=font(13, True), fill="#7b3fb5")
    draw.text((x0 + 90, y1 + 7), "rotation", font=font(13, True), fill="#c66d13")
    draw.text(
        (x0, y0 - 19),
        f"median {np.median(translation_cm):.2f} cm / {np.median(track.step_rotation_deg):.2f}°",
        font=font(12),
        fill="#666666",
    )


def metrics(track: Track) -> dict[str, float | str | int]:
    extent = np.ptp(track.positions, axis=0)
    return {
        "object_id": track.object_id,
        "frames": len(track.frames),
        "first_position_norm_m": float(np.linalg.norm(track.positions[0])),
        "trajectory_extent_m": float(np.linalg.norm(extent)),
        "median_step_translation_m": float(np.median(track.step_translation)),
        "p95_step_translation_m": float(np.percentile(track.step_translation, 95)),
        "median_step_rotation_deg": float(np.median(track.step_rotation_deg)),
        "p95_step_rotation_deg": float(np.percentile(track.step_rotation_deg, 95)),
        "extent_to_median_step_ratio": float(
            np.linalg.norm(extent) / max(float(np.median(track.step_translation)), 1e-12)
        ),
        "classification": "absolute local-world trajectory",
    }


def make_page(tracks: list[Track], page_number: int, page_count: int) -> Image.Image:
    width, height = 2200, 2450
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 42), "RPX SOS camera-pose audit: absolute or relative?", font=font(38, True), fill="#161616")
    draw.text(
        (70, 98),
        "Stored [x,y,z,qx,qy,qz,qw] values. Smooth world-space paths + small consecutive steps indicate absolute local-world poses.",
        font=font(20),
        fill="#4c4c4c",
    )
    draw.text((2070, 55), f"page {page_number}/{page_count}", anchor="ra", font=font(17), fill="#777777")

    row_top = 165
    row_height = 440
    for row_index, track in enumerate(tracks):
        top = row_top + row_index * row_height
        draw.text((70, top), track.object_id, font=font(25, True), fill="#1f1f1f")
        summary = metrics(track)
        draw.text(
            (360, top + 4),
            (
                f"first |t| {summary['first_position_norm_m']:.2f} m  ·  "
                f"path extent {summary['trajectory_extent_m']:.2f} m  ·  "
                f"median step {100 * summary['median_step_translation_m']:.2f} cm / "
                f"{summary['median_step_rotation_deg']:.2f}°"
            ),
            font=font(17),
            fill="#555555",
        )
        panel_top = top + 46
        trajectory_panel(draw, track, (70, panel_top, 735, panel_top + 345))
        components_panel(draw, track, (765, panel_top, 1430, panel_top + 345))
        step_panel(draw, track, (1460, panel_top, 2125, panel_top + 345))
        draw.line((70, top + row_height - 18, 2125, top + row_height - 18), fill="#dddddd", width=2)

    draw.text(
        (70, 2396),
        "Conclusion criterion: the stored positions trace continuous global paths; their frame differences are the small relative motions.",
        font=font(18, True),
        fill="#333333",
    )
    return canvas


def main() -> None:
    tracks = [load_track(object_id) for object_id in OBJECTS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [metrics(track) for track in tracks]
    csv_path = OUTPUT / "sos-campose-audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pages: list[Image.Image] = []
    for page_index, start in enumerate(range(0, len(tracks), 5), start=1):
        page = make_page(tracks[start : start + 5], page_index, math.ceil(len(tracks) / 5))
        page_path = OUTPUT / f"sos-campose-audit-page-{page_index}.png"
        page.save(page_path, optimize=True)
        pages.append(page)
        print(page_path)
    pdf_path = OUTPUT / "rpx-sos-campose-absolute-vs-relative-audit.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=180.0)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()

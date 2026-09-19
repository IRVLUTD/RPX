from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
WORKSPACE = Path("/media/naren/New Volume/rpx")
SNAPSHOT = (
    WORKSPACE
    / "docker-depth-smoke-cache/hub/datasets--IRVLUTD--RPX/snapshots"
    / REVISION
)
DEFAULT_ROOT = Path("/home/naren/Downloads/rpx-depthsplat-sos-10/sos")
ROOT = DEFAULT_ROOT
RESULT = ROOT / "result.json"
OUTPUT = ROOT / "visualizations"
REFERENCE_CACHE = ROOT / "reference_frames"
PDF = ROOT / "depthsplat-sos-all-predictions.pdf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=text_font,
        fill=fill,
    )


def pipeline(draw: ImageDraw.ImageDraw) -> None:
    y0, y1 = 130, 218
    boxes = [
        (55, y0, 340, y1, "REFERENCE VIEWS", "RGB + camera poses"),
        (385, y0, 625, y1, "TARGET QUERY", "camera pose only"),
        (685, y0, 905, y1, "DEPTHSPLAT", "synthesis model"),
        (955, y0, 1185, y1, "PREDICTION", "target RGB + depth"),
        (1245, y0, 1485, y1, "GROUND TRUTH", "comparison only"),
    ]
    for left, top, right, bottom, title, subtitle in boxes:
        draw.rounded_rectangle(
            (left, top, right, bottom), radius=10, fill="#f7f7f7", outline="#cccccc", width=2
        )
        centered_text(draw, (left, top + 10, right, top + 48), title, font(19, True), "#222222")
        centered_text(draw, (left, top + 44, right, bottom - 7), subtitle, font(16), "#666666")
    centered_text(draw, (340, y0, 385, y1), "+", font(25, True), "#777777")
    for start, end in ((625, 685), (905, 955)):
        draw.line((start + 7, 174, end - 10, 174), fill="#777777", width=3)
        draw.polygon([(end - 10, 174), (end - 19, 168), (end - 19, 180)], fill="#777777")
    centered_text(draw, (1185, y0, 1245, y1), "↔", font(25, True), "#777777")


def reference_frame(object_id: str, frame: int) -> Path:
    cached = REFERENCE_CACHE / object_id / f"{frame:05d}.webp"
    if cached.is_file():
        return cached

    archive_path = SNAPSHOT / "objects" / object_id / "0" / "rgb.tar"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with tarfile.open(archive_path) as archive:
        member = None
        for suffix in (".webp", ".png", ".jpg", ".jpeg"):
            try:
                member = archive.getmember(f"rgb/{frame:05d}{suffix}")
                break
            except KeyError:
                continue
        if member is None:
            raise FileNotFoundError(f"RGB frame {object_id}/{frame:05d} not found")
        stream = archive.extractfile(member)
        if stream is None:
            raise OSError(f"Could not read {member.name}")
        image = Image.open(io.BytesIO(stream.read())).convert("RGB")
        cached.parent.mkdir(parents=True, exist_ok=True)
        image.save(cached, format="WEBP", lossless=True)
    return cached


def fitted(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return ImageOps.fit(image, size, Image.Resampling.LANCZOS)


def panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    path: Path,
    title: str,
    subtitle: str,
) -> None:
    width, height = 350, 263
    draw.text((x, y), title, font=font(20, True), fill="#202020")
    draw.text((x, y + 31), subtitle, font=font(15), fill="#666666")
    image_y = y + 62
    canvas.paste(fitted(path, (width, height)), (x, image_y))
    draw.rectangle((x, image_y, x + width, image_y + height), outline="#b8b8b8", width=2)


def yes_no(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "not available"


def metric(row: dict, key: str, digits: int = 3) -> str:
    value = row.get(key)
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def make_page(row: dict, index: int, total: int) -> Image.Image:
    object_id = str(row["object_id"])
    context_frames = [int(value) for value in row["context_frames"]]
    target = int(row["target_frame"])
    protocol = str(row.get("protocol") or "far")
    protocol_label = protocol.replace("-", " ")
    if len(context_frames) != 2:
        raise ValueError(f"Expected K=2 for {object_id}, got {context_frames}")

    reference_paths = [reference_frame(object_id, frame) for frame in context_frames]
    sample_stem = f"ctx{context_frames[0]:05d}_{context_frames[1]:05d}__tgt{target:05d}.png"
    prediction = ROOT / "prediction_frames" / object_id / sample_stem
    ground_truth = ROOT / "target_frames" / object_id / sample_stem
    if not prediction.is_file():
        prediction = ROOT / "prediction_frames" / object_id / f"{target:05d}.png"
    if not ground_truth.is_file():
        ground_truth = ROOT / "target_frames" / object_id / f"{target:05d}.png"
    for path in (*reference_paths, prediction, ground_truth):
        if not path.is_file():
            raise FileNotFoundError(path)

    canvas = Image.new("RGB", (1540, 1040), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 36), "DepthSplat single-object prediction", font=font(31, True), fill="#171717")
    draw.text(
        (55, 82),
        f"{object_id}  ·  target {target:05d}  ·  {protocol_label}  ·  sample {index} of {total}",
        font=font(18),
        fill="#555555",
    )
    pipeline(draw)

    x_positions = (55, 430, 805, 1180)
    panel(
        canvas,
        draw,
        x_positions[0],
        280,
        reference_paths[0],
        "Reference 1",
        f"frame {context_frames[0]:05d} · model input",
    )
    panel(
        canvas,
        draw,
        x_positions[1],
        280,
        reference_paths[1],
        "Reference 2",
        f"frame {context_frames[1]:05d} · model input",
    )
    panel(
        canvas,
        draw,
        x_positions[2],
        280,
        prediction,
        "Prediction",
        f"frame {target:05d} · model output",
    )
    panel(
        canvas,
        draw,
        x_positions[3],
        280,
        ground_truth,
        "Ground truth",
        f"frame {target:05d} · scoring only",
    )

    draw.line((55, 690, 1485, 690), fill="#dddddd", width=2)
    draw.text((55, 724), "Sample metrics", font=font(19, True), fill="#222222")
    draw.text(
        (55, 758),
        (
            f"PSNR {metric(row, 'psnr')} dB   ·   SSIM {metric(row, 'ssim')}   ·   "
            f"Depth AbsRel {metric(row, 'depth_absrel')}   ·   "
            f"RMSE {metric(row, 'depth_rmse')} m   ·   δ1 {metric(row, 'depth_delta1')}"
        ),
        font=font(16),
        fill="#444444",
    )

    ar = row.get("ar_tag_check") or {}
    draw.text((55, 814), "AR-tag check", font=font(19, True), fill="#222222")
    ar_text = (
        f"Ground truth detected: {yes_no(ar.get('gt_board_detected'))}   ·   "
        f"Prediction detected: {yes_no(ar.get('prediction_board_detected'))}"
    )
    if ar.get("marker_id_recall") is not None:
        ar_text += f"   ·   Marker recall {float(ar['marker_id_recall']):.3f}"
    if ar.get("board_pose_translation_error_m") is not None:
        ar_text += f"   ·   Translation error {float(ar['board_pose_translation_error_m']):.3f} m"
    if ar.get("board_pose_rotation_error_deg") is not None:
        ar_text += f"   ·   Rotation error {float(ar['board_pose_rotation_error_deg']):.1f}°"
    draw.text((55, 848), ar_text, font=font(16), fill="#444444")

    draw.text((55, 904), "Protocol", font=font(19, True), fill="#222222")
    draw.text(
        (55, 938),
        (
            f"Frames {context_frames[0]:05d} and {context_frames[1]:05d} supply RGB and pose. "
            f"Only the pose of target {target:05d} is supplied; its real RGB/depth are withheld for scoring."
        ),
        font=font(16),
        fill="#444444",
    )
    draw.text(
        (1485, 1000),
        f"RPX SOS · K=2 · {protocol_label}",
        anchor="ra",
        font=font(15),
        fill="#777777",
    )
    return canvas


def main() -> None:
    global ROOT, RESULT, OUTPUT, REFERENCE_CACHE, PDF, SNAPSHOT

    parser = argparse.ArgumentParser(description="Create a PDF report from RPX SOS results")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT)
    args = parser.parse_args()

    ROOT = args.result_root
    SNAPSHOT = args.snapshot_root
    RESULT = ROOT / "result.json"
    REFERENCE_CACHE = ROOT / "reference_frames"

    data = json.loads(RESULT.read_text(encoding="utf-8"))
    requested = str((data.get("sampling_protocol") or {}).get("requested") or "far")
    OUTPUT = ROOT / f"visualizations-{requested}"
    PDF = ROOT / f"depthsplat-sos-{requested}-predictions.pdf"
    rows = data.get("samples") or []
    if not rows:
        raise SystemExit(f"No samples in {RESULT}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    for index, row in enumerate(rows, start=1):
        page = make_page(row, index, len(rows))
        contexts = [int(value) for value in row["context_frames"]]
        destination = (
            OUTPUT
            / (
                f"{index:03d}-{row['object_id']}--ctx-{contexts[0]:05d}-{contexts[1]:05d}"
                f"--target-{int(row['target_frame']):05d}.png"
            )
        )
        page.save(destination, optimize=True)
        pages.append(page)
        print(destination)

    pages[0].save(PDF, save_all=True, append_images=pages[1:], resolution=150.0)
    print(f"PDF: {PDF}")
    print(f"Pages: {len(pages)}")


if __name__ == "__main__":
    main()

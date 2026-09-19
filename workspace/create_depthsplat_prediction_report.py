from __future__ import annotations

import re
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REVISION = "2e2a387f7f93e98c177b2e039c141eacda94e5fc"
WORKSPACE = Path("/media/naren/New Volume/rpx")
SCENES = (
    WORKSPACE
    / "docker-depth-smoke-cache/hub/datasets--IRVLUTD--RPX/snapshots"
    / REVISION
    / "extracted/scenes"
)
ACCEPTANCE = Path("/home/naren/Downloads/rpx-depthsplat-acceptance/easy")
PREDICTIONS = ACCEPTANCE / "prediction_frames"
OUTPUT = ACCEPTANCE / "depthsplat-visualizations"
PDF = ACCEPTANCE / "depthsplat-all-predictions.pdf"
SOURCE_FRAMES = ACCEPTANCE / "source-frames"

NAME = re.compile(
    r"^(?P<scene>scene\d+)__(?P<phase>\d+)__ctx(?P<k>\d+)__"
    r"(?P<sample_type>[^_]+)__(?P<contexts>\d+(?:_\d+)*)__tgt(?P<target>\d+)$"
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def rgb_path(scene: str, phase: str, frame: str) -> Path:
    directory = SCENES / scene / phase / "rgb"
    for suffix in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = directory / f"{frame}{suffix}"
        if candidate.is_file():
            return candidate

    cached = SOURCE_FRAMES / scene / phase / f"{frame}.webp"
    if cached.is_file():
        return cached

    archive = SCENES.parent.parent / "scenes" / scene / phase / "rgb.tar"
    if not archive.is_file():
        raise FileNotFoundError(f"RGB archive not found: {archive}")
    member_name = f"rgb/{frame}.webp"
    with tarfile.open(archive) as source:
        member = source.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{member_name} not found in {archive}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(member.read())
    return cached


def image_box(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return ImageOps.fit(image, size, Image.Resampling.LANCZOS)


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


def pipeline(canvas: Image.Image, draw: ImageDraw.ImageDraw) -> None:
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
    canvas.paste(image_box(path, (width, height)), (x, image_y))
    draw.rectangle((x, image_y, x + width, image_y + height), outline="#b8b8b8", width=2)


def make_page(prediction: Path, index: int, total: int) -> Image.Image:
    match = NAME.match(prediction.stem)
    if match is None:
        raise ValueError(f"Unexpected prediction filename: {prediction.name}")
    parts = match.groupdict()
    contexts = parts["contexts"].split("_")
    if len(contexts) != 2:
        raise ValueError(f"Expected exactly two contexts in {prediction.name}")

    references = [rgb_path(parts["scene"], parts["phase"], frame) for frame in contexts]
    ground_truth = rgb_path(parts["scene"], parts["phase"], parts["target"])
    target_number = int(parts["target"])
    context_numbers = [int(frame) for frame in contexts]
    direction = "backward" if target_number < min(context_numbers) else "forward"

    canvas = Image.new("RGB", (1540, 960), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 36), "DepthSplat acceptance prediction", font=font(31, True), fill="#171717")
    draw.text(
        (55, 82),
        f"{parts['scene']}  ·  phase {parts['phase']}  ·  {direction} extrapolation  ·  sample {index} of {total}",
        font=font(18),
        fill="#555555",
    )
    pipeline(canvas, draw)

    x_positions = (55, 430, 805, 1180)
    panel(canvas, draw, x_positions[0], 280, references[0], "Reference 1", f"frame {contexts[0]} · model input")
    panel(canvas, draw, x_positions[1], 280, references[1], "Reference 2", f"frame {contexts[1]} · model input")
    panel(canvas, draw, x_positions[2], 280, prediction, "Prediction", f"frame {parts['target']} · model output")
    panel(canvas, draw, x_positions[3], 280, ground_truth, "Ground truth", f"frame {parts['target']} · scoring only")

    draw.line((55, 690, 1485, 690), fill="#dddddd", width=2)
    draw.text((55, 724), "What the model receives", font=font(19, True), fill="#222222")
    draw.text(
        (55, 758),
        f"Frames {contexts[0]} and {contexts[1]}: RGB and camera pose; frame {parts['target']}: camera pose only.",
        font=font(17),
        fill="#444444",
    )
    draw.text((55, 810), "What is withheld", font=font(19, True), fill="#222222")
    draw.text(
        (55, 844),
        f"The real RGB and depth at frame {parts['target']} are not model inputs. They are used only to score the prediction.",
        font=font(17),
        fill="#444444",
    )
    draw.text(
        (1485, 916),
        f"RPX · K=2 · {direction} extrapolation",
        anchor="ra",
        font=font(15),
        fill="#777777",
    )
    return canvas


def main() -> None:
    predictions = sorted(PREDICTIONS.rglob("*.png"))
    if not predictions:
        raise FileNotFoundError(f"No prediction PNGs found under {PREDICTIONS}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    for index, prediction in enumerate(predictions, start=1):
        page = make_page(prediction, index, len(predictions))
        destination = OUTPUT / f"{index:02d}-{prediction.stem}.png"
        page.save(destination, optimize=True)
        pages.append(page)
        print(destination)

    pages[0].save(PDF, save_all=True, append_images=pages[1:], resolution=150.0)
    print(f"PDF: {PDF}")
    print(f"Pages: {len(pages)}")


if __name__ == "__main__":
    main()

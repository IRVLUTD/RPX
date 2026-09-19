from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/media/naren/New Volume/rpx")
SNAPSHOT = (
    ROOT
    / "docker-depth-smoke-cache/hub/datasets--IRVLUTD--RPX/snapshots"
    / "2e2a387f7f93e98c177b2e039c141eacda94e5fc/extracted/scenes/scene004/0"
)
RESULT = Path("/home/naren/Downloads/rpx-depthsplat-acceptance/easy")
PREDICTION = (
    RESULT
    / "prediction_frames/scene004/0"
    / "scene004__0__ctx2__extrapolation__00000_00099__tgt00179.png"
)
OUTPUT = Path("/home/naren/Downloads/rpx-depthsplat-acceptance/depthsplat-input-output-diagram.png")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fitted(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#101827")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    image_path: Path,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    x, y = xy
    width, height = 340, 255
    draw.rounded_rectangle(
        (x - 8, y - 58, x + width + 8, y + height + 58),
        radius=18,
        fill="#172235",
        outline=color,
        width=4,
    )
    draw.text((x, y - 48), title, font=font(25, True), fill=color)
    draw.text((x, y - 18), subtitle, font=font(17), fill="#d5dfef")
    canvas.paste(fitted(image_path, (width, height)), (x, y))


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line((start, end), fill=color, width=6)
    ex, ey = end
    draw.polygon([(ex, ey), (ex - 18, ey - 11), (ex - 18, ey + 11)], fill=color)


def main() -> None:
    context_a = SNAPSHOT / "rgb/00000.webp"
    context_b = SNAPSHOT / "rgb/00099.webp"
    target_gt = SNAPSHOT / "rgb/00179.webp"
    for path in (context_a, context_b, target_gt, PREDICTION):
        if not path.is_file():
            raise FileNotFoundError(path)

    canvas = Image.new("RGB", (1680, 1080), "#0b1220")
    draw = ImageDraw.Draw(canvas)

    draw.text((60, 35), "DepthSplat on RPX: what goes in and what is evaluated", font=font(40, True), fill="white")
    draw.text(
        (60, 88),
        "Actual acceptance example · scene004 / phase 0 · K=2 · forward extrapolation",
        font=font(23),
        fill="#a9b8d0",
    )

    panel(canvas, draw, (60, 205), context_a, "REFERENCE / CONTEXT 1", "Frame 00000 · RGB + depth + camera pose", "#5eead4")
    panel(canvas, draw, (455, 205), context_b, "REFERENCE / CONTEXT 2", "Frame 00099 · RGB + depth + camera pose", "#5eead4")

    draw.rounded_rectangle((850, 180, 1120, 505), radius=24, fill="#3b2c16", outline="#fbbf24", width=4)
    draw.text((905, 225), "TARGET QUERY", font=font(27, True), fill="#fbbf24")
    draw.text((900, 285), "Frame 00179", font=font(26, True), fill="white")
    draw.text((885, 335), "Camera pose only", font=font(23), fill="white")
    draw.text((875, 375), "Target RGB/depth are", font=font(19), fill="#f8d98b")
    draw.text((888, 405), "hidden from model", font=font(19, True), fill="#f8d98b")

    draw.rounded_rectangle((1190, 180, 1615, 505), radius=24, fill="#182c47", outline="#60a5fa", width=4)
    draw.text((1280, 230), "DEPTHSPLAT", font=font(32, True), fill="#60a5fa")
    draw.text((1240, 300), "2 context RGB-D views", font=font(21), fill="white")
    draw.text((1240, 338), "+ their camera poses", font=font(21), fill="white")
    draw.text((1240, 376), "+ target camera pose", font=font(21), fill="white")
    draw.text((1270, 430), "renders novel view", font=font(22, True), fill="#bfdbfe")

    arrow(draw, (405, 335), (445, 335), "#5eead4")
    arrow(draw, (803, 335), (842, 335), "#fbbf24")
    arrow(draw, (1128, 335), (1182, 335), "#60a5fa")

    panel(canvas, draw, (180, 665), PREDICTION, "MODEL OUTPUT", "Predicted RGB at target pose · frame 00179", "#60a5fa")
    panel(canvas, draw, (1160, 665), target_gt, "HELD-OUT GROUND TRUTH", "Real D435 RGB · frame 00179 · evaluation only", "#f472b6")

    draw.rounded_rectangle((610, 650, 1070, 982), radius=24, fill="#241b34", outline="#c084fc", width=4)
    draw.text((680, 690), "COMPARE AFTER RENDERING", font=font(25, True), fill="#c084fc")
    draw.text((670, 750), "Predicted RGB ↔ real target RGB", font=font(20), fill="white")
    draw.text((670, 792), "Predicted depth ↔ D435 target depth", font=font(20), fill="white")
    draw.text((670, 850), "Metrics: PSNR, SSIM, depth", font=font(20), fill="#e9d5ff")
    draw.text((670, 886), "AbsRel, RMSE and δ<1.25", font=font(20), fill="#e9d5ff")
    draw.text((670, 935), "Target image is never an input.", font=font(20, True), fill="#f0abfc")
    arrow(draw, (528, 793), (602, 793), "#c084fc")
    arrow(draw, (1152, 793), (1078, 793), "#c084fc")

    timeline_y = 570
    draw.text((60, 535), "Temporal protocol", font=font(21, True), fill="white")
    draw.rounded_rectangle((250, timeline_y, 750, timeline_y + 28), radius=12, fill="#2dd4bf")
    draw.rounded_rectangle((750, timeline_y, 1000, timeline_y + 28), radius=12, fill="#64748b")
    draw.rounded_rectangle((1000, timeline_y, 1500, timeline_y + 28), radius=12, fill="#f59e0b")
    draw.text((360, timeline_y + 35), "context region 0–99", font=font(18), fill="#99f6e4")
    draw.text((795, timeline_y + 35), "20% guard", font=font(18), fill="#cbd5e1")
    draw.text((1120, timeline_y + 35), "target region 150–249", font=font(18), fill="#fde68a")
    marker_x = 1000 + int((179 - 150) / 99 * 500)
    draw.line((marker_x, timeline_y - 16, marker_x, timeline_y + 30), fill="white", width=4)
    draw.text((marker_x - 47, timeline_y - 47), "target 179", font=font(17, True), fill="white")

    draw.text(
        (60, 1030),
        "Reference = context/input view. Ground truth target = withheld answer used only for scoring.",
        font=font(20, True),
        fill="#dbeafe",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()

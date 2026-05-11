import os
from PIL import Image
from pathlib import Path
from absl import app, flags, logging

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Directory path to input video frames")

def main(argv):
    video_dir = sanity_check(argv)
    base_path = Path(video_dir)

    input_rgb_dir = base_path / "rgb"
    output_jpg_dir = base_path / "jpg"
    output_jpg_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directory: {output_jpg_dir}")

    # Get all PNG files and reverse the list
    png_files = sorted(input_rgb_dir.glob("*.png"))[::-1]

    for i, file in enumerate(png_files):
        with Image.open(file) as im:
            rgb_image = im.convert("RGB")
            new_filename = f"{i:06d}.jpg"
            rgb_image.save(output_jpg_dir / new_filename, "JPEG")
            print(f"✔️ {file.name} → {new_filename}")

    print("🎉 All PNG files converted to reversed JPGs in 'jpg/'")

def sanity_check(argv):
    input_dir = FLAGS.input_dir
    rgb_dir = os.path.join(input_dir, "rgb")

    if not os.path.isdir(rgb_dir):
        raise Exception(f"Error: The directory '{rgb_dir}' does not exist or is not valid.")

    image_files = [
        f for f in os.listdir(rgb_dir)
        if os.path.isfile(os.path.join(rgb_dir, f)) and f.lower().endswith(".png")
    ]

    if not image_files:
        raise Exception(f"No PNG image files found in the directory '{rgb_dir}'.")

    return input_dir

if __name__ == "__main__":
    flags.mark_flag_as_required("input_dir")
    app.run(main)

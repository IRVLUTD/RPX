#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + Grok 3)
#----------------------------------------------------------------------------------------------------
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from .config import logger

def vis_prop_masks(scene_dir):
    """
    Visualize RGB images and contour masks frame by frame interactively, with option to toggle faulty frames.
    Press X to mark/unmark a frame as faulty, shown with a red border. Faulty frame numbers are saved to sam2/iter1_faulty.txt.
    Displays a fault counter in the title.
    
    Args:
        scene_dir (str): Directory containing rgb/ and sam2/contour_gt_masks/ folders.
    """
    # Set TkAgg backend for interactive display
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        logger.warning("TkAgg backend not available, falling back to default.")

    # Define paths
    rgb_dir = Path(scene_dir) / "rgb"
    contour_dir = Path(scene_dir) / "sam2" / "contour_gt_masks"
    sam2_dir = Path(scene_dir) / "sam2"
    faulty_file = sam2_dir / "iter1_faulty.txt"

    # Check if directories exist
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not contour_dir.exists():
        raise FileNotFoundError(f"Contour masks directory not found: {contour_dir}")

    # Load and sort PNG files
    rgb_files = sorted(rgb_dir.glob("*.png"))
    contour_files = sorted(contour_dir.glob("*.png"))

    if not rgb_files:
        raise FileNotFoundError(f"No PNG files found in {rgb_dir}")
    if not contour_files:
        raise FileNotFoundError(f"No PNG files found in {contour_dir}")

    if len(rgb_files) != len(contour_files):
        logger.warning(f"Mismatch: {len(rgb_files)} RGB files, {len(contour_files)} contour files")

    # Set to store faulty contour mask paths
    faulty_frames = set()
    # Fault counter
    fault_count = [0]

    # Initialize figure
    fig, (ax_rgb, ax_contour) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1)

    # State for current frame
    current_frame = [0]

    def update_display():
        """Update the display with the current frame."""
        frame_idx = current_frame[0]
        if frame_idx < 0 or frame_idx >= len(rgb_files):
            return

        # Load images
        rgb_img = Image.open(rgb_files[frame_idx]).convert("RGB")
        contour_img = Image.open(contour_files[frame_idx]).convert("RGB")

        # Clear axes
        ax_rgb.clear()
        ax_contour.clear()

        # Display images
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f"RGB: {rgb_files[frame_idx].name}")
        ax_rgb.axis("off")

        ax_contour.imshow(contour_img)
        ax_contour.set_title(f"Contour Mask: {contour_files[frame_idx].name}")
        ax_contour.axis("off")

        # Add red border if faulty
        if str(contour_files[frame_idx]) in faulty_frames:
            # Get image dimensions
            img_width, img_height = rgb_img.size
            # Add border around RGB
            rgb_border = Rectangle((-0.5, -0.5), img_width, img_height, 
                                 edgecolor='red', facecolor='none', lw=3, linestyle='--')
            ax_rgb.add_patch(rgb_border)
            # Add border around contour
            contour_border = Rectangle((-0.5, -0.5), img_width, img_height, 
                                     edgecolor='red', facecolor='none', lw=3, linestyle='--')
            ax_contour.add_patch(contour_border)

        # Update figure title with fault counter
        status = "Faulty" if str(contour_files[frame_idx]) in faulty_frames else "Normal"
        fig.suptitle(f"Frame {frame_idx + 1}/{len(rgb_files)} ({status}) | Faulty Frames: {fault_count[0]}")
        plt.draw()

    def on_key(event):
        """Handle keyboard events."""
        frame_idx = current_frame[0]
        if event.key == "right" and frame_idx < len(rgb_files) - 1:
            current_frame[0] += 1
            update_display()
        elif event.key == "left" and frame_idx > 0:
            current_frame[0] -= 1
            update_display()
        elif event.key == "x" and 0 <= frame_idx < len(rgb_files):
            # Toggle faulty status
            contour_path = str(contour_files[frame_idx])
            if contour_path in faulty_frames:
                faulty_frames.remove(contour_path)
                fault_count[0] -= 1
                logger.info(f"Unmarked frame {contour_files[frame_idx].name} as faulty")
            else:
                faulty_frames.add(contour_path)
                fault_count[0] += 1
                logger.info(f"Marked frame {contour_files[frame_idx].name} as faulty")
            update_display()
        elif event.key == "q":
            plt.close()

    def on_close(event):
        """Save faulty frame numbers to text file when window is closed."""
        if faulty_frames:
            try:
                with open(faulty_file, "w") as f:
                    for contour_path in sorted(faulty_frames):
                        frame_number = Path(contour_path).stem
                        f.write(f"{frame_number}\n")
                logger.info(f"Saved {len(faulty_frames)} faulty frames to {faulty_file}")
            except Exception as e:
                logger.error(f"Failed to save faulty frames to {faulty_file}: {e}")
        else:
            logger.info("No faulty frames marked.")

    # Connect events
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    # Initial display
    update_display()

    # Show plot
    logger.info("Starting interactive visualization. Use Right/Left Arrow to navigate, X to toggle faulty, Q to quit.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RGB and contour masks frame by frame, with faulty frame toggling and counter.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Scene directory containing rgb/ and sam2/")
    args = parser.parse_args()
    vis_prop_masks(args.scene_dir)
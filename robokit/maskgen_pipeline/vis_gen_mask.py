#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check if you have the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025). (with copilot + Grok 3)
#----------------------------------------------------------------------------------------------------
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from .config import logger


def vis_refined_masks(args):
    """
    Show all frames listed in iter{iter-1}_faulty.txt (previous iteration),
    but pre-mark only those listed in iter{iter}_faulty.txt (current iteration).
    On close, save the currently marked set to iter{iter}_faulty.txt.
    """
    scene_dir = Path(args.scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    rgb_dir   = scene_dir / "rgb"
    sam2_dir  = scene_dir / "sam2"

    # ---- files: prev list for display; curr list for pre-mark + save ----
    prev_faulty_file = sam2_dir / f"iter{args.iter-1}_faulty.txt"
    curr_faulty_file = sam2_dir / f"iter{args.iter}_faulty.txt"

    # refined contours live in previous iteration's refinement output
    refined_sam2_dir = sam2_dir / "mask_refinement" / f"iter_{args.iter-1}" / "sam2"
    contour_dir      = refined_sam2_dir / "contour_gt_masks"

    # sanity
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not contour_dir.exists():
        raise FileNotFoundError(f"Contour masks directory not found: {contour_dir}")
    if not prev_faulty_file.exists():
        raise FileNotFoundError(f"Previous faulty list not found: {prev_faulty_file}")

    # ---- build the display list from PREVIOUS faulty frames ----
    rgb_files = []
    contour_files = []
    with open(prev_faulty_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            rgb_files.append(rgb_dir / f"{s}.png")
            contour_files.append(contour_dir / f"{s}.png")

    if len(rgb_files) != len(contour_files):
        logger.warning(f"Mismatch: {len(rgb_files)} RGB files vs {len(contour_files)} contour files")

    # ---- load CURRENT faulty marks (if exists) to pre-select which of the displayed frames start as 'faulty' ----
    premarked_set = set()
    curr_exists = curr_faulty_file.exists()
    if curr_exists:
        try:
            with open(curr_faulty_file, "r") as f:
                premarked_set = {line.strip() for line in f if line.strip()}
            logger.info(f"Loaded {len(premarked_set)} current-faulty markers from {curr_faulty_file}")
        except Exception as e:
            logger.error(f"Failed to load current faulty list {curr_faulty_file}: {e}")

    # ---- interactive state ----
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image as PILImg

    faulty_frames = set()   # set of full contour file paths currently marked in the UI
    fault_count = [0]
    # Initialize marks: only mark those present in the displayed list whose stem is in the current-faulty set
    for cf in contour_files:
        if cf.stem in premarked_set:
            faulty_frames.add(str(cf))
            fault_count[0] += 1

    fig, (ax_rgb, ax_contour) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1)
    current_frame = [0]

    def update_display():
        i = current_frame[0]
        if i < 0 or i >= len(rgb_files): return

        rgb_img = PILImg.open(rgb_files[i]).convert("RGB")
        contour_img = PILImg.open(contour_files[i]).convert("RGB")

        ax_rgb.clear(); ax_contour.clear()
        ax_rgb.imshow(rgb_img);       ax_rgb.set_title(f"RGB: {rgb_files[i].name}");           ax_rgb.axis("off")
        ax_contour.imshow(contour_img); ax_contour.set_title(f"Contour: {contour_files[i].name}"); ax_contour.axis("off")

        # red border if currently marked
        if str(contour_files[i]) in faulty_frames:
            w, h = rgb_img.size
            ax_rgb.add_patch(Rectangle((-0.5,-0.5), w, h, edgecolor='red', facecolor='none', lw=8, linestyle='--'))
            ax_contour.add_patch(Rectangle((-0.5,-0.5), w, h, edgecolor='red', facecolor='none', lw=8, linestyle='--'))

        status = "Faulty" if str(contour_files[i]) in faulty_frames else "Normal"
        prev_status = f"Prev list: {prev_faulty_file.name}"
        curr_status = f"Curr preload: {'present' if curr_exists else 'absent'}"
        fig.suptitle(f"Frame {i+1}/{len(rgb_files)} ({status}) | Marked: {fault_count[0]} | {prev_status} | {curr_status}")
        plt.draw()

    def on_key(event):
        i = current_frame[0]
        if event.key == "right" and i < len(rgb_files) - 1:
            current_frame[0] += 1; update_display()
        elif event.key == "left" and i > 0:
            current_frame[0] -= 1; update_display()
        elif event.key == "x" and 0 <= i < len(rgb_files):
            path = str(contour_files[i])
            if path in faulty_frames:
                faulty_frames.remove(path); fault_count[0] -= 1
                logger.info(f"Unmarked {contour_files[i].name}")
            else:
                faulty_frames.add(path); fault_count[0] += 1
                logger.info(f"Marked {contour_files[i].name}")
            update_display()
        elif event.key == "q":
            plt.close()

    def on_close(event):
        # Save CURRENT marks to iter{iter}_faulty.txt
        try:
            curr_faulty_file.parent.mkdir(parents=True, exist_ok=True)
            with open(curr_faulty_file, "w") as f:
                frame_numbers = sorted({Path(p).stem for p in faulty_frames})
                for num in frame_numbers: f.write(f"{num}\n")
            logger.info(f"Saved {len(frame_numbers)} faulty frames to {curr_faulty_file}")
        except Exception as e:
            logger.error(f"Failed to save faulty frames to {curr_faulty_file}: {e}")

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    logger.info(
        "Showing ALL frames from previous faulty list. "
        "Frames pre-marked only if present in current faulty list. "
        "Arrows=nav, X=toggle, Q=quit."
    )
    update_display()
    plt.show()

def vis_prop_masks(scene_dir):
    """
    Visualize RGB images and contour masks frame by frame interactively, with option to toggle faulty frames.
    Press X to mark/unmark a frame as faulty, shown with a red border. Faulty frame numbers are saved to sam2/iter1_faulty.txt.
    Displays a fault counter in the title. Loads existing faulty frames from iter1_faulty.txt if present.
    Title indicates if faulty file is present (green if present, red if not).
    
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
    # Check if faulty file exists
    faulty_file_exists = faulty_file.exists()

    # Load existing faulty frames if iter1_faulty.txt exists
    if faulty_file_exists:
        try:
            with open(faulty_file, "r") as f:
                faulty_frame_numbers = {line.strip() for line in f if line.strip()}
            # Map frame numbers to full contour file paths
            for contour_file in contour_files:
                if contour_file.stem in faulty_frame_numbers:
                    faulty_frames.add(str(contour_file))
                    fault_count[0] += 1
            logger.info(f"Loaded {len(faulty_frames)} faulty frames from {faulty_file}")
        except Exception as e:
            logger.error(f"Failed to load faulty frames from {faulty_file}: {e}")

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
                                    edgecolor='red', facecolor='none', lw=8, linestyle='--')
            ax_rgb.add_patch(rgb_border)
            # Add border around contour
            contour_border = Rectangle((-0.5, -0.5), img_width, img_height, 
                                        edgecolor='red', facecolor='none', lw=8, linestyle='--')
            ax_contour.add_patch(contour_border)

        # Update figure title with fault counter and faulty file status
        status = "Faulty" if str(contour_files[frame_idx]) in faulty_frames else "Normal"
        faulty_status = "Faulty File Present" if faulty_file_exists else "No Faulty File"
        # Note: faulty_status should appear green if faulty_file_exists, red otherwise
        fig.suptitle(f"Frame {frame_idx + 1}/{len(rgb_files)} ({status}) | Faulty Frames: {fault_count[0]} | {faulty_status}")
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
        """Save faulty frame numbers to text file when window is closed, ensuring no duplicates."""
        if faulty_frames:
            try:
                # Ensure parent directory exists
                faulty_file.parent.mkdir(parents=True, exist_ok=True)
                with open(faulty_file, "w") as f:
                    # Write unique frame numbers in sorted order
                    frame_numbers = sorted({Path(contour_path).stem for contour_path in faulty_frames})
                    for frame_number in frame_numbers:
                        f.write(f"{frame_number}\n")
                logger.info(f"Saved {len(frame_numbers)} faulty frames to {faulty_file}")
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
    parser.add_argument(
        "--iter", type=int, default=1, help="Iteration for refinement after initial run"
    )
    args = parser.parse_args()
    
    if args.iter > 1:
        vis_refined_masks(args)
    else:
        vis_prop_masks(args.scene_dir)
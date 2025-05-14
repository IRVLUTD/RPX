#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


import os
import torch
import requests
import numpy as np
from tqdm import tqdm
import torchvision.ops as ops
from PIL import Image as PILImg
from absl import app, flags, logging
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None, 'Directory path to input video frames')
flags.DEFINE_string('text_prompt', None, 'Text prompt for initial object detection')
flags.DEFINE_integer('n', 1, 'N for removeing N largest bboxes from GDINO preds')


def load_palette_from_url(url="https://raw.githubusercontent.com/IRVLUTD/fewsol-toolkit/refs/heads/main/palette.txt"):
    """
    Load the color palette from a remote URL.

    Args:
        url (str): URL to the palette.txt file.

    Returns:
        list: List of (R, G, B) tuples.
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    palette = [tuple(map(int, line.strip().split())) for line in response.text.strip().split("\n")]
    return palette

def apply_palette(mask, palette):
    """
    Apply a color palette to a segmentation mask.

    Args:
        mask (torch.Tensor): Tensor of shape [H, W] containing integer labels.
        palette (list): List of (R, G, B) tuples.

    Returns:
        np.ndarray: RGB image of shape [H, W, 3].
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = torch.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background (assuming background is 0)
        # Ensure label is an integer before indexing the palette
        label_int = int(label.item())  # Convert to int explicitly
        color = palette[label_int % len(palette)]  # Cycle through palette if needed
        colored_mask[mask == label_int] = color

    return PILImg.fromarray(colored_mask)


def combine_masks(gt_masks):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [1,2,0].

    Args:
        gt_masks (torch.Tensor): Tensor of shape [N, H, W] representing multiple bit masks.

    Returns:
        torch.Tensor: Combined mask of shape [H, W].
    """
    try:
        gt_masks = torch.flip(gt_masks, dims=(0,))
        num, h, w = gt_masks.shape
        bin_mask = torch.zeros((h, w), device=gt_masks.device)
        num_instance = len(gt_masks)

        # if there is not any instance, just return a mask full of 0s.
        if num_instance == 0:
            return bin_mask

        for m, object_label in zip(gt_masks, range(1, 1 + num_instance)):
            label_pos = torch.nonzero(m, as_tuple=True)
            bin_mask[label_pos] = object_label
        return bin_mask

    except Exception as e:
        logging.error(f"Error combining masks: {e}")
        raise e


def remove_n_largest_bboxes(initial_bboxes, phrases, gdino_conf, n):
    """
    Remove the 'n' bounding boxes with the largest areas and their corresponding phrases and confidence scores.

    Args:
        initial_bboxes (torch.Tensor): Tensor of shape [N, 4] representing bounding boxes (x, y, w, h).
        phrases (list): List of phrases corresponding to each bounding box.
        gdino_conf (torch.Tensor): Tensor of confidence scores for each bounding box.
        n (int): Number of largest bounding boxes to remove.

    Returns:
        Tuple: Updated initial_bboxes, phrases, and gdino_conf with the 'n' largest bboxes removed.
    """
    if n <= 0:
        return initial_bboxes, phrases, gdino_conf

    # Compute the area of each bounding box (w * h)
    bbox_areas = initial_bboxes[:, 2] * initial_bboxes[:, 3]  # width * height

    # Get indices of the 'n' largest bounding boxes
    largest_indices = torch.argsort(bbox_areas, descending=True)[:n]

    # Create a mask to keep only the non-largest bounding boxes
    keep_indices = torch.ones(len(initial_bboxes), dtype=torch.bool)
    keep_indices[largest_indices] = False  # Set the largest ones to False

    # Apply the mask to filter out the largest bounding boxes
    initial_bboxes = initial_bboxes[keep_indices]
    gdino_conf = gdino_conf[keep_indices]

    # Remove corresponding phrases
    phrases = [phrases[i] for i in range(len(phrases)) if keep_indices[i]]

    return initial_bboxes, phrases, gdino_conf


def merge_rgb_with_mask(rgb_pil, masks_pil):
    """
    Merges the RGB image with the mask while applying a semi-transparent black overlay to the background.

    Args:
        rgb_pil (PIL.Image): The RGB image.
        masks_pil (PIL.Image): The mask image with a palette.

    Returns:
        PIL.Image: Merged image with highlighted masks and darkened background.
    """
    # Convert images to RGBA
    rgb_image = rgb_pil.convert("RGB")
    mask_image = masks_pil.convert("RGBA")

    # Convert images to NumPy arrays
    rgb_array = np.array(rgb_image, dtype=np.float32)
    mask_array = np.array(mask_image, dtype=np.float32)

    # Detect non-black regions in the mask
    mask_binary = (mask_array[..., :3] != [0, 0, 0]).any(axis=-1)

    # Darken the background (apply transparency)
    bg_alpha = 0.6  # Transparency level for background
    rgb_darkened = rgb_array * (1 - bg_alpha)

    # Blend the mask with transparency
    mask_alpha = 0.5  # Transparency for the mask colors
    blended_mask = rgb_darkened.copy()
    blended_mask[mask_binary] = (
        mask_array[mask_binary][:, :3] * mask_alpha + rgb_darkened[mask_binary] * (1 - mask_alpha)
    )

    # Convert back to PIL Image
    blended_image = PILImg.fromarray(np.uint8(blended_mask))
    
    return blended_image


def apply_nms(bboxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Args:
        bboxes (torch.Tensor): Tensor of shape [N, 4] representing bounding boxes in (x1, y1, x2, y2) format.
        scores (torch.Tensor): Confidence scores for each bounding box.
        iou_threshold (float): Overlap threshold for suppressing boxes (default=0.5).

    Returns:
        torch.Tensor: Filtered bounding boxes after NMS.
    """
    # Perform NMS using PyTorch's built-in function
    keep_indices = ops.nms(bboxes, scores, iou_threshold)

    # Return only the bounding boxes that survived NMS
    return bboxes[keep_indices], keep_indices

def main(argv):
    # Get input values from flags
    video_dir, text_prompt, n_rm_large_bboxes = sanity_check(argv)

    # Check if required flags are provided
    if not video_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    logging.info("Initialize object detectors")

    # Initialize Grounding DINO for initial bbox detection
    gdino = GroundingDINOObjectPredictor()

    # Initialize SAM2 for tracking across frames
    sam2 = SAM2Predictor()

    mask_out_dir = os.path.join(video_dir, '../gsam2/masks/')
    overlay_out_dir = os.path.join(video_dir, '../gsam2/gdino_overlay/')
    palette_out_dir = os.path.join(video_dir, '../gsam2/palette/')
    rgbm_out_dir = os.path.join(video_dir, '../gsam2/rgb_and_mask/')
    
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(overlay_out_dir, exist_ok=True)
    os.makedirs(palette_out_dir, exist_ok=True)
    os.makedirs(rgbm_out_dir, exist_ok=True)

    # Load the palette from the URL
    palette = load_palette_from_url()

    img_name = sorted(os.listdir(video_dir))[0]
    img_path = os.path.join(video_dir, img_name)
    
    # Read the first frame to detect initial bounding boxes
    img_pil = PILImg.open(img_path).convert("RGB")

    logging.info("GDINO: Predict initial bounding boxes, phrases, and confidence scores")
    initial_bboxes, phrases, gdino_conf = gdino.predict(img_pil, text_prompt)

    initial_bboxes, kept_indices = apply_nms(initial_bboxes, gdino_conf, iou_threshold=0.5)
    phrases =  [phrases[i] for i in kept_indices]
    gdino_conf = gdino_conf[kept_indices]


    print(f"{len(initial_bboxes)} Initial BBoxes detected!")

    ######## Keep only the bboxes containing object keyword to avoid unwanted objects like chair, table etc. 
    # Get the indices of the "objects" phrases
    indices_to_keep = [i for i, phrase in enumerate(phrases) if 'objects' in phrase]

    # Filter the phrases to keep only those containing "objects"
    phrases =  [phrases[i] for i in indices_to_keep]

    # Filter initial_bboxes and gdino_conf based on the indices_to_keep
    initial_bboxes = initial_bboxes[indices_to_keep]
    gdino_conf = gdino_conf[indices_to_keep]

    print(f"{len(initial_bboxes)} Initial BBoxes filtered!")
    ######################################################################################################## 

    # rm the largest bbox as most of the times it covers all the objects and is not needed
    initial_bboxes, phrases, gdino_conf = \
                    remove_n_largest_bboxes(initial_bboxes, phrases, gdino_conf, n_rm_large_bboxes)

    # Scale bounding boxes to match the original image 
    w, h = img_pil.size # Get image width and height 
    image_pil_bboxes = gdino.bbox_to_scaled_xyxy(initial_bboxes, w, h)

    # Apply overlay and annotations only on first frame
    bbox_annotated_pil = annotate(img_pil, image_pil_bboxes, gdino_conf, phrases)  # Annotate
    bbox_annotated_pil.save(os.path.join(overlay_out_dir, img_name.replace('jpg', 'png')))

    bbox_annotated_pil.show()

    video_segments = sam2.propagate_bbox_prompt_masks_and_save(video_dir, image_pil_bboxes)
            
    for idx, (filename, masks) in tqdm(enumerate(video_segments.items())):
        
        # Remove the extra dimension (from shape (N, 1, H, W) -> (N, H, W))
        masks_squeezed = torch.tensor(masks).squeeze(1)

        # Generate the combined mask after removing the largest object
        combined_mask = combine_masks(masks_squeezed)

        img_pil = PILImg.open(os.path.join(video_dir, filename)).convert("RGB")

        img_name = filename.replace('jpg', 'png')

        # if idx == 0:
        #     # Apply overlay and annotations only on first frame
        #     bbox_annotated_pil = annotate( overlay_masks(img_pil, masks_squeezed), image_pil_bboxes, gdino_conf, phrases)  # Annotate
        #     bbox_annotated_pil.save(os.path.join(overlay_out_dir, img_name))
        

        # Convert to numpy
        combined_mask_np = np.array(combined_mask.cpu(), dtype=np.uint8)  # Ensure it's uint8

        # Convert to PIL Image in 'L' mode (grayscale)
        to_save = PILImg.fromarray(combined_mask_np, mode="L")
        to_save.save(os.path.join(mask_out_dir, img_name))

        masks_squeezed = np.array(masks_squeezed.cpu(), dtype=np.uint16)
        np.savez_compressed(os.path.join(mask_out_dir, img_name.replace('png', 'npz')), masks=masks_squeezed)

        # Apply the palette
        colored_mask = apply_palette(combined_mask, palette)
        colored_mask.save(os.path.join(palette_out_dir, img_name))

        combined_image = merge_rgb_with_mask(img_pil, colored_mask)
        combined_image.save(os.path.join(rgbm_out_dir, img_name))


def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    text_prompt = flags.FLAGS.text_prompt
    n_rm_largest_bboxes = flags.FLAGS.n
    
    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        raise Exception(f"Error: The directory '{input_dir}' does not exist or is not a valid directory.")
        

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise Exception(f"No image files found in the directory '{input_dir}'.")
    
    # Ensure text_prompt is provided
    if not text_prompt:
        raise Exception("Error: 'text_prompt' is required but not provided.")

    return input_dir, text_prompt, n_rm_largest_bboxes


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('text_prompt')
    
    # Run the main function
    app.run(main)

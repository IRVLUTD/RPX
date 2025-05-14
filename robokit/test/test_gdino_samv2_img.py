#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

"""
Works with png images as well; no need to convert to jpg as required for mask propogation in video (test_gdino_samv2.py).
python test_gdino_samv2_img.py --input_dir=<path_to_frames> --text_prompt="object description"
"""


import os
from PIL import Image as PILImg
from absl import app, flags, logging
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None, 'Directory path to input video frames')
flags.DEFINE_string('text_prompt', None, 'Text prompt for initial object detection')


def main(argv):
    # Get input values from flags
    frames_dir, text_prompt = sanity_check(argv)

    # Check if required flags are provided
    if not frames_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    logging.info("Initialize object detectors")

    # Initialize Grounding DINO for initial bbox detection
    gdino = GroundingDINOObjectPredictor()

    # Initialize SAM2 for tracking across frames
    sam2 = SAM2Predictor(text_prompt)

    for img_path in sorted(os.listdir(frames_dir)):
        img_path = os.path.join(frames_dir, img_path)
        try:
            # Read the first frame to detect initial bounding boxes
            image_pil = PILImg.open(img_path).convert("RGB")

            logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

            logging.info("GDINO post processing")
            w, h = image_pil.size # Get image width and height 
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            logging.info("SAM prediction")
            masks, scores, logits = sam2.predict_mask_in_image(image_pil, image_pil_bboxes)

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

            bbox_annotated_pil.show()

            print(masks.shape)

        except Exception as e:
            # Handle unexpected errors
            print(f"An unexpected error occurred: {e}")
        
        # break


def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    text_prompt = flags.FLAGS.text_prompt
    
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

    return input_dir, text_prompt


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('text_prompt')
    
    # Run the main function
    app.run(main)

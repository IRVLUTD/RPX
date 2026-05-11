#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------


import os
import numpy as np
from absl import app, flags, logging
from PIL import Image as PILImg
from robokit.perception import SAM2Predictor

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', "imgs/sam2-test/rgb", 'Directory path to input video frames')
flags.DEFINE_string('text_prompt', "objects", 'Text prompt for initial object detection')
flags.DEFINE_integer('save_interval', 1, 'Interval for saving tracked frames')

def main(argv):
    # Get input values from flags
    video_dir, text_prompt = sanity_check(argv)

    point_prompts_one_per_obj = [
        # x, y, pos or neg
        [(271, 195, 1)],
        [(331, 208, 1)],
        [(392, 207, 1)],
        [(453, 211, 1)],
        [(462, 300, 1)],
        [(402, 307, 1)],
        [(373, 330, 1)],
        [(296, 318, 1)],
        [(377, 430, 1)],
        [(436, 431, 1)],
        [(578, 463, 0)],
        [(558, 437, 1)],
        [(513, 288, 1)],
        [(162, 417, 1)],
        [(97, 437, 1)],
        [(40, 470, 1)]
    ]

    point_prompts_n_per_obj = [
        [(375, 194, 1), (379, 195, 1), (388, 195, 1), (396, 195, 1), (403, 207, 1), (397, 206, 1), (396, 204, 1), (387, 208, 1), (376, 195, 1), (376, 206, 1), (379, 212, 1), (389, 210, 1), (391, 197, 1)],
        [(419, 214, 1), (430, 214, 1), (438, 213, 1), (452, 213, 1), (463, 213, 1), (450, 206, 1), (430, 209, 1)],
        [(324, 210, 1), (324, 202, 1), (330, 201, 1), (335, 201, 1), (340, 205, 1), (333, 210, 1)],
        [(293, 192, 1), (294, 196, 1), (294, 203, 1), (291, 210, 1)],
        [(280, 188, 1), (278, 195, 1), (276, 211, 1), (266, 208, 1), (267, 197, 1), (267, 188, 1), (272, 183, 1), (271, 200, 1), (271, 203, 1)],
        [(148, 414, 1), (165, 422, 1), (190, 408, 1), (198, 419, 1), (193, 430, 1), (168, 415, 1), (144, 423, 1), (161, 407, 1)],
        [(84, 437, 1), (101, 441, 1), (114, 434, 1), (100, 429, 1), (96, 437, 1), (102, 430, 1)],
        [(31, 479, 1), (34, 476, 1), (41, 470, 1), (47, 463, 1), (48, 462, 1), (65, 460, 1), (29, 458, 1), (27, 465, 1), (16, 468, 1)],
        [(290, 277, 1), (277, 300, 1), (282, 316, 1), (277, 333, 1), (300, 328, 1), (300, 316, 1), (297, 308, 1), (291, 292, 1)],
        [(340, 338, 1), (348, 333, 1), (356, 331, 1), (384, 336, 1), (331, 325, 1), (393, 329, 1)],
        [(393, 267, 1), (400, 267, 1), (401, 273, 1), (405, 289, 1), (406, 312, 1), (404, 320, 1), (395, 303, 1), (407, 303, 1), (409, 324, 1)],
        [(430, 266, 1), (452, 265, 1), (467, 268, 1), (459, 303, 1), (472, 330, 1), (457, 334, 1), (448, 335, 1), (435, 325, 1), (437, 307, 1), (461, 302, 1), (443, 263, 1), (449, 303, 1), (457, 299, 1)],
        [(504, 297, 1), (505, 288, 1), (522, 293, 1), (510, 298, 1), (530, 298, 1), (525, 301, 1), (510, 286, 1)],
        [(379, 411, 1), (385, 413, 1), (385, 416, 1), (385, 421, 1), (385, 434, 1), (388, 442, 1), (381, 442, 1), (382, 430, 1), (382, 427, 1), (368, 429, 1), (278, 453, 1), (292, 451, 1), (308, 443, 1), (337, 437, 1)],
        [(424, 408, 1), (447, 409, 1), (440, 429, 1), (420, 443, 1), (444, 444, 1), (448, 443, 1), (436, 441, 1), (436, 400, 1)],
        [(557, 456, 1), (583, 451, 1), (597, 462, 1), (566, 472, 1), (570, 459, 1)],
        [(564, 435, 0), (584, 435, 0), (555, 440, 0), (558, 427, 0)]
    ]


    clicked_points = point_prompts_n_per_obj


    # Check if required flags are provided
    if not video_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    try:
        logging.info("Initialize object detectors")

        # Initialize SAM2 for tracking across frames
        sam2 = SAM2Predictor(text_prompt)

        # Read the first frame to detect initial bounding boxes
        first_frame_path = os.path.join(video_dir, sorted(os.listdir(video_dir))[0])
        first_frame = PILImg.open(first_frame_path).convert("RGB")

        if clicked_points is not None:
            
            logging.info(f"SAM2: Track object point prompts across all frames")
            # Track and propagate the bounding box across frames with the specified save interval
            video_segments = sam2.propagate_point_prompt_masks_and_save(video_dir, clicked_points)
            
            logging.info("Tracking complete for all bounding boxes")
        else:
            logging.error("No bounding boxes detected in the first frame")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


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
    
    return input_dir, text_prompt


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('text_prompt')
    
    # Run the main function
    app.run(main)

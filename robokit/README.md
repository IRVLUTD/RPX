# Maskgen Iteration 3 (Final) Instructions

<!-- This is what worked for me -->
## Setup
```shell
export CUDA_HOME=/usr/local/cuda-12.6/
conda create -n rkit-rpx python=3.9
conda activate rkit-rpx
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
python setup.py install
```

## Overview
- Download one partition (unique) from [maskgen-iter-3-out](https://utdallas.app.box.com/folder/366072704511)
- unzip all scenes (```unzip '*.zip'```)
- rm the zip file
- annotate & verify the faulty frames
- run the offline SAM2 script
- check correctness of masks (Needs to be implemented by me)
- upload to box

## 📌 Running the pipeline for mask annotation
```shell
# NOTE: All data downloaded from box in partitions has been preprocessed into the necessary file structure, as well as had bboxes
# generated for every faulty frame by groundingDino. All that is left is to verify the bboxes, and run SAM2 on them.

# Step 1: Download the data from [here](https://utdallas.app.box.com/folder/366072704511)

# Step 2: In the base dir (/partition_#), run the command ``` unzip '*.zip' ``` in your terminal. This will unzip all the necessary scene folders.

# Step 3: Running the bbox verification script. In the robokit dir, do the following:

# Instructions:
# NOTE: This step will require you to be very attentive to the bboxes generated and to the objects, in some instances visibility will be
# hindered by the bboxes, so it may be necessary to go to the scene folder and look at the raw frame, this will help identify object boundaries.
# It is also important to note, that when adding bboxes, it is often better to make the bbox too large than too small, with that being said, please
# be wary of stray shadows, as those may be segmented as well later on.

# Run the below script in the robokit directory

python -m maskgen_pipeline.review_faulty_bboxes --base_dir /<insert path>/partition_#

# Explanation of UI:
# You are presented with an overlaid image of the raw faulty frame & the bboxes.
# Each bbox is numbered and has a color assigned to it, these are arbitrarily done and are only for clarity of use.
# The user is able to remove bboxes by pressing in the X mark, or in general the box that encompasses their name. An additional way to remove is to double right click on the bbox itself,
# this is often finnicky so please beware.
# To confirm that the bboxes on the screen are correct, press Q. This will save the bboxes in another folder named "bboxes_verified". This will also move on to the next frame automatically.
# To traverse the set without saving you can use the a/d keys or the arrow keys.
# To exit the annotating script, press ESC. Please be aware that this will save the current frame, so please annotate the frame you are on before saving/
# Once you come back to the annotation script, you will be faced with the first unverified frame, and you can continue annotating.

# Once a good portion of frames are done for the session, or you are done for the time being, you can move on to step 4. Please note that step 3 must be repeated until all frames are verified.

# Step 4: Running the SAM2 offline script. In the robokit dir, do the following:

# Run the below script in the robokit directory

python -m maskgen_pipeline.run_sam2_from_verified_bboxes --base_dir /<insert path>/partition_#

# This script will run SAM2 on each verified frame. This is fully autonomous so all that is needed is to run the script.

# Step 5: Verify the masks. This step is not yet implemented, and will be implemented hopefully by 02/14/26 night.



## Quick demo of the process for a few frames:

# This will hopefully be completed by 02/14/26. For now, please feel free to message me with any concerns and questions.

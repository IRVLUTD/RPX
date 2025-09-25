# Maskgen Iteration 2 Instructions

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
- Download one file from [maskgen-iter-1-out](https://utdallas.app.box.com/folder/321198327745?s=59ois7sifqlaoojr0l2isdmk7pkjabga)
- unzip it
- rm the zip file
- do the label refining for 0/1/2
- do the object -> mask mapping on the base dir
- zip the results and upload to box in this folder [maskgen-iter-2-out](https://utdallas.app.box.com/folder/342100579130)

## 📌 Running SAM2 for Ground Truth Mask Generation
```shell
# Step 1: Download the data from [here](https://utdallas.app.box.com/folder/321198327745?s=59ois7sifqlaoojr0l2isdmk7pkjabga)

# Step 2: For each scene directory (e.g., <scene_dir>/0, <scene_dir>/1, <scene_dir>/2), run the following:

# NOTE: This may not give an error if the folder doesn't have an existing iter1.text file, this is because
# the portion of the scene simply has no faulty frames, in this case please disregard the folder and move on to the next one
# (i.e. from 0->1->2).
# Make sure to complete step 3 regardless of the ability to perform step 2.

# Instructions:
# This portion is fully automatic, so there are no instructions for this one.
# The --iter refers to the latest iteration this will be

python -m maskgen_pipeline.refine_masks --scene_dir /home/jishnu/Projects/RPX/data/test/1 --iter n

# The --iter portion refers to the latest iteration this will be, since this process can occur multiple times,
# make sure to let the program know which iteration this will be put into.
# Since this is the second iteration, --iter 2 is what will be given

# For this script, please mark the CORRECTLY masked frames, this will be quicker as the faulty frames are already marked. 

python -m maskgen_pipeline.vis_gen_mask --scene_dir /home/jishnu/Projects/RPX/data/test/1 --iter 2

# Step 3: For each scene (e.g. <scene_dir>/) run the following:

# To connect objects ids to masks, run the following script:
# You simply click the number which has the correct object number for the mask that is highlighted
python -m visual_grounding_gt.mask_to_object --scene_dir /home/jishnu/Projects/RPX/data/test/1 --json
\\ visual_grounding_gt/scenes_test.json 

# Step 4: Finally, zip the scene and upload to the correct folder in the box folder:
Upload them here: [here](https://utdallas.app.box.com/folder/342100579130)
```

## Quick demo of the process for 1 scene directory + the correspondance

https://github.com/user-attachments/assets/66527b18-e7ba-4d12-a5b5-82b5858dbeb5
- Sorry for the low quality, size was too big lol


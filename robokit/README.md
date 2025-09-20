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
<<<<<<< HEAD
=======

TODO (jishnu): Add install instructions for RDD if it works

>>>>>>> Fixed maskgen iter 2 + added object_id->mask correspondance
## Download data from Box
- Download one file from [maskgen-iter-1-out](https://utdallas.app.box.com/folder/321198327745?s=59ois7sifqlaoojr0l2isdmk7pkjabga)
- unzip it
- rm the zip file
- then do the label refining for 0/1/2
- then do the object -> mask mapping on the base dir
- zip the results and upload to box

## 📌 Running SAM2 for Ground Truth Mask Generation
```shell
# Step 1: Download the data from [https://utdallas.box.com/s/saifhadoad3w136tbvfgcrd4n2zk8e7t](https://utdallas.app.box.com/folder/321198327745?s=59ois7sifqlaoojr0l2isdmk7pkjabga
)

# Step 2: For each scene directory (e.g., <scene_dir>/0, <scene_dir>/1, <scene_dir>/2), run the following:
<<<<<<< HEAD
python -m maskgen_pipeline.refine_masks --scene_dir /home/jishnu/Projects/RPX/data/test/1

=======
>>>>>>> Fixed maskgen iter 2 + added object_id->mask correspondance
# Instructions:
# This portion is fully automatic, so there are no instructions for this one.


# To visualize the content and mark correct samples
# These faulty samples will be refined later
<<<<<<< HEAD
=======
# (TODO) Need a more concrete multi stage pipeline
python -m maskgen_pipeline.vis_gen_mask --scene_dir /home/jishnu/Projects/RPX/data/test/1 --iter 1

# to refine using iter1_faulty.txt
python -m maskgen_pipeline.refine_masks --scene_dir /home/jishnu/Projects/RPX/data/test/1 --iter 1

# To viz objects masks on a single frame
python -m maskgen_pipeline.viz.single_frame_masks \
/home/jishnu/Projects/RPX/data/scene82.jsom.garden.pot/2/sam2/masks/00000.png

# To convert palette data to mask data (16 bit int labels)
# This is for already processed data; runs fast
# If some scene are not processed then no need for this
python -m maskgen_pipeline.convert_palette_to_int_mask --scene_dir /h
ome/jishnu/Projects/RPX/data/scene83.jsom.garden.pot/2
```
>>>>>>> Fixed maskgen iter 2 + added object_id->mask correspondance

# the --iter portion refers to the latest iteration this will be, since this process can occur multiple times, make sure to let the program know which iteration this will be put into.
# Since this is the second iteration, --iter 2 is what will be given 
python -m maskgen_pipeline.vis_gen_masks --scene_dir /home/jishnu/Projects/RPX/data/test/1 --iter 2

# To connect objects ids to masks, run the following script:
# You simply click the number which has the correct object number for the mask that is highlighted
python visual_grounding_gt/mask_to_object.py --scene_dir /home/jishnu/Projects/RPX/data/test/1 --json visual_grounding_gt/scenes_test.json 


```

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

## Download data from Box
- Download one file from [obj-1](https://utdallas.box.com/s/3n7l6rbmgycfzsr30rowsvkooyzh1ipv), [obj-n](https://utdallas.box.com/s/saifhadoad3w136tbvfgcrd4n2zk8e7t)
- unzip it
- rm the zip file
- then do the label generation process on 0/1/2
- zip the results and upload to box

## 📌 Running SAM2 for Ground Truth Mask Generation
```shell
# Step 1: Download the data from https://utdallas.box.com/s/saifhadoad3w136tbvfgcrd4n2zk8e7t

# Step 2: For each scene directory (e.g., <scene_dir>/0, <scene_dir>/1, <scene_dir>/2), run the following:
python -m maskgen_pipeline.interactive_gsam2 --scene_dir /home/jishnu/Projects/RPX/data/test/1

# Instructions:
# A window opens displaying the last frame with pre-generated GroundingDINO bounding boxes.
# Keep, delete, or add new bounding boxes (right mouse button) around objects of interest.
# Press q to confirm and close the window.
# A second window appears to refine bounding box quality.
# Draw or resize bounding boxes (left mouse button) as needed.
# Press q to finalize.
# SAM2 propagates the bounding boxes backward to generate masks.

# To visualize the content and mark faulty samples
# These faulty samples will be refined later
# (TODO) Need a more concrete multi stage pipeline
python -m maskgen_pipeline.viz.generated_masks --scene_dir /home/jishnu/Projects/RPX/data/test/1

# To viz objects masks on a single frame
python -m maskgen_pipeline.viz.single_frame_masks \
/home/jishnu/Projects/RPX/data/scene82.jsom.garden.pot/2/sam2/masks/00000.png
```

https://github.com/user-attachments/assets/c64415cf-79c2-4e7b-b28a-78bc4a2d51cd


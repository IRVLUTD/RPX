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
python run_sam2_reverse_pipeline.py --scene_dir /home/jishnu/Projects/RPX/data/scene1.library.fountain/0

# Instructions:
# - A window will pop up showing the last frame; draw the bounding box over the object of interest.
# - Press 'y' to save the bounding box.
# - On subsequent runs, the saved bounding box will be displayed.
#   - Press 'n' to use this saved box for mask propagation.
# - After processing, manually verify the outputs inside <scene_dir>/<num>/sam2/contour_masks
```

## ⚠️ (Experimental) GSAM2-Object Pipeline
```shell
# Note: This pipeline is can be used for easy background scenes.
python run_gsam2_reverse_pipeline.py --scene_dir /home/jishnu/Projects/RPX/data/scene1.library.fountain/0
```

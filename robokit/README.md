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

TODO (jishnu): Add install instructions for RDD if it works

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

https://github.com/user-attachments/assets/c64415cf-79c2-4e7b-b28a-78bc4a2d51cd

## Finalizing sample frames
After generating labels/masks, you may want to **subsample frames** from each scene (0/1/2) to avoid redundancy while keeping viewpoint diversity.  

We provide `pds_frames_filter.py` for **order-preserving Poisson Disk Sampling** based on pose distance:  

$$
d = \sqrt{ \| \Delta t \|^2 + (\lambda_{\text{rot}} \cdot \theta)^2 }
$$

where  

- **Δt** = translation difference between two poses  
  $$
  \Delta t = t_i - t_j, \quad \| \Delta t \| = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
  $$  

- **θ** = geodesic rotation angle between two quaternions  
  $$
  \theta = 2 \cdot \arccos(|q_i \cdot q_j|)
  $$  
  where $q_i \cdot q_j = q_{xi}q_{xj} + q_{yi}q_{yj} + q_{zi}q_{zj} + q_{wi}q_{wj}$  

- **λ<sub>rot</sub>** = weighting factor (meters per radian) that balances translation and rotation contributions  

- First & last frames are always kept.  
- Sequence order is preserved.  
- Outputs neat filename lists per subdir.  
- Uses **Numba JIT** for speed.

### Usage
```shell
# Fixed radius (meters in pose space)
python pds_frames_filter.py \
  --scene-dir ./scene \
  --r 0.25 \
  --lambda-rot 0.3

# Target count per subdir (solves radius automatically)
python pds_frames_filter.py \
  --scene-dir ./scene \
  --k 100 \
  --lambda-rot 0.3
```

Example output:
```
# Subdir 0
./scene/0/cam_pose/00000.npz
./scene/0/cam_pose/00042.npz
...

# Subdir 1
./scene/1/cam_pose/00000.npz
./scene/1/cam_pose/00038.npz
...
```

Programmatic use:
```python
from pds_frames_filter import run_per_subdir
results = run_per_subdir("./scene", r=0.25, lambda_rot=0.3)
print(results["0"][:5])
```

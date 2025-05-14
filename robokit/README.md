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
- Download one file
- unzip it
- rm the zip file
- then 1, 2
- zip the results and upload to box

## 1. Prepare data for SAM2
Convert the pngs to jpgs and rename in reverse manner
```shell
python convert2jpg_in_reverse.py --input_dir /home/jishnu/Projects/sam2-rt/data/scene30.gaming2.JUMBOTRONSTAGE/0
```

### 2. Propogate mask in reverse
```shell
python test_bbox_prompt_samv2.py --input_dir /home/jishnu/Projects/sam2-rt/data/scene30.gaming2.JUMBOTRONSTAGE/1/jpg
```

### 3. Filename reverse to match with GT masks
```shell
python reverse_rgb_flenames.py --rgb_dir /home/jishnu/Projects/sam2-rt/data/scene30.gaming2.JUMBOTRONSTAGE/1/rgb
```

### Test model ckpt


TODO:
check which mask corresponds to which; is it first to last or last oto first
at the end it should correspond to rgb/*.png  sequence
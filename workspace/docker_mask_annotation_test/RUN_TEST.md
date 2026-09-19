# RPX Mask Annotation Docker Test

This folder is a local-only test workspace for manually checking the lightweight
Docker image:

```text
irvlutd/rpx-mask-annotation:latest
```

It mounts a small real scene into Docker. No sample data is baked into the image.

## Folder Layout

```text
/media/naren/New Volume/rpx/docker_mask_annotation_test/
  sample_scene/
  checkpoints/
  hf_cache/
  outputs/
  RUN_TEST.md
  prepare_sample_scene.sh
  run_docker_test.sh
```

## Sample Scene

Source scene:

```text
/media/naren/New Volume/rpx/ego_example/2/2
```

Copied subset:

```text
sample_scene/
  rgb/             00000.png through 00009.png
  depth/           matching 00000.png through 00009.png
  cam_pose/        matching 00000.npz through 00009.npz
  fisheye/left/    matching 00000.png through 00009.png
  fisheye/right/   matching 00000.png through 00009.png
```

The original dataset is not modified.

To recreate the sample:

```bash
/media/naren/New\ Volume/rpx/docker_mask_annotation_test/prepare_sample_scene.sh
```

The preparation script also makes `sample_scene/`, `checkpoints/`, `hf_cache/`,
and `outputs/` writable by the container user.

## Checkpoint Status

The checkpoint search used was:

```bash
find "/media/naren/New Volume/rpx" \( -name "sam2.1_hiera_large.pth" -o -name "*.pth" \) | head -50
```

The SAM2 checkpoint is now present for the test workspace:

```text
/media/naren/New Volume/rpx/docker_mask_annotation_test/checkpoints/sam2.1_hiera_large.pth
```

Inside Docker this is mounted as:

```text
/workspace/checkpoints/sam2.1_hiera_large.pth
```

## Start Docker

From the host:

```bash
/media/naren/New\ Volume/rpx/docker_mask_annotation_test/run_docker_test.sh
```

Equivalent Docker command:

```bash
docker run -it --rm \
  -v "/media/naren/New Volume/rpx/github_rpx:/workspace/rpx" \
  -v "/media/naren/New Volume/rpx/docker_mask_annotation_test/sample_scene:/workspace/data/sample_scene" \
  -v "/media/naren/New Volume/rpx/docker_mask_annotation_test/checkpoints:/workspace/checkpoints" \
  -v "/media/naren/New Volume/rpx/docker_mask_annotation_test/hf_cache:/home/mambauser/.cache/huggingface" \
  -e HF_HOME=/home/mambauser/.cache/huggingface \
  -e SAM2_DEVICE=cpu \
  -e SAM2_CHECKPOINT=/workspace/checkpoints/sam2.1_hiera_large.pth \
  -e SAM2_PIPELINE_LOG=/tmp/sam2_pipeline.log \
  -w /workspace/rpx/data/mask_annotation \
  irvlutd/rpx-mask-annotation:latest
```

## GUI/X11

The annotation workflow uses Matplotlib/OpenCV windows. If you are using X11,
run this on the host before starting Docker:

```bash
xhost +local:docker
```

Then make sure Docker receives:

```bash
-e DISPLAY=$DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix
```

`run_docker_test.sh` adds these automatically when `DISPLAY` is set and the X11
socket exists.

## Commands To Run Inside Docker

Check that the scene is visible:

```bash
find /workspace/data/sample_scene/rgb -maxdepth 1 -type f | sort
```

Check CLI help:

```bash
python -m maskgen_pipeline.interactive_gsam2 --help
```

Run the actual annotation/segmentation command:

```bash
python -m maskgen_pipeline.interactive_gsam2 --scene_dir /workspace/data/sample_scene
```

This command opens an interactive bbox UI and then uses GroundingDINO + SAM2.
It will not complete full segmentation until the SAM2 checkpoint exists at:

```text
/workspace/checkpoints/sam2.1_hiera_large.pth
```

The wrapper also sets `SAM2_DEVICE=cpu` so SAM2 stays on CPU in this lightweight
image.

## Outputs To Inspect

If the run starts, expect temporary JPG conversion:

```text
sample_scene/jpg/
```

If full segmentation completes, inspect:

```text
sample_scene/sam2/
sample_scene/sam2/masks/
sample_scene/sam2/palette/
sample_scene/sam2/rgb_and_mask/
sample_scene/sam2/bbox_overlay/
sample_scene/sam2/contour_gt_masks/
```

The pipeline log is written inside the container to:

```text
/tmp/sam2_pipeline.log
```

## Verification Already Performed

- `sample_scene/rgb/` contains real PNG images copied from the real scene.
- Docker can mount and see `/workspace/data/sample_scene`.
- `python -m maskgen_pipeline.interactive_gsam2 --help` works inside Docker.
- The test workspace was made writable for the container user.
- The full 10-frame command completed with `SAM2_DEVICE=cpu`:
  `python -m maskgen_pipeline.interactive_gsam2 --scene_dir /workspace/data/sample_scene`.
- SAM2 initialized image and video predictors on CPU, propagated all 10 frames,
  and wrote outputs under `sample_scene/sam2/`.

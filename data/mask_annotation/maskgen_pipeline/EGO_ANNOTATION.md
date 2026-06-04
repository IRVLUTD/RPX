# Ego (GoPro) Video Annotation — Student Guide

Welcome! This guide walks you through annotating the **GoPro ego videos**, one
scene at a time. You do **not** need any background in computer vision, robotics,
or machine learning. You'll run one command and then click on objects.

If you get stuck at any point, **stop and ask** — that is always better than
guessing. (Whom to ask is at the very bottom.)

---

## 1. The 30-second picture of what you're doing

Every "scene" we recorded has **two cameras**:

| Camera | What it is | Status |
|---|---|---|
| **D435 rig** | The camera mounted on the robot setup. | ✅ Already annotated. |
| **GoPro ego** | A separate GoPro that recorded the same scene (`ego.mp4`). | ⬅️ **Your job.** |

The two cameras were **not** running on a shared clock, so we do **not** try
to line them up frame-by-frame. Instead, the tool simply pulls **250 frames
spread evenly across the whole GoPro video**, and you label the objects in
those 250 frames.

You'll do two things, and **one command runs the extraction for you**:

1. **EXTRACT** *(automatic)* — the tool reads the GoPro video and saves 250
   evenly-spaced frames as PNGs. You do nothing.
2. **LABEL** *(you)* — you outline each object in those frames, the same way
   it was done for the rig camera.

That's it.

---

## 2. One-time setup (do this once, ever)

> You only need to do this section the **first time** on a given computer.
> After that, jump straight to Section 3 for each scene.

You need a computer with an **NVIDIA GPU** (a "CUDA box" — your mentor will tell
you which machine).

Open a terminal and run these, one block at a time:

```bash
# Go to the annotation code.
cd /path/to/RPX/data/mask_annotation

# Create the environment (this matches the rest of the mask pipeline).
export CUDA_HOME=/usr/local/cuda-12.6/
conda create -n rkit-rpx python=3.9 && conda activate rkit-rpx
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
```

> If your mentor already set up the environment on this machine, you can skip
> the `create`/`install` lines and just run `conda activate rkit-rpx`.

**Test that it worked:**

```bash
conda activate rkit-rpx
python -c "import cv2; print('OpenCV', cv2.__version__, '— ready')"
```

If you see something like `OpenCV 4.x.x — ready`, you're good. If you see an
error, stop and ask.

---

## 3. Annotating one scene (the everyday workflow)

Every time you sit down to do a scene, run these two lines first:

```bash
cd /path/to/RPX/data/mask_annotation
conda activate rkit-rpx
```

### Step 1 — EXTRACT 250 frames (one command, automatic)

**Replace the path** with the real scene folder you were given:

```bash
python -m maskgen_pipeline.annotate_ego --scene_dir /path/to/the/scene
```

- `--scene_dir` is the folder for this scene. It contains the GoPro video
  (`ego.mp4` or similar) and the rig camera folders (`0/`, `1/`, `2/`).

If the GoPro mp4 isn't right inside the scene folder, point at it explicitly:

```bash
python -m maskgen_pipeline.annotate_ego \
  --scene_dir /path/to/the/scene \
  --ego_video /path/to/the/ego.mp4
```

You'll see output like:

```
[annotate-ego] scene     : /path/to/the/scene
[annotate-ego] ego video : /path/to/the/scene/ego.mp4
[annotate-ego] wrote 250 frames → /path/to/the/scene/ego/rgb
```

That's the EXTRACT step done.

> If the GoPro video happens to be **shorter than 250 frames**, you'll see a
> `WARNING` saying it took all available frames instead of 250. That's fine —
> short videos are short. Continue to Step 2 as normal.

### Step 2 — LABEL the objects

The tool prints the **exact command** for this step. It uses the **same
labeling tool** the lab already uses for the rig camera, so if you've seen it
before, this is identical:

```bash
python -m maskgen_pipeline.interactive_gsam2 --scene_dir /path/to/the/scene/ego
```

**The one rule that matters:** give each object the **same ID number** it has
in the rig camera. The tool prints where the rig masks live as a reference. If
the cup is object `3` in the rig, it must be object `3` in your ego labels too.

> Want labeling to start automatically right after extraction, without
> copy-pasting the command? Add `--auto_label` to your Step-1 command.

---

## 4. How to know you're done (checklist)

After a scene, peek inside its folder. You should see a new `ego/` folder:

```
<scene>/
  ego/
    ego_frame_map.json   ✅ bookkeeping (created for you)
    rgb/
      00000.png 00001.png ... NNNNN.png  ✅ the 250 (or fewer) extracted frames
      sam2/ ...                           ✅ the masks, after you finish Step 2
```

A scene is **fully done** when:
- [ ] `ego/rgb/` has the extracted frames (250, unless the video was shorter),
- [ ] every object in `ego/rgb/sam2/` is labeled with the **same ID** as the rig.

Then move on to your next assigned scene. 🎉

---

## 5. When something goes wrong

| What you see | What to do |
|---|---|
| `no ego mp4 found` | Pass the file explicitly: `--ego_video /full/path/to/ego.mp4`. |
| `WARNING: ... has only N frames; requested 250` | Fine — the video was short. Continue. |
| `OpenCV` import error or `conda: command not found` | Setup (Section 2) didn't complete. Stop and ask. |
| Anything else, or you're unsure | **Stop and ask.** Don't guess. |

---

## 6. Quick reference (the whole thing on one screen)

```bash
# every session:
cd /path/to/RPX/data/mask_annotation
conda activate rkit-rpx

# one scene:
python -m maskgen_pipeline.annotate_ego --scene_dir /path/to/the/scene
#   Step 1 (auto): 250 evenly-spaced frames into <scene>/ego/rgb/
#   Step 2 (you):  python -m maskgen_pipeline.interactive_gsam2 --scene_dir /path/to/the/scene/ego
#                  -> give each object the SAME id as the rig camera
```

---

## 7. Who to ask

If you're stuck for more than a few minutes, or anything in this guide doesn't
match what you see on screen, message **your mentor / the person who assigned you
these scenes**. Include: the scene folder name, the command you ran, and a
screenshot or copy of the error. Asking early saves everyone time.

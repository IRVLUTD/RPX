# `data_capture/` — Intel D435 + T265 capture rig

End-to-end capture pipeline for one RPX scene: synchronized **RGB + depth
(Intel RealSense D435)** with **6-DoF pose (Intel RealSense T265)**, logged
into a per-task directory the rest of the toolkit consumes.

This is what you run on the **capture machine** — the laptop or workstation
the cameras are plugged into. You **don't** need this if you're just
benchmarking models on the already-captured dataset (use
[`benchmark/`](../benchmark/README.md) for that).

---

## ⚡ Quick start (one scene)

```bash
# 0. Plug in both cameras (D435 first, then T265).
# 1. Reset USB if either device is in a stuck state:
python reset_vision_usb.py --name_list 4xx,Intel

# 2. Capture the scene at 30 fps with a 5 ms sync threshold:
python save_device_data.py <task-or-object-name> 30 5

# 3. Eyeball the captured trajectory:
python plot_poses.py recorded_data/<task-or-object-name>/cam_pose/
```

Output goes to `recorded_data/<task-or-object-name>/` with one file per
modality per frame. The benchmark loader reads exactly this layout.

---

## 🛠️ Requirements

- **Intel RealSense D435** (RGB-D, mandatory)
- **Intel RealSense T265** (6-DoF VIO, mandatory)
- `librealsense2 == 2.47.0` and `pyrealsense2 == 2.47.0.3313` — **pinned**
  because T265 support was removed in librealsense > 2.47.0. Install
  instructions are in the [root README](../README.md#-data-capture-prerequisites).

Verify both devices enumerate:

```bash
lsusb | grep -E "RealSense|Movidius"
```

You should see two lines — one per camera. If you don't, run the reset
script above.

---

## 📂 What each script does

| Script | Purpose |
|---|---|
| `save_device_data.py <name> <fps> <sync_ms>` | Main capture loop. Streams RGB+depth+pose+fisheye-L/R at the given fps; only writes frames whose D435 and T265 timestamps are within `<sync_ms>` of each other. |
| `reset_vision_usb.py --name_list <a,b>` | Power-cycles the listed USB devices via `usbreset`. Useful when a camera locks up between captures. |
| `plot_poses.py <pose-dir>` | Renders the T265 trajectory as a 3D matplotlib plot — fast sanity check after a capture. |
| `test_t265.py [--save]` | Lightweight T265-only loop: prints pose + fisheye stream. `--save` writes the xyz trajectory. |
| `test_d4xx.py` | D435-only smoke test: opens the depth + colour streams and prints frame counts. |
| `test_tracking_n_depth_cameras.py` | Joint D435 + T265 smoke test (no save) — useful to check that both come up before a real capture. |
| `check_devices_connected.py` | One-shot: lists every connected RealSense device with its serial number. |
| `test_callbacks.py` | Callback-based capture variant (alternative to the polling loop in `save_device_data.py`). |

Per-modality config (preset filters, serial numbers, D435 JSON config)
lives in [`config/`](config/).

Two small sample T265 captures from earlier dev sessions are kept in
[`data/T265/`](data/T265/) so `plot_poses.py` has something to render
out-of-the-box.

---

## 🐳 Docker

A pre-built Docker image with all RealSense dependencies is available
under [`../docker/`](../docker/README.md). Use it if you don't want to
deal with the librealsense pin on your host.

---

## 🔗 Related

- [`../benchmark/`](../benchmark/README.md) — benchmark a model against
  already-captured RPX scenes (the common case; doesn't need any of these
  scripts).
- [`../mask_pipeline/`](../mask_pipeline/README.md) — generate
  ground-truth instance masks for a captured scene.

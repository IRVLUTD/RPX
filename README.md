# 🧠 Robot Perception X (RPX)

The repository for the **Robot Perception X (RPX)** dataset and benchmarking suite.

---

## 📚 Index

1. [Installing RealSense Libraries](#1-installing-realsense-libraries)  
   1.1 [Installing Dependencies](#11-installing-dependencies)  
   1.2 [Cloning SDK v2.47.0](#12-cloning-sdk-v2470)  
   1.3 [Installation and Build](#13-installation-and-build)  
   1.4 [Install Python Wrapper](#14-install-python-wrapper)


---

## 1. Installing RealSense Libraries

`librealsense2` is the core Intel RealSense library. `realsense-viewer` and `pyrealsense2` wrapper are built on top of it.

> :warning: **Currently, support for the T265 module is removed in the latest versions of `librealsense2`.** Hence, we need to install an older version (v2.47.0) to access T265.

The installation process is adapted directly from the [official installation docs](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md). This guide builds from source for more control. Alternatively, you can use `.dpkg` distributions from [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).

> :warning: **Make sure to disconnect all RealSense devices before proceeding with the installation!**

### 1.1 Installing Dependencies
```bash
sudo apt-get update && sudo apt install     libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev     git wget cmake build-essential libglfw3-dev libgl1-mesa-dev     libglu1-mesa-dev at
```

### 1.2 Cloning SDK v2.47.0
```bash
git clone -b v2.47.0 https://github.com/IntelRealSense/librealsense.git
```

### 1.3 Installation and Build
```bash
cd librealsense && ./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake ../ -DBUILD_EXAMPLES=true -DBUILDTYPE=Release
sudo make uninstall && make clean && make && sudo make -j12 install
```

> **Tip:** `-j12` uses 12 CPU cores to build. Adjust this based on your machine. Leave at least 2 cores free to avoid system hangs.

You can now launch the viewer using:
```bash
realsense-viewer
```
Then connect the cameras to view data.

### 1.4 Install Python Wrapper
```bash
pip install pyrealsense2==2.47.0.3313
```
> **Note:** This version should **match** the installed `librealsense2` version.

---


# Run sam2 for gt masks
```shell
# Download data from https://utdallas.box.com/s/saifhadoad3w136tbvfgcrd4n2zk8e7t
# for each scene, run for <0,1,2>
# first draw bbox on the popped up image
# then press y to save; next time when ran then this saved bbox prompt will be read (displayed) and used (when n is pressed after display)
# manually verify the output created on <0,1,2>/sam2/contour_masks
python run_sam2_reverse_pipeline.py --scene_dir /home/jishnu/Projects/RPX/data/scene1.library.fountain/0


# for sgam2-obj pipeline; doesn't work
python run_gsam2_reverse_pipeline.py --scene_dir /home/jishnu/Projects/RPX/data/scene1.library.fountain/0

```

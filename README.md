# RPX
The repository for Robot Perception X dataset and benchmarking

# Installing Realsense libraries
```librealsense2``` is the core intel realsense library. ```realsense-viewer``` and ```pyrealsense2``` wrapper are built on top of it. 
Currently the support for T265 module is removed in the latest versions of ```librealsense2``` library. Hence we need to install older version of the sdk to acess T265.

The installation process is adapted directly from [Installation docs](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md). This version is to build from source. Installation can also be done via ```dpkg``` format distributions from [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md). But let's prefer to build from source for more control. 

:warning: $\color{red} \text{Make sure to disconnect all the realsense devices before proceeding with the installation} $

### A. Installing dependencies
```
sudo apt-get update && sudo apt install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev git wget cmake build-essential libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
```

### B. Clone sdk version 2.47.0
```
git clone -b v2.47.0 https://github.com/IntelRealSense/librealsense.git
```

### C. Installation and Build
```
cd librealsense && ./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake ../ -DBUILD_EXAMPLES=true -DBUILDTYPE=Release
sudo make uninstall && make clean && make && sudo make -j12 install
```
```-j12``` indicates to use 12 core of cpu to build. You may change it as you desire. the more cores, the faster it build. But be sure to leave out atleast 2 cores, so the system doesn't hang up.

Launch the viewer using ```realsense-vievwer``` and connect the cameras to view the data. 

### D. Install Python wrapper
```
pip install pyrealsense2==2.47.0.3313
```
***Note***: This version should match the vesion of librealsense2 installed earlier.








# RPX Visualization - vis Folder
This folder contains visualization tools for the RPX Project, including scripts to process and display RGB-D data using Rerun.

### Downloading and Setting Your Data Path
Download the dataset you want to visualize.

Note the path to your dataset directory.

Edit config.yaml and replace the BASE_PATH value with your dataset path.

Example:
```yaml
BASE_PATH: C:\Users\govin\OneDrive\Robotics Research\RPX_Project\RPX\data\scene1.library.fountain\2
```

### Expected Data Directory Structure
Your data_path directory must have this structure:
```css
<your data_path>/
├── depth/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── rgb/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── cam_pose/
│   ├── 00000.npz
│   ├── 00001.npz
│   └── ...
├── fisheye/
│   ├── left/
│   │   ├── 00000.jpg
│   │   └── ...
│   └── right/
│       ├── 00000.jpg
│       └── ...
```




## Quick Start
Follow these steps to set up your environment and run the visualization script.

## Python Version
**Python 3.12.x is required.**
## Setup Instructions

### 1️.) Create a Virtual Environment
```powershell
python -m venv .venv
```


### 2️.) Activate the Virtual Environment

PowerShell:
```powershell
.\.venv\Scripts\Activate
```
Command Prompt:
```cmd
.venv\Scripts\activate.bat
```
macOS/Linux:
```bash
source .venv/bin/activate
```

### 3️.) Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4.) Running the Script

After activating your environment, run:
```powershell
python RerunVis.py
```

### Understanding Downsampling and Memory Limits
Rerun has memory constraints when visualizing large point clouds.
If you show all pixels in every frame, memory usage grows rapidly.
Eventually, the viewer stops displaying the most recent frames.
To avoid this, we downsample each frame to a limited number of points (MAX_POINTS_PER_FRAME).


🔹 Tradeoffs

Higher downsampling (lower MAX_POINTS_PER_FRAME):
Less detail per frame.
Can display more frames before running out of memory.

Lower downsampling (higher MAX_POINTS_PER_FRAME):
Better visual quality.
Fewer total frames will be displayed before memory fills up.

Recommendation: Start with the default (2000) and adjust as needed based on your GPU and dataset size.

Note:
The script enforces a maximum total point cloud size (MAX_ACCUMULATED_POINTS) to 
prevent the viewer from running out of memory as more frames are processed.

🔹 Truncated Depth
MAX_DEPTH_METERS specifies how far points are included in visualization.
Points beyond this distance are ignored (to avoid clutter and performance issues).

Example:
If MAX_DEPTH_METERS: 2.0, all pixels farther than 2 meters will be excluded.

### 4.) Using the Configuration File
This script uses a config.yaml file to control key parameters. You only ever need to change these parameters


### What You Will See When You Run the Script

When you launch the visualization, the Rerun Viewer will open in your web browser or a dedicated window. 
You will see a few of these multiple panels (“viewports”) automatically displayed:

- **Point Clouds**
- **RGB View** 
- **Depth View** 
- **Camera Frustums(RGB 3D axis)** 
- **Camera Trail**
- **Fisheye View**

The rest need to be manually added in Rerun to display

### How to Add Viewports

Watch the demonstration video for detailed instructions on how to display 
RGB images, depth images, the 3D point cloud, and camera frustums in the Rerun Viewer.

### Navigating the Rerun UI

Keys
- A: Left
- D: Right
- W: Zoom in
- S: Zoom out
- Q: Down
- E: Up
  
Mouse
- Click&Drag: Rotate
- Scroll: Zoom In/Out

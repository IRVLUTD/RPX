import numpy as np
import cv2
import plotly.graph_objects as go

# Intrinsics
intrinsic_matrix = [
    [386.6580810546875, 0, 323.2966003417969],
    [0, 386.6580810546875, 237.33880615234375],
    [0, 0, 1]
]
fx, fy = intrinsic_matrix[0][0], intrinsic_matrix[1][1]
px, py = intrinsic_matrix[0][2], intrinsic_matrix[1][2]

# Load depth and RGB images
depth_img_path = "/home/jishnu/rpx-data/rpx-data.1/air_duster_can/depth/00093.png"  # replace
rgb_img_path = "/home/jishnu/rpx-data/rpx-data.1/air_duster_can/rgb/00093.png"      # replace

depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
rgb_img = cv2.cvtColor(cv2.imread(rgb_img_path), cv2.COLOR_BGR2RGB)

# Convert mm to meters if needed
# if depth_img.max() > 255:
depth_img /= 1000.0

height, width = depth_img.shape
rgb_img = cv2.resize(rgb_img, (width, height))

# Compute XYZ from depth
def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)
    return xyz_img

xyz = compute_xyz(depth_img, fx, fy, px, py, height, width)
points = xyz.reshape(-1, 3)
colors = rgb_img.reshape(-1, 3)

# Filter out invalid depths
# valid = points[:, 2] > 0
# points = points[valid]
# colors = colors[valid]

# Normalize colors to hex for Plotly
colors_hex = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

# Subsample if too many points
MAX_POINTS = 100_000
if len(points) > MAX_POINTS:
    idx = np.random.choice(len(points), MAX_POINTS, replace=False)
    points = points[idx]
    colors_hex = [colors_hex[i] for i in idx]

# Plotly 3D scatter
fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    mode='markers',
    marker=dict(size=1, color=colors_hex, opacity=0.8)
)])
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z',
    aspectmode='data'
))
fig.show()

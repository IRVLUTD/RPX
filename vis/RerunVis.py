import os
import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import yaml

# === Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# === Default fallbacks in case keys are missing
BASE_PATH = config.get("BASE_PATH", "C:/path/to/your/default/data/")

NUM_FRAMES = config.get("NUM_FRAMES", 250)
MAX_DEPTH_METERS = config.get("MAX_DEPTH_METERS", 2.0)
MAX_POINTS_PER_FRAME = config.get("MAX_POINTS_PER_FRAME", 2000)
MAX_ACCUMULATED_POINTS = config.get("MAX_ACCUMULATED_POINTS", 4000000)

# === Validate BASE_PATH
if not os.path.exists(BASE_PATH):
    raise FileNotFoundError(f"Data path does not exist: {BASE_PATH}")

print("Using data path:", BASE_PATH)
print("NUM_FRAMES:", NUM_FRAMES)
print("MAX_DEPTH_METERS:", MAX_DEPTH_METERS)
print("MAX_POINTS_PER_FRAME:", MAX_POINTS_PER_FRAME)
print("MAX_ACCUMULATED_POINTS:", MAX_ACCUMULATED_POINTS)

# === Config ===
DEPTH_SCALE = 0.001  # 1mm -> meters
FX, FY = 390.469, 390.469
CX, CY = 326.312, 245.182

DEPTH_DIR = os.path.join(BASE_PATH, "depth")
RGB_DIR = os.path.join(BASE_PATH, "rgb")
POSE_DIR = os.path.join(BASE_PATH, "cam_pose")
FISHEYE_LEFT_DIR = os.path.join(BASE_PATH, "fisheye", "left")
FISHEYE_RIGHT_DIR = os.path.join(BASE_PATH, "fisheye", "right")

# === Rerun setup ===
rec = rr.RecordingStream(application_id="RGBD Viewer - Accum + Current + All Frustums")
rr.set_global_data_recording(rec)

# === Accumulation containers
all_world_points = []
all_colors = []
camera_positions = []

# === Fixed rotation correction (180 degrees around X axis)
fixed_rot_X180 = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

for index in range(NUM_FRAMES):
    idx = f"{index:05d}"
    pose_path = os.path.join(POSE_DIR, f"{idx}.npz")
    depth_path = os.path.join(DEPTH_DIR, f"{idx}.png")
    rgb_path = os.path.join(RGB_DIR, f"{idx}.png")
    fisheye_left_path = os.path.join(FISHEYE_LEFT_DIR, f"{idx}.png")
    fisheye_right_path = os.path.join(FISHEYE_RIGHT_DIR, f"{idx}.png")

    if not (os.path.exists(pose_path) and os.path.exists(depth_path) and os.path.exists(rgb_path)):
        continue

    pose = np.load(pose_path)
    position = pose["position"]
    quat = np.asarray(pose["orientation"], dtype=np.float32)

    rot_matrix = R.from_quat(quat).as_matrix()
    corrected_rot = rot_matrix @ fixed_rot_X180

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if depth is None or rgb is None:
        continue

    h, w = depth.shape
    rec.set_time("frame", sequence=index)

    # === Log transform for this frame
    rec.log(f"scene/cameras/frame_{index}/transform", rr.Transform3D(
        translation=position,
        rotation=rr.Quaternion(xyzw=R.from_matrix(corrected_rot).as_quat())
    ))

    # === Log frustum for this frame (so all persist over time)
    rec.log(f"scene/cameras/frame_{index}/frustum", rr.Arrows3D(
        origins=[position] * 3,
        vectors=corrected_rot.T * 0.10,
        colors=[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
    ))

    # === Compute 3D points in camera space
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth.astype(np.float32) * DEPTH_SCALE
    mask = (zs > 0) & (zs < MAX_DEPTH_METERS)

    xs = (us - CX) * zs / FX
    ys = (vs - CY) * zs / FY

    cam_points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    cam_points = cam_points[mask.flatten()]

    # === Transform to world coordinates
    ones = np.ones((cam_points.shape[0], 1))
    cam_hom = np.hstack((cam_points, ones))
    transform = np.eye(4)
    transform[:3, :3] = corrected_rot
    transform[:3, 3] = position
    world_points = (transform @ cam_hom.T).T[:, :3]

    colors = rgb.reshape(-1, 3)[mask.flatten()][:, ::-1]

    camera_positions.append(position.copy())

    # === Downsample per frame for accumulated reconstruction
    downsampled_world = world_points
    downsampled_colors = colors
    if world_points.shape[0] > MAX_POINTS_PER_FRAME:
        idxs = np.random.choice(world_points.shape[0], MAX_POINTS_PER_FRAME, replace=False)
        downsampled_world = world_points[idxs]
        downsampled_colors = colors[idxs]

    all_world_points.append(downsampled_world)
    all_colors.append(downsampled_colors)

    # === Stack accumulated so far
    stacked_points = np.vstack(all_world_points)
    stacked_colors = np.vstack(all_colors)

    if stacked_points.shape[0] > MAX_ACCUMULATED_POINTS:
        idxs = np.random.choice(stacked_points.shape[0], MAX_ACCUMULATED_POINTS, replace=False)
        stacked_points = stacked_points[idxs]
        stacked_colors = stacked_colors[idxs]

    # === Log accumulated reconstruction
    rec.log("scene/accumulated_reconstruction", rr.Points3D(
        positions=stacked_points,
        colors=stacked_colors
    ))

    # === Log current frame point cloud (so it only shows the current one)
    rec.log("scene/current_frame/pointcloud", rr.Points3D(
        positions=world_points,
        colors=colors
    ))

    # === Also log the current frustum again under scene/current_frame so it shows together
    rec.log("scene/current_frame/frustum", rr.Arrows3D(
        origins=[position] * 3,
        vectors=corrected_rot.T * 0.10,
        colors=[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
    ))

    # === Log images
    rec.log("scene/images/rgb", rr.Image(rgb[:, :, ::-1]).compress(jpeg_quality=95))
    rec.log("scene/images/depth", rr.DepthImage(depth, meter=DEPTH_SCALE))

    if os.path.exists(fisheye_left_path):
        fisheye_left_img = cv2.imread(fisheye_left_path, cv2.IMREAD_GRAYSCALE)
        if fisheye_left_img is not None:
            rec.log("scene/fisheye/left", rr.Image(fisheye_left_img))

    if os.path.exists(fisheye_right_path):
        fisheye_right_img = cv2.imread(fisheye_right_path, cv2.IMREAD_GRAYSCALE)
        if fisheye_right_img is not None:
            rec.log("scene/fisheye/right", rr.Image(fisheye_right_img))

    if index % 50 == 0:
        print(f"Processed frame {index}")

# === Log camera trail
rec.log("scene/camera_trail", rr.LineStrips3D([np.array(camera_positions)]))

print("✅ All reconstructions and frustums logged! Launching viewer...")
rr.spawn()

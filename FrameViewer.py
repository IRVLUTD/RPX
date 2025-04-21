import os
import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import pygetwindow as gw
import pyautogui
import time

# Config
NUM_FRAMES = 500
DEPTH_SCALE = 0.001
FX, FY = 390.469, 390.469
CX, CY = 326.312, 245.182

BASE_PATH = r'C:\Users\govin\OneDrive\Robotics Research\Rerun\air_duster_can\0'
DEPTH_DIR = os.path.join(BASE_PATH, 'depth')
RGB_DIR = os.path.join(BASE_PATH, 'rgb')
POSE_DIR = os.path.join(BASE_PATH, 'cam_pose')
CATEGORIES = [r'depth', r'fisheye\left', r'fisheye\right', r'rgb']

def get_frame_paths(index):
    return [os.path.join(BASE_PATH, cat, f'{index:05d}.png') for cat in CATEGORIES]

def load_frame(index):
    paths = get_frame_paths(index)
    images = [cv2.imread(p) for p in paths]
    if any(img is None for img in images):
        print(f"Missing frame at index {index}")
        return None
    return images

def resize_images(images, height=240):
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = height / h
        resized.append(cv2.resize(img, (int(w * scale), height)))
    return resized

def move_windows():
    time.sleep(2)  # Give time for both windows to appear
    try:
        cv2_win = gw.getWindowsWithTitle('Frame Viewer')[0]
        rerun_win = gw.getWindowsWithTitle('Rerun Viewer')[0]
        screen_width, screen_height = pyautogui.size()

        half_h = int(screen_height / 2)

        # Resize and position OpenCV viewer
        cv2_win.resizeTo(screen_width, half_h)
        cv2_win.moveTo(0, 0)

        # Resize and position Rerun viewer
        rerun_win.resizeTo(screen_width, half_h)
        rerun_win.moveTo(0, half_h)

    except IndexError:
        print("⚠️ Could not find one or both viewer windows.")

def display_frame(index):
    images = load_frame(index)
    if images is None:
        return
    images = resize_images(images)
    combined = np.hstack(images)
    cv2.imshow('Frame Viewer', combined)

def main():
    rr.init("3D RGB-D Playback", spawn=True)
    camera_positions = []

    index = 0
    cv2.namedWindow('Frame Viewer', cv2.WINDOW_NORMAL)
    display_frame(index)
    move_windows()  # Rearrange window layout

    playing_forward = False
    playing_backward = False
    key_held = None

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break

        elif key == ord('d') or key == 83:
            if index < NUM_FRAMES - 1:
                index += 1
                display_frame(index)
            playing_forward = True
            playing_backward = False
            key_held = 'd'

        elif key == ord('a') or key == 81:
            if index > 0:
                index -= 1
                display_frame(index)
            playing_backward = True
            playing_forward = False
            key_held = 'a'

        elif key == 255:
            if key_held == 'd' and playing_forward and index < NUM_FRAMES - 1:
                index += 1
                display_frame(index)
            elif key_held == 'a' and playing_backward and index > 0:
                index -= 1
                display_frame(index)

        else:
            playing_forward = False
            playing_backward = False
            key_held = None

        if key != 255:
            playing_forward = False
            playing_backward = False
            key_held = None

        # === Rerun Logging ===
        idx = f"{index:05d}"
        depth = cv2.imread(os.path.join(DEPTH_DIR, f"{idx}.png"), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(os.path.join(RGB_DIR, f"{idx}.png"), cv2.IMREAD_COLOR)
        pose = np.load(os.path.join(POSE_DIR, f"{idx}.npz"))

        if depth is None or rgb is None:
            continue

        position = pose['position']
        quat = pose['orientation']
        rotation_matrix = R.from_quat(quat).as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position

        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        zs = depth.astype(np.float32) * DEPTH_SCALE
        xs = (us - CX) * zs / FX
        ys = (vs - CY) * zs / FY

        points_cam = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        valid = zs.flatten() > 0
        points_cam = points_cam[valid]

        ones = np.ones((points_cam.shape[0], 1))
        cam_hom = np.hstack((points_cam, ones))
        points_world = (transform @ cam_hom.T).T[:, :3]

        colors = rgb.reshape(-1, 3)[valid][:, ::-1]

        if points_world.shape[0] > 50_000:
            idxs = np.random.choice(points_world.shape[0], 50_000, replace=False)
            points_world = points_world[idxs]
            colors = colors[idxs]

        rr.set_time_sequence("frame", index)
        rr.log("scene/points", rr.Points3D(positions=points_world, colors=colors))
        rr.log(f"scene/camera/{idx}", rr.Transform3D(translation=position, rotation=quat, from_parent=True))
        rr.log("viewer/camera", rr.Transform3D(translation=position, rotation=quat, from_parent=True))
        camera_positions.append(position.copy())
        if len(camera_positions) >= 2:
            rr.log("scene/camera_trail", rr.LineStrips3D([np.array(camera_positions)]))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
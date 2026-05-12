import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import threading
import time

from config.serial_nums import T265_serial_num, D4xx_serial_num

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./data/{timestamp}"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/rgb", exist_ok=True)
os.makedirs(f"{save_dir}/depth", exist_ok=True)
os.makedirs(f"{save_dir}/pose", exist_ok=True)

file_index = 0
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS 

T265_pipeline = rs.pipeline()
D4xx_pipeline = rs.pipeline()

T265_config = rs.config()
T265_config.enable_device(T265_serial_num)
T265_config.enable_stream(rs.stream.pose)

D4xx_config = rs.config()
D4xx_config.enable_device(D4xx_serial_num)
D4xx_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
D4xx_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)

frame_buffer = {
    "pose": None,
    "color": None,
    "depth": None,
    "pose_timestamp": None,
    "color_timestamp": None,
    "depth_timestamp": None
}
frame_lock = threading.Lock()

align = rs.align(rs.stream.color)
SYNC_THRESHOLD_MS = 10  # Sync tolerance in milliseconds

def T265_callback(frame):
    """Handles T265 pose frames."""
    pose_frame = frame.as_pose_frame()
    if pose_frame:
        with frame_lock:
            frame_buffer["pose"] = pose_frame
            frame_buffer["pose_timestamp"] = pose_frame.get_timestamp()  # Hardware timestamp. Refer to screenshots

def D4xx_callback(frame):
    """Handles D4xx color & depth frames."""
    frames = frame.as_frameset()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if color_frame and depth_frame:
        with frame_lock:
            frame_buffer["color"] = color_frame
            frame_buffer["depth"] = depth_frame
            frame_buffer["color_timestamp"] = color_frame.get_timestamp()  # Hardware timestamp
            frame_buffer["depth_timestamp"] = depth_frame.get_timestamp()

T265_pipeline.start(T265_config, T265_callback)
D4xx_pipeline.start(D4xx_config, D4xx_callback)

T265_device = T265_pipeline.get_active_profile().get_device()
D4xx_device = D4xx_pipeline.get_active_profile().get_device()

# Enable global timestamp synchronization . not working 
# D4xx_device.as_rs400_device().set_option(rs.option.global_time_enabled, 1)

depth_sensor = D4xx_device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale} meters per unit")

print("Waiting 4 seconds")
time.sleep(4)

try:
    last_capture_time = time.time()

    while True:
        with frame_lock:
            pose_frame = frame_buffer["pose"]
            color_frame = frame_buffer["color"]
            depth_frame = frame_buffer["depth"]
            pose_timestamp = frame_buffer["pose_timestamp"]
            color_timestamp = frame_buffer["color_timestamp"]
            depth_timestamp = frame_buffer["depth_timestamp"]

        if pose_frame and color_frame and depth_frame:

            pose_color_diff = abs(pose_timestamp - color_timestamp)
            pose_depth_diff = abs(pose_timestamp - depth_timestamp)

            if pose_color_diff < SYNC_THRESHOLD_MS and pose_depth_diff < SYNC_THRESHOLD_MS:
                current_time = time.time()
                if current_time - last_capture_time >= FRAME_INTERVAL:
                    pose_data = pose_frame.get_pose_data()
                    position = [pose_data.translation.x, pose_data.translation.y, pose_data.translation.z]
                    orientation = [pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w]

                    depth_image = (np.asanyarray(depth_frame.get_data()) * depth_scale * 1000.0).astype(np.uint16)
                    color_image = np.asanyarray(color_frame.get_data())

                    rgb_filename = f"{save_dir}/rgb/{file_index:05d}.png"
                    depth_filename = f"{save_dir}/depth/{file_index:05d}.png"
                    pose_filename = f"{save_dir}/pose/{file_index:05d}.npz"

                    cv2.imwrite(rgb_filename, color_image)
                    cv2.imwrite(depth_filename, depth_image)
                    np.savez(pose_filename, position=position, orientation=orientation)

                    print(f"Saved: {rgb_filename}, {depth_filename}, {pose_filename}")
                    print(f"Pose-Color Latency: {pose_color_diff:.2f} ms, Pose-Depth Latency: {pose_depth_diff:.2f} ms")

                    file_index += 1  
                    last_capture_time = current_time  # Update last capture time

finally:
    T265_pipeline.stop()
    D4xx_pipeline.stop()
    print("Streaming stopped.")

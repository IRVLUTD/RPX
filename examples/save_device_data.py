import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import time

from config.serial_nums import T265_serial_num, D4xx_serial_num

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./data/{timestamp}"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/rgb", exist_ok=True)
os.makedirs(f"{save_dir}/depth", exist_ok=True)
os.makedirs(f"{save_dir}/pose", exist_ok=True)
os.makedirs(f"{save_dir}/fisheye/left", exist_ok=True)
os.makedirs(f"{save_dir}/fisheye/right", exist_ok=True)


file_index = 0

T265_pipeline = rs.pipeline()
D4xx_pipeline = rs.pipeline()

T265_config = rs.config()
T265_config.enable_device(T265_serial_num)
T265_config.enable_stream(rs.stream.pose)
# Add fisheye stream
T265_config.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.raw8, 30)
T265_config.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.raw8, 30)

D4xx_config = rs.config()
D4xx_config.enable_device(D4xx_serial_num)
D4xx_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
D4xx_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60) #saving at 60 is reducing latency. check in vieweer as awell

T265_profile = T265_pipeline.start(T265_config)
D4xx_profile = D4xx_pipeline.start(D4xx_config)

align = rs.align(rs.stream.color)
depth_sensor = D4xx_pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print(f"Depth Scale: {depth_scale} meters per unit")

frame_interval = 1 / 20  # 20fps
sync_threshold_ms = 80  # Sync threshold in milliseconds  #10 - default; 80 - when including fisheye. 

# Discard for 10 seconds after starting. bad data
for _ in range(600):
    T265_frames = T265_pipeline.wait_for_frames()
    D4xx_frames = D4xx_pipeline.wait_for_frames()

start_time1 = time.time()  # total pipeline start time
try:
    while True:
        start_time = time.time()  

        T265_frames = T265_pipeline.wait_for_frames()
        D4xx_frames = D4xx_pipeline.wait_for_frames()

        aligned_frames = align.process(D4xx_frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        pose_frame = T265_frames.get_pose_frame()
        left_fisheye = T265_frames.get_fisheye_frame(1)
        right_fisheye = T265_frames.get_fisheye_frame(2)


        if pose_frame and color_frame and depth_frame and left_fisheye and right_fisheye:
            pose_ts = pose_frame.get_timestamp()
            color_ts = color_frame.get_timestamp()
            depth_ts = depth_frame.get_timestamp()

            pose_color_latency = abs(pose_ts - color_ts)  # Pose-to-Color latency (ms)
            pose_depth_latency = abs(pose_ts - depth_ts)  # Pose-to-Depth latency (ms)

            if pose_color_latency < sync_threshold_ms and pose_depth_latency < sync_threshold_ms:
                # for now lets compare only rgb and depth latencies w.r.t pose. ususally if pose is in sync, so does stereo too. --check
                pose_data = pose_frame.get_pose_data()
                position = [pose_data.translation.x, pose_data.translation.y, pose_data.translation.z]
                orientation = [pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w]

                depth_image = (np.asanyarray(depth_frame.get_data()) * depth_scale * 1000.0).astype(np.uint16)
                color_image = np.asanyarray(color_frame.get_data())

                left_fisheye_image = np.asanyarray(left_fisheye.get_data())
                right_fisheye_image = np.asanyarray(right_fisheye.get_data())

                rgb_filename = f"{save_dir}/rgb/{file_index:05d}.png"
                depth_filename = f"{save_dir}/depth/{file_index:05d}.png"
                pose_filename = f"{save_dir}/pose/{file_index:05d}.npz"

                left_fisheye_filename = f"{save_dir}/fisheye/left/{file_index:05d}.png"
                right_fisheye_filename = f"{save_dir}/fisheye/right/{file_index:05d}.png"

                cv2.imwrite(rgb_filename, color_image)
                cv2.imwrite(depth_filename, depth_image)
                cv2.imwrite(left_fisheye_filename, left_fisheye_image)
                cv2.imwrite(right_fisheye_filename, right_fisheye_image)

                np.savez(pose_filename, position=position, orientation=orientation)

                print(f"saving {file_index} th data point\n")
                print(f"Saved: {rgb_filename}, {depth_filename}, {pose_filename}, {left_fisheye_filename}, {right_fisheye_filename}")
                print(f"Pose-Color Latency: {pose_color_latency:.2f} ms, Pose-Depth Latency: {pose_depth_latency:.2f} ms")

                file_index += 1  
        """
        If i sleep for required ferequency, time stamps are not getting matched often. so leave it as it is
        and store maximum possible synced frames. 
        """
        # Maintain 20 FPS
        # elapsed_time = time.time() - start_time
        # sleep_time = max(0, frame_interval - elapsed_time)
        # time.sleep(sleep_time)

finally:
    final_time1 = time.time()  # Pipeline end time
    print(-start_time1+final_time1)
    T265_pipeline.stop()
    D4xx_pipeline.stop()
    print("Streaming stopped.")

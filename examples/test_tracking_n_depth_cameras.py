import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

from config.serial_nums import T265_serial_num, D4xx_serial_num

# File system - similar template as of tto and iTeach
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./data/{timestamp}"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/rgb", exist_ok=True)
os.makedirs(f"{save_dir}/depth", exist_ok=True)
os.makedirs(f"{save_dir}/pose", exist_ok=True)


file_index = 0

T265_pipeline = rs.pipeline()
D4xx_pipeline = rs.pipeline()

T265_config = rs.config()
T265_config.enable_device(T265_serial_num)
T265_config.enable_stream(rs.stream.pose)

D4xx_config = rs.config()
D4xx_config.enable_device(D4xx_serial_num)
D4xx_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
D4xx_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

T265_pipeline.start(T265_config)
D4xx_pipeline.start(D4xx_config)

align = rs.align(rs.stream.color)

SYNC_THRESHOLD_MS = 30  # in milliseconds
define_intrinsics = False
frequency = 10 

depth_sensor = D4xx_pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale} meters per unit")

try:
    while True:
        T265_frames = T265_pipeline.wait_for_frames()
        D4xx_frames = D4xx_pipeline.wait_for_frames()

        pose_frame = T265_frames.get_pose_frame()
        aligned_frames = align.process(D4xx_frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not define_intrinsics:
            define_intrinsics = True
            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            fx, fy = intrinsics.fx, intrinsics.fy  
            cx, cy = intrinsics.ppx, intrinsics.ppy  
            print(f"Color Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        if pose_frame and color_frame and depth_frame:
            pose_ts = pose_frame.get_timestamp()
            color_ts = color_frame.get_timestamp()
            depth_ts = depth_frame.get_timestamp()

            pose_color_diff = abs(pose_ts - color_ts)
            pose_depth_diff = abs(pose_ts - depth_ts)

            print(f"Time Differences -> Pose-Color: {pose_color_diff:.2f} ms, Pose-Depth: {pose_depth_diff:.2f} ms")

            if pose_color_diff < SYNC_THRESHOLD_MS and pose_depth_diff < SYNC_THRESHOLD_MS:
                print("Frames are synchronized!")

                pose_data = pose_frame.get_pose_data()
                position = [pose_data.translation.x, pose_data.translation.y, pose_data.translation.z]
                orientation = [pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w]

                #TODO: Check this scaling
                # import pdb;pdb.set_trace()
                depth_image = (np.asanyarray(depth_frame.get_data()) * depth_scale * 1000.0).astype(np.uint16)
                color_image = np.asanyarray(color_frame.get_data())
                # pdb.set_trace()

                # saving rgb - png files
                rgb_filename = f"{save_dir}/rgb/{file_index:05d}.png"
                cv2.imwrite(rgb_filename, color_image)

                # saving depth - png files (16-bit png for raw depth values)
                depth_filename = f"{save_dir}/depth/{file_index:05d}.png"
                cv2.imwrite(depth_filename, depth_image)

                pose_filename = f"{save_dir}/pose/{file_index:05d}.npz"
                np.savez(pose_filename, position=position, orientation=orientation)

                print(f"Saved: {rgb_filename}, {depth_filename}, {pose_filename}")

                file_index += 1  

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                stacked_images = np.hstack((color_image, depth_colormap))
                cv2.imshow("RGB + Aligned Depth", stacked_images)

        key = cv2.waitKey(int(1000/frequency))
        if key & 0xFF == ord("q"):
            break

finally:
    T265_pipeline.stop()
    D4xx_pipeline.stop()
    cv2.destroyAllWindows()
    print("Streaming stopped.")

# Reference for alignment 
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

import pyrealsense2 as rs
import numpy as np
import cv2

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
#TODO: Enable fisheye frames

D4xx_config = rs.config()
D4xx_config.enable_device(D4xx_serial_num)
D4xx_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
D4xx_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

T265_pipeline.start(T265_config)
D4xx_pipeline.start(D4xx_config)

align = rs.align(rs.stream.color)

SYNC_THRESHOLD_MS = 20  # in millinsecs

define_intrinsics = False

try:
    while True:
        T265_frames = T265_pipeline.wait_for_frames()
        D4xx_frames = D4xx_pipeline.wait_for_frames()

        pose_frame = T265_frames.get_pose_frame()
        aligned_frames = align.process(D4xx_frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        #TODO: get depth scale, save it for actual depth
        if not define_intrinsics:
            define_intrinsics = True
            # Extract color camera intrinsics -- we aligned depth to color., so we need to use color cam instrinsics. 
            # If reverse, then depth ones
            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            fx, fy = intrinsics.fx, intrinsics.fy  # Focal lengths
            cx, cy = intrinsics.ppx, intrinsics.ppy  # Principal points

        if pose_frame and color_frame and depth_frame:
            pose_ts = pose_frame.get_timestamp()
            color_ts = color_frame.get_timestamp()
            depth_ts = depth_frame.get_timestamp()

            pose_color_diff = abs(pose_ts - color_ts)
            pose_depth_diff = abs(pose_ts - depth_ts)

            print(
                f"Pose TS: {pose_ts:.2f} ms, Color TS: {color_ts:.2f} ms, Depth TS: {depth_ts:.2f} ms"
            )
            print(
                f"Time Differences -> Pose-Color: {pose_color_diff:.2f} ms, Pose-Depth: {pose_depth_diff:.2f} ms"
            )

            if (
                pose_color_diff < SYNC_THRESHOLD_MS
                and pose_depth_diff < SYNC_THRESHOLD_MS
            ):
                print("Frames are synchronized!")

                # Process pose data
                pose_data = pose_frame.get_pose_data()
                position = pose_data.translation
                orientation = pose_data.rotation

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )

                stacked_images = np.hstack((color_image, depth_colormap))
                cv2.imshow("RGB + Aligned Depth", stacked_images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

finally:
    # Stop pipelines
    # TODO: Add file closings here latereon. Add saving intrinsics.
    T265_pipeline.stop()
    D4xx_pipeline.stop()
    cv2.destroyAllWindows()
    print("Streaming stopped.")

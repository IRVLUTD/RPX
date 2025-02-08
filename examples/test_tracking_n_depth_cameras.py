## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################
# - Modified by Sai Haneesh Allu, working at IRVL UTDallas -#


# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

from config.serial_nums import T265_serial_num, D4xx_serial_num

# Create a pipeline
T265_pipeline = None
D4xx_pipeline = None

T265_pipeline = rs.pipeline()
T265_config = rs.config()
T265_config.enable_device(T265_serial_num)
import pdb;pdb.set_trace()
# T265_config.enable_stream(rs.stream.fisheye, 848, 800, rs.format.y8)
T265_config.enable_stream(rs.stream.pose)

D4xx_pipeline = rs.pipeline()
D4xx_config = rs.config()
D4xx_config.enable_device(D4xx_serial_num)        
D4xx_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
D4xx_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# # Create a config and configure the pipeline to stream
# #  different resolutions of color and depth streams
# D4xx_config = rs.config()

# ctx = rs.context()
# print("Connected devices:")
# for i in range(len(ctx.devices)):
#     device = ctx.devices[i]
#     name = device.get_info(rs.camera_info.name)
#     serial_no = device.get_info(rs.camera_info.serial_number)
#     print(f"Device {i}: {name} - S/N: {serial_no}")
#     if "T265" in name:
#         print(f"creating T265 pipeline")
#         T265_pipeline = rs.pipeline()
#         T265_config = rs.config()

#         # T265_config.enable_device(serial_no)
#         # T265_config.enable_stream(rs.stream.fisheye, 848, 800, rs.format.y8)
#         # T265_config.enable_stream(rs.stream.pose)

#     if "D415" in name:
#         print(f"creating D415 pipeline")
#         D4xx_pipeline = rs.pipeline(ctx)
#         D4xx_config.enable_device(serial_no)        
#         # D4xx_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         # D4xx_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # pipeline = D4xx_pipeline
# # config = D4xx_config

"""
Start both the pipelines
"""
print(f"starting T265")
print(f"T265 pipeline {T265_pipeline}")
T265_pipeline.start(T265_config)
print(f"starting D4xx")
D4xx_pipeline.start(D4xx_config)

while True:
    

# # Start streaming
# profile = pipeline.start(config)

# # Getting the depth sensor's depth scale (see rs-align example for explanation)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# # We will be removing the background of objects more than
# #  clipping_distance_in_meters meters away
# clipping_distance_in_meters = 5 #1 meter
# clipping_distance = clipping_distance_in_meters / depth_scale

# # Create an align object
# # rs.align allows us to perform alignment of depth frames to others frames
# # The "align_to" is the stream type to which we plan to align depth frames.
# align_to = rs.stream.color
# align = rs.align(align_to)

# # Streaming loop
# try:
#     while True:
#         # Get frameset of color and depth
#         frames = pipeline.wait_for_frames()
#         # frames.get_depth_frame() is a 640x360 depth image

#         # Align the depth frame to color frame
#         aligned_frames = align.process(frames)

#         # Get aligned frames
#         aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
#         color_frame = aligned_frames.get_color_frame()

#         # Validate that both frames are valid
#         if not aligned_depth_frame or not color_frame:
#             continue

#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # Remove background - Set pixels further than clipping_distance to grey
#         grey_color = 153
#         depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
#         bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

#         # Render images:
#         #   depth align to color on left
#         #   depth on right
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#         images = np.hstack((bg_removed, depth_colormap))

#         cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
#         cv2.imshow('Align Example', images)
#         key = cv2.waitKey(1)
#         # Press esc or 'q' to close the image window
#         if key & 0xFF == ord('q') or key == 27:
#             cv2.destroyAllWindows()
#             break
# finally:
#     pipeline.stop()
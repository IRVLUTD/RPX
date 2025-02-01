#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.

# - Modified by Sai Haneesh Allu, working at IRVL UTDallas -#

#####################################################
##           librealsense T265 example             ##
#####################################################

# First import the library
import pyrealsense2 as rs
import time
import argparse
import numpy as np
import datetime
import pdb
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--save', action="store_true")
args = parser.parse_args()
print(args.save)
# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
# Build config object and request pose data
cfg = rs.config()

# Start streaming with requested config
pipe.start(cfg)


save_filename = f"./data/T265/{(datetime.datetime.now()).strftime('%c')}.npz"
xyz_array = []
WINDOW_TITLE = 'Realsense'
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
while True:
    try:
        frames = pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        data = pose.get_pose_data()
        fisheye_left = frames.get_fisheye_frame(1).as_video_frame()
        fisheye_right = frames.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(fisheye_left.get_data())
        right_data = np.asanyarray(fisheye_right.get_data())
        x = data.translation.x
        y = data.translation.y
        z = data.translation.z
        print(f"x: {x}, y: {y}, z: {z}")
        cv2.imshow(WINDOW_TITLE, np.hstack((left_data, right_data)))
        key = cv2.waitKey(1)
        if args.save:
            xyz_array.append([x,y,z])
        # time.sleep(1)
    except KeyboardInterrupt:
        break
if args.save:
    np.savez(save_filename, trajectory=xyz_array)
pipe.stop()
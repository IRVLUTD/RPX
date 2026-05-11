'''
code adapted from 
https://github.com/IntelRealSense/librealsense/tree/95a824a8d6ff49161fa28605994190de49e62f73/wrappers/python#examples
'''
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
WINDOW_TITLE = 'Realsense_4xx'
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

while True:
    try:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipe.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue
        # pdb.set_trace()
        depth_image =  np.asanyarray(depth.get_data())
        cv2.imshow(WINDOW_TITLE ,depth_image)
        cv2.waitKey(1)
        
    except KeyboardInterrupt:
        break
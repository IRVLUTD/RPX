# Running the example scripts

## Reset the vision devices
Run ```lsusb``` to check for the devices connected to the usb ports. Depth camera D4xx series bears the name ```Intel Corp. Intel(R) RealSense(TM) Depth Camera 4xx Intel(R) RealSense(TM) Depth Camera 4xx Intel(R) RealSense(TM) Depth Camera 4xx```

while T265 will have the name ```Intel Myriad VPU [Movidius Neural Compute Stick] Movidius Ltd. Movidius MA2X5X```

### A. Reset a specific device
```
python reset_vision_usb.py --name <device-name>
```

device-name ~ ```4xx``` for depth device. Replace 4xx with ```435``` or ```415``` depending on the camera being used.  ```Intel``` for T265

### B. Reset both 4xx and T265
```
python reset_vision_usb.py --name_list 4xx, Intel
```

## Testing 
To test the pose data and fisheye steam from T265. Exlude save argument if you dont want to save the xyz trajectory
```
python test_t265 --save
```

# Main Run 
run the following script to save the RGB, Depth, Pose, Fisheye 1,2
```
python save_device_data.py
```



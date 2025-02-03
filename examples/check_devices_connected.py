import pyrealsense2 as rs

ctx = rs.context()
print("Connected devices:")
for i in range(len(ctx.devices)):
    device = ctx.devices[i]
    print(f"Device {i}: {device.get_info(rs.camera_info.name)} - S/N: {device.get_info(rs.camera_info.serial_number)}")

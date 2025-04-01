import pyrealsense2 as rs


presets = {
            "Custom":0, 
            "Default":1, 
            "Hand":2, 
            "High Accuracy":3, 
            "High Density":4, 
            "Medium Density":5
} 

filters = {
            rs.decimation_filter(): [
                [rs.option.filter_magnitude, 2]
            ],
            rs.disparity_transform(True): [],
            rs.spatial_filter(): [
                [rs.option.filter_magnitude, 2],
                [rs.option.filter_smooth_alpha, 0.5],
                [rs.option.filter_smooth_delta, 20]
            ],
            rs.temporal_filter(): [
                [rs.option.filter_smooth_alpha, 0.4],
                [rs.option.filter_smooth_delta, 20],
                [rs.option.holes_fill, 2]
            ]
}
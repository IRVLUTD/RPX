# date: 2024-01-10
# description: Script to run mask refinement on all scenes in the specified directory.
# version: 1.0
# dependencies:
#   - Python 3.x
#   - maskgen_pipeline package
#   - Directory structure: /data/rpx/scene_id/{0,1,2}/
# usage: Run this script to refine masks for all scenes in the specified base directory.
# notes:
#   - Ensure the maskgen_pipeline package is installed and accessible in your Python environment.
#   - The script assumes the directory structure is consistent with scene IDs and subdirectories 0, 1, 2.
#   - The script runs the mask refinement process for each scene subdirectory
#     with an iteration count of 1.
# version: 1.0
# tags:
#   - mask refinement
#   - scene processing
#   - automation
# license: MIT License

#!/bin/bash

BASE_DIR="${RPX_DATA_ROOT:-/data/rpx}"


for scene in "$BASE_DIR"/*; do
    if [ -d "$scene" ]; then
        # Skip hidden or non-scene dirs like zip files
        for i in 0 1 2; do
            SUBDIR="$scene/$i"
            if [ -d "$SUBDIR" ]; then
                echo "Running on: $SUBDIR"
                python -m maskgen_pipeline.refine_masks --scene_dir "$SUBDIR" --iter 1
            fi
        done
    fi
done

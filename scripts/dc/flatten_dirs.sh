#!/bin/bash

root_dir=$1

# Loop through every object directory in rpx-data/
for obj_dir in "$root_dir"/*/; do
    sub_dir="${obj_dir}0/"

    # Check if the '0' subdirectory exists
    if [ -d "$sub_dir" ]; then
        echo "Processing $sub_dir"

        # Move all contents from '0/' to the object root dir
        if mv "$sub_dir"* "$obj_dir"; then
            echo "Moved contents of $sub_dir to $obj_dir"

            # If move was successful and '0/' is now empty, delete it
            if [ -z "$(ls -A "$sub_dir")" ]; then
                rmdir "$sub_dir"
                echo "Deleted empty $sub_dir"
            else
                echo "Warning: $sub_dir not empty, not deleting"
            fi
        else
            echo "Error moving contents of $sub_dir"
        fi
    fi
done

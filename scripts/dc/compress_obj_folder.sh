#!/bin/bash

root_dir=$1

# Loop through each object folder in rpx-data/
for obj_dir in "$root_dir"/*/; do
    # Remove trailing slash and extract the folder name
    obj_name=$(basename "$obj_dir")
    zip_file="${root_dir}/${obj_name}.zip"

    echo "Compressing $obj_name into $zip_file..."

    # Create the zip (overwrite if exists)
    zip -r "$zip_file" "$obj_dir"

    echo "Done compressing $obj_name"
done

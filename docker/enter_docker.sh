#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

# Get the optional directory argument
TARGET_DIR=$1

# If a directory is provided
if [ -n "$TARGET_DIR" ]; then
    docker exec -it rpx bash -c "cd $TARGET_DIR && exec bash"
else
    docker exec -it rpx bash
fi
docker rm -f rpx
DIR=$(pwd)/../
xhost +local:docker  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -itd --network=host --name rpx  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE -e ROS_MASTER_URI=$1 ROS_HOSTNAME=$2 irvlutd/rpx:latest bash
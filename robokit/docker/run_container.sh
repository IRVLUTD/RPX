# Source: https://github.com/NVlabs/BundleSDF/blob/master/docker/run_container.sh
# The BundleSDF docker image worked without the error of cuda mismatch after package installation. Hence using it.
# This will open the root folder so be careful and navigate to the project folder before dev

container_name="robokit"
docker rm -f $container_name
DIR=$(pwd)/../
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name $container_name  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE irvlutd/bundle-sdf:latest bash

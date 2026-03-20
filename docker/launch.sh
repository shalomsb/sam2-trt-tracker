#!/bin/bash
cd "${0%/*}"

function usage {
    echo "usage: ./docker/launch.sh [-b/-c/-d/-r] [--onnx]"
    echo "  -b  Build: setup models (export ONNX + build TRT engines)"
    echo "  -r  Run tracker (TRT default, --onnx for ONNX runtime)"
    echo "  -c  Compare: run TRT + PyTorch trackers and compare masks"
    echo "  -d  Develop: bash shell inside container"
}

ACTION=""
BACKEND=""

if [[ $# -lt 1 ]]; then usage && exit; fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -c | -d | -r ) ACTION=$1 ;;
        --onnx )             BACKEND="--onnx" ;;
        -h )                 usage ; exit ;;
        * )                  usage ; exit ;;
    esac
    shift
done

if [[ $ACTION == "" ]]; then usage && exit; fi

DOCKER_TAG=sam2-tracker
DOCKER_TAG_VERSION=v1.0
DOCKER_NAME=sam2-tracker

if [[ $ACTION == '-b' ]]; then
    time \
    docker build -f Dockerfile -t $DOCKER_TAG:$DOCKER_TAG_VERSION "$(pwd)/../" \
    || { echo "Docker build failed"; exit 1; }

    docker run --name ${DOCKER_NAME} \
        --rm --net=host --ipc=host --shm-size=4g --privileged -it \
        --runtime=nvidia --gpus all \
        -v "$(pwd)/../models":/models \
        -v "$(pwd)/../app":/app \
        -v "$(pwd)/scripts":/opt/scripts \
        -v "$(pwd)/entrypoint.sh":/opt/entrypoint.sh \
        --entrypoint /opt/entrypoint.sh \
        $DOCKER_TAG:$DOCKER_TAG_VERSION \
        $ACTION
    exit

elif [[ $ACTION == '-c' ]] || [[ $ACTION == '-r' ]] || [[ $ACTION == '-d' ]]; then
    docker run --name=${DOCKER_NAME} --rm \
        --net=host --ipc=host --shm-size=4g --privileged -it \
        --runtime=nvidia --gpus all \
        -v "$(pwd)/../models":/models \
        -v "$(pwd)/../app":/app \
        -v "$(pwd)/scripts":/opt/scripts \
        -v "$(pwd)/entrypoint.sh":/opt/entrypoint.sh \
        -v "$(pwd)/../streams":/streams \
        -v "$(pwd)/../output":/output \
        --entrypoint /opt/entrypoint.sh \
        $DOCKER_TAG:$DOCKER_TAG_VERSION \
        $ACTION $BACKEND
    exit

else
    echo "Invalid option"
    usage
    exit 1
fi

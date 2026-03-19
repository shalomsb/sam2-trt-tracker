#!/bin/bash
cd "${0%/*}"

function usage {
    echo "usage: ./docker/launch.sh [-b/-d/-r] [--onnx/--hybrid]"
    echo "  -b  Build: setup models (export ONNX + build TRT engines)"
    echo "  -r  Run tracker (TRT default, --onnx for ONNX, --hybrid for TRT+ORT)"
    echo "  -d  Develop: bash shell inside container"
}

ACTION=""
BACKEND=""

if [[ $# -lt 1 ]]; then usage && exit; fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -d | -r ) ACTION=$1 ;;
        --onnx )       BACKEND="--onnx" ;;
        --hybrid )     BACKEND="--hybrid" ;;
        -h )           usage ; exit ;;
        * )            usage ; exit ;;
    esac
    shift
done

if [[ $ACTION == "" ]]; then usage && exit; fi

DOCKER_TAG=sam2-tracker
DOCKER_TAG_VERSION=v1.0
DOCKER_NAME=sam2-tracker

if [[ $ACTION == '-b' ]]; then
    time \
    docker build -f Dockerfile -t $DOCKER_TAG:$DOCKER_TAG_VERSION "$(pwd)/../"

    docker run --name ${DOCKER_NAME} \
        --rm --net=host --ipc=host --shm-size=4g --privileged -it \
        --runtime=nvidia --gpus all \
        -v "$(pwd)/../models":/models \
        -v "$(pwd)/../app":/app \
        -v "$(pwd)/scripts":/opt/scripts \
        --entrypoint /opt/entrypoint.sh \
        $DOCKER_TAG:$DOCKER_TAG_VERSION \
        $ACTION
    exit

elif [[ $ACTION == '-r' ]] || [[ $ACTION == '-d' ]]; then
    docker run --name=${DOCKER_NAME} --rm \
        --net=host --ipc=host --shm-size=4g --privileged -it \
        --runtime=nvidia --gpus all \
        -v "$(pwd)/../models":/models \
        -v "$(pwd)/../app":/app \
        -v "$(pwd)/scripts":/opt/scripts \
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

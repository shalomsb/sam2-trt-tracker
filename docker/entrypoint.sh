#!/usr/bin/env bash
set -e

function usage {
    echo "usage: entrypoint.sh [-b/-r/-c/-d] [--onnx/--hybrid]"
    echo "  -b  Build: export ONNX models + build TRT engines"
    echo "  -r  Run tracker (TRT default, --onnx for ONNX, --hybrid for TRT+ORT)"
    echo "  -c  Compare: run hybrid + PyTorch trackers, compare masks via IoU"
    echo "  -d  Develop: bash shell"
}

ACTION=""
BACKEND="trt"

if [[ $# -lt 1 ]]; then usage && exit; fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -r | -c | -d ) ACTION=$1 ;;
        --onnx )       BACKEND="onnx" ;;
        --hybrid )     BACKEND="hybrid" ;;
        -h )           usage && exit ;;
        * )            usage && exit ;;
    esac
    shift
done

if [[ $ACTION == '-b' ]]; then
    echo "=== Step 1/2: Setup models (clone, patch, export ONNX) ==="
    bash /opt/scripts/setup_models.sh
    echo ""
    echo "=== Step 2/2: Build TRT engines ==="
    bash /opt/scripts/build_trt_engines.sh
elif [[ $ACTION == '-r' ]]; then
    if [[ $BACKEND == "onnx" ]]; then
        bash /opt/scripts/run_onnx_tracker.sh
    elif [[ $BACKEND == "hybrid" ]]; then
        bash /opt/scripts/run_hybrid_tracker.sh
    else
        bash /opt/scripts/run_trt_tracker.sh
    fi
elif [[ $ACTION == '-c' ]]; then
    bash /opt/scripts/run_compare.sh
elif [[ $ACTION == '-d' ]]; then
    /bin/bash
else
    usage && exit
fi

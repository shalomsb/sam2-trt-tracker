#!/bin/bash
set -e
START=$SECONDS

VIDEO="${VIDEO:-/streams/sample.mp4}"
POINT="${POINT:-640,360}"
BBOX="${BBOX:-}"
OUTPUT="${OUTPUT:-/output/output_tracked_hybrid.mp4}"
MODEL_DIR="${MODEL_DIR:-/models/sam2/tiny/trt/}"
ONNX_DIR="${ONNX_DIR:-/models/sam2/tiny/onnx/}"

PROMPT_ARG=""
if [[ -n "$BBOX" ]]; then
    PROMPT_ARG="--bbox $BBOX"
else
    PROMPT_ARG="--point $POINT"
fi

python /app/sam2_hybrid_video_tracker.py \
    --video "$VIDEO" \
    $PROMPT_ARG \
    --model-dir "$MODEL_DIR" \
    --onnx-dir "$ONNX_DIR" \
    --output "$OUTPUT"

echo "Total time: $((SECONDS - START))s"

#!/bin/bash
set -e
START=$SECONDS

VIDEO="${VIDEO:-/streams/sample.mp4}"
POINT="${POINT:-640,360}"
BBOX="${BBOX:-}"
OUTPUT="${OUTPUT:-/output/output_tracked_hybrid.mp4}"
MODEL_DIR="${MODEL_DIR:-/models/sam2/tiny/trt/}"
ONNX_DIR="${ONNX_DIR:-/models/sam2/tiny/onnx/}"
SAVE_BBOXES="${SAVE_BBOXES:-/output/bboxes.csv}"

PROMPT_ARG=""
if [[ -n "$BBOX" ]]; then
    PROMPT_ARG="--bbox $BBOX"
else
    PROMPT_ARG="--point $POINT"
fi

MODE="${1:-}"
BBOX_ARGS=""
if [[ "$MODE" == "bbox" ]]; then
    BBOX_ARGS="--bbox-only --save-bboxes $SAVE_BBOXES"
fi

python /app/sam2_hybrid_video_tracker.py \
    --video "$VIDEO" \
    $PROMPT_ARG \
    --model-dir "$MODEL_DIR" \
    --onnx-dir "$ONNX_DIR" \
    --output "$OUTPUT" \
    $BBOX_ARGS

echo "Total time: $((SECONDS - START))s"

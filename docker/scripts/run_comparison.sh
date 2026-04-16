#!/bin/bash
set -e
START=$SECONDS

VIDEO="${VIDEO:-/streams/sample.mp4}"
POINT="${POINT:-640,360}"
BBOX="${BBOX:-}"
MODEL_DIR_TRT="${MODEL_DIR_TRT:-/models/edgetam/trt/}"
CHECKPOINT="${CHECKPOINT:-/models/edgetam/edgetam.pt}"

DURATION="${DURATION:-15}"
MASKS_TRT="/output/masks_trt"
MASKS_PT="/output/masks_pytorch"

# Trim video to keep GPU memory manageable
TRIMMED="/tmp/comparison_clip.mp4"
echo "Trimming video to ${DURATION}s..."
ffmpeg -y -i "$VIDEO" -t "$DURATION" -an "$TRIMMED" 2>/dev/null
VIDEO="$TRIMMED"

PROMPT_ARG=""
if [[ -n "$BBOX" ]]; then
    PROMPT_ARG="--bbox $BBOX"
else
    PROMPT_ARG="--point $POINT"
fi

# Clean old results
rm -rf "$MASKS_TRT" "$MASKS_PT" \
    /output/iou_diff_trt_vs_pt /output/iou_overlay_trt_vs_pt

echo "============================================"
echo "  EdgeTAM — TRT vs PyTorch Comparison"
echo "============================================"

# Step 1: Run TRT tracker
echo ""
echo ">>> [1/3] Running TRT tracker..."
python /app/edgetam_trt_video_tracker.py \
    --video "$VIDEO" \
    $PROMPT_ARG \
    --model-dir "$MODEL_DIR_TRT" \
    --output /output/output_tracked_trt.mp4 \
    --save-masks "$MASKS_TRT"

# Step 2: Run PyTorch tracker
echo ""
echo ">>> [2/3] Running PyTorch reference tracker..."
python /app/edgetam_pytorch_video_tracker.py \
    --video "$VIDEO" \
    $PROMPT_ARG \
    --checkpoint "$CHECKPOINT" \
    --save-masks "$MASKS_PT"

# Step 3: Compare TRT vs PyTorch
echo ""
echo ">>> [3/3] Comparing TRT vs PyTorch..."
python /app/compare_masks.py \
    "$MASKS_TRT" "$MASKS_PT" \
    --label-a "TRT" \
    --label-b "PyTorch" \
    --csv /output/iou_trt_vs_pytorch.csv \
    --viz-dir /output/iou_diff_trt_vs_pt \
    --overlay-dir /output/iou_overlay_trt_vs_pt

echo ""
echo "Total comparison time: $((SECONDS - START))s"

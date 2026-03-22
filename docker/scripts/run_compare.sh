#!/bin/bash
set -e

VIDEO="${VIDEO:-/streams/sample.mp4}"
POINT="${POINT:-640,360}"
BBOX="${BBOX:-}"
MODEL_DIR="${MODEL_DIR:-/models/sam2/tiny/trt/}"
ONNX_DIR="${ONNX_DIR:-/models/sam2/tiny/onnx/}"
CHECKPOINT="${CHECKPOINT:-/models/sam2/tiny/sam2.1_hiera_tiny.pt}"
MAX_FRAMES="${MAX_FRAMES:-350}"

MASKS_HYBRID="/output/masks_hybrid"
MASKS_TORCH="/output/masks_torch"
CSV_OUT="/output/compare_iou.csv"
OVERLAY_DIR="/output/compare_overlays"

PROMPT_ARG=""
if [[ -n "$BBOX" ]]; then
    PROMPT_ARG="--bbox $BBOX"
else
    PROMPT_ARG="--point $POINT"
fi

echo "============================================"
echo "  Mask Comparison: Hybrid vs PyTorch"
echo "============================================"

# Download checkpoint if not present
if [[ ! -f "$CHECKPOINT" ]]; then
    echo ">>> Downloading SAM2.1 Hiera Tiny checkpoint..."
    mkdir -p "$(dirname "$CHECKPOINT")"
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
        -O "$CHECKPOINT"
fi

# Trim video to MAX_FRAMES for both trackers to use the exact same input
TRIMMED_VIDEO="/output/compare_input.mp4"
echo ">>> Trimming video to $MAX_FRAMES frames..."
ffmpeg -y -i "$VIDEO" -frames:v "$MAX_FRAMES" -c:v copy -an "$TRIMMED_VIDEO" 2>/dev/null
VIDEO="$TRIMMED_VIDEO"

# Clean previous masks
rm -rf "$MASKS_HYBRID" "$MASKS_TORCH" "$OVERLAY_DIR" "$CSV_OUT" /output/compare_overlay_video.mp4

# Step 1: Run hybrid tracker with mask saving
echo ""
echo ">>> [1/3] Running hybrid tracker..."
python /app/sam2_hybrid_video_tracker.py \
    --video "$VIDEO" \
    $PROMPT_ARG \
    --model-dir "$MODEL_DIR" \
    --onnx-dir "$ONNX_DIR" \
    --output /output/output_tracked_hybrid.mp4 \
    --save-masks "$MASKS_HYBRID"

# Step 2: Run PyTorch reference tracker
echo ""
echo ">>> [2/3] Running PyTorch reference tracker..."
python /app/sam2_torch_video_tracker.py \
    --video "$VIDEO" \
    $PROMPT_ARG \
    --checkpoint "$CHECKPOINT" \
    --save-masks "$MASKS_TORCH"

# Step 3: Compare masks
echo ""
echo ">>> [3/4] Comparing masks..."
python /app/compare_masks.py \
    "$MASKS_HYBRID" "$MASKS_TORCH" \
    --label-a "Hybrid" \
    --label-b "PyTorch" \
    --skip-first 5 \
    --csv "$CSV_OUT" \
    --overlay-dir "$OVERLAY_DIR"

# Step 4: Overlay comparison video
echo ""
echo ">>> [4/4] Generating overlay video..."
OVERLAY_VIDEO="/output/compare_overlay_video.mp4"
python /app/overlay_compare_video.py \
    --video "$VIDEO" \
    --masks-a "$MASKS_HYBRID" \
    --masks-b "$MASKS_TORCH" \
    --label-a "Hybrid" \
    --label-b "PyTorch" \
    --output "$OVERLAY_VIDEO"

echo ""
echo "Results:"
echo "  CSV:      $CSV_OUT"
echo "  Overlays: $OVERLAY_DIR"
echo "  Video:    $OVERLAY_VIDEO"

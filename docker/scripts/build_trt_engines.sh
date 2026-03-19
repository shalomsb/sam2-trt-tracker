#!/bin/bash
set -e

ONNX_DIR="/models/sam2/tiny/onnx"
TRT_DIR="/models/sam2/tiny/trt"

mkdir -p "$TRT_DIR"

echo "============================================"
echo "  Building TRT engines (FP16) from ONNX"
echo "  Source: $ONNX_DIR"
echo "  Output: $TRT_DIR"
echo "============================================"

echo "[1/4] image_encoder"
if [[ ! -f "$TRT_DIR/image_encoder.engine" ]]; then
    trtexec --onnx=$ONNX_DIR/image_encoder.onnx \
            --saveEngine=$TRT_DIR/image_encoder.engine \
            --fp16 \
            --memPoolSize=workspace:4096
else
    echo "    Already exists — skipping"
fi

echo "[2/4] mask_decoder"
if [[ ! -f "$TRT_DIR/mask_decoder.engine" ]]; then
    trtexec --onnx=$ONNX_DIR/mask_decoder.onnx \
            --saveEngine=$TRT_DIR/mask_decoder.engine \
            --fp16
else
    echo "    Already exists — skipping"
fi

echo "[3/4] memory_encoder"
if [[ ! -f "$TRT_DIR/memory_encoder.engine" ]]; then
    trtexec --onnx=$ONNX_DIR/memory_encoder.onnx \
            --saveEngine=$TRT_DIR/memory_encoder.engine \
            --fp16 \
            --minShapes=mask_for_mem:1x1x1024x1024,occ_logit:1x1 \
            --optShapes=mask_for_mem:1x1x1024x1024,occ_logit:1x1 \
            --maxShapes=mask_for_mem:1x1x1024x1024,occ_logit:1x1
else
    echo "    Already exists — skipping"
fi

echo "[4/4] memory_attention (static NUM_MASKMEM=3)"
if [[ ! -f "$TRT_DIR/memory_attention.engine" ]]; then
    trtexec --onnx=$ONNX_DIR/memory_attention.onnx \
            --saveEngine=$TRT_DIR/memory_attention.engine \
            --fp16 \
            --builderOptimizationLevel=5 \
            --minShapes=memory_0:1x3x256,memory_1:1x3x64x64x64,memory_pos_embed:1x12300x64 \
            --optShapes=memory_0:1x3x256,memory_1:1x3x64x64x64,memory_pos_embed:1x12300x64 \
            --maxShapes=memory_0:1x3x256,memory_1:1x3x64x64x64,memory_pos_embed:1x12300x64
else
    echo "    Already exists — skipping"
fi

echo ""
echo "Done — all engines saved to $TRT_DIR/"
ls -lh "$TRT_DIR/"

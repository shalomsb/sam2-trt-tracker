#!/bin/bash
set -e

ONNX_DIR="/models/edgetam/onnx"
TRT_DIR="/models/edgetam/trt"

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
            --fp16
else
    echo "    Already exists — skipping"
fi

echo "[2/4] mask_decoder"
if [[ ! -f "$TRT_DIR/mask_decoder.engine" ]]; then
    trtexec --onnx=$ONNX_DIR/mask_decoder.onnx \
            --saveEngine=$TRT_DIR/mask_decoder.engine \
            --fp16 \
            --minShapes=point_coords:1x1x2,point_labels:1x1,image_embed:1x256x64x64 \
            --optShapes=point_coords:1x2x2,point_labels:1x2,image_embed:1x256x64x64 \
            --maxShapes=point_coords:1x2x2,point_labels:1x2,image_embed:1x256x64x64
else
    echo "    Already exists — skipping"
fi

# EdgeTAM: memory_encoder may not have occ_logit input (optimized away)
# Check if occ_logit is an input in the ONNX model
echo "[3/4] memory_encoder"
if [[ ! -f "$TRT_DIR/memory_encoder.engine" ]]; then
    if python3 -c "import onnx; m=onnx.load('$ONNX_DIR/memory_encoder.onnx'); print([i.name for i in m.graph.input])" 2>/dev/null | grep -q "occ_logit"; then
        trtexec --onnx=$ONNX_DIR/memory_encoder.onnx \
                --saveEngine=$TRT_DIR/memory_encoder.engine \
                --fp16 \
                --minShapes=mask_for_mem:1x1x1024x1024,occ_logit:1x1 \
                --optShapes=mask_for_mem:1x1x1024x1024,occ_logit:1x1 \
                --maxShapes=mask_for_mem:1x1x1024x1024,occ_logit:1x1
    else
        trtexec --onnx=$ONNX_DIR/memory_encoder.onnx \
                --saveEngine=$TRT_DIR/memory_encoder.engine \
                --fp16
    fi
else
    echo "    Already exists — skipping"
fi

# EdgeTAM memory attention:
# - memory_1 is [1, N, 512, 64] (perceiver output, not [1, N, 64, 64, 64])
# - memory_pos_embed: N*512 + N*4 tokens (not N*4096 + N*4)
# - With NUM_MASKMEM=3: 3*512 + 3*4 = 1548
# - num_obj_ptr may be folded as constant (check ONNX)
# EdgeTAM memory attention — dynamic shapes (1-7 memories)
# NUM_MASKMEM=7 to match temporal_code size; mem_tokens=512 (perceiver output)
# memory_pos_embed tokens = N*512 + N*4 (N masks * 512 spatial + N ptrs * 4 split)
echo "[4/4] memory_attention (dynamic NUM_MASKMEM=1..7)"
if [[ ! -f "$TRT_DIR/memory_attention.engine" ]]; then
    trtexec --onnx=$ONNX_DIR/memory_attention.onnx \
            --saveEngine=$TRT_DIR/memory_attention.engine \
            --fp16 \
            --minShapes=memory_0:1x1x256,memory_1:1x1x512x64,memory_pos_embed:1x516x64 \
            --optShapes=memory_0:1x7x256,memory_1:1x7x512x64,memory_pos_embed:1x3612x64 \
            --maxShapes=memory_0:1x7x256,memory_1:1x7x512x64,memory_pos_embed:1x3612x64
else
    echo "    Already exists — skipping"
fi

echo ""
echo "Done — all engines saved to $TRT_DIR/"
ls -lh "$TRT_DIR/"
